import numpy as np
import logging
from imitation.algorithms.dagger import DAggerTrainer
from imitation.data.rollout import (
    make_sample_until,
    GenTrajTerminationFn,
    TrajectoryAccumulator,
    flatten_trajectories,
)
from collections import deque
from typing import Sequence
from imitation.data import types
from gymnasium import spaces


class OracleDaggerTrainer(DAggerTrainer):
    def __init__(
        self,
        venv,
        scratch_dir,
        rng,
        max_stored_trajs=1_000,
        beta_schedule=None,
        **dagger_trainer_kwargs,
    ):
        super().__init__(
            venv=venv,
            scratch_dir=scratch_dir,
            rng=rng,
            beta_schedule=beta_schedule,
            **dagger_trainer_kwargs,
        )
        self.expert_trajs = deque(maxlen=max_stored_trajs)

    def train(
        self,
        total_timesteps,
        save_every=10,
        *,
        rollout_round_min_episodes=3,
        rollout_round_min_timesteps=500,
        bc_train_kwargs=None,
    ):
        total_timestep_count = 0
        round_num = 0
        while total_timestep_count < total_timesteps:
            beta = self.beta_schedule(self.round_num)
            round_episode_count = 0
            round_timestep_count = 0

            sample_until = make_sample_until(
                min_timesteps=max(rollout_round_min_timesteps, self.batch_size),
                min_episodes=rollout_round_min_episodes,
            )

            trajectories = self.generate_oracle_trajectories(
                sample_until=sample_until, beta=beta
            )

            for traj in trajectories:
                self._logger.record(
                    "dagger/episode_reward",
                    np.sum(traj.rews),
                )
                self._logger.record("dagger/mean_action", np.mean(traj.acts))

                round_timestep_count += len(traj)
                total_timestep_count += len(traj)

            round_episode_count += len(trajectories)

            self._logger.record("dagger/total_timesteps", total_timestep_count)
            self._logger.record("dagger/round_num", round_num)
            self._logger.record("dagger/round_episode_count", round_episode_count)
            self._logger.record("dagger/round_timestep_count", round_timestep_count)
            for traj in trajectories:
                info = traj.infos[-1]  # Info of the last step
                self._logger.record("dagger/conversions", info.get("conversions", 0))
                self._logger.record("dagger/cost", info.get("cost", 0))
                self._logger.record("dagger/cpa", info.get("cpa", 0))
                self._logger.record("dagger/target_cpa", info.get("target_cpa", 0))
                self._logger.record("dagger/budget", info.get("budget", 0))
                self._logger.record(
                    "dagger/score_over_pvalue", info.get("score_over_pvalue", 0)
                )
                self._logger.record(
                    "dagger/score_over_budget", info.get("score_over_budget", 0)
                )
                self._logger.record(
                    "dagger/score_over_cpa", info.get("score_over_cpa", 0)
                )
                self._logger.record(
                    "dagger/cost_over_budget", info.get("cost_over_budget", 0)
                )
                self._logger.record(
                    "dagger/target_cpa_over_cpa", info.get("target_cpa_over_cpa", 0)
                )
                self._logger.record("dagger/score", info.get("score", 0))
                self._logger.record("dagger/beta", beta)

            self.extend_and_update(trajectories, bc_train_kwargs)
            round_num += 1
            logging.info(
                f"Round {round_num} complete. Total timesteps: {total_timestep_count}."
            )
            # logging.info(f"Vecnormalize mean, var, count:", self.venv.obs_rms.mean, self.venv.obs_rms.var, self.venv.obs_rms.count)
            if round_num % save_every == 0:
                logging.info(f"Saving trainer at round {round_num}")
                self.save_trainer()

    def generate_oracle_trajectories(
        self,
        sample_until: GenTrajTerminationFn,
        beta: float = 0.0,
    ) -> Sequence[types.TrajectoryWithRew]:

        # Collect rollout tuples.
        trajectories = []
        # accumulator for incomplete trajectories
        trajectories_accum = TrajectoryAccumulator()
        obs = self.venv.reset()
        wrapped_obs = types.maybe_wrap_in_dictobs(obs)

        # we use dictobs to iterate over the envs in a vecenv
        for env_idx, ob in enumerate(wrapped_obs):
            # Seed with first obs only. Inside loop, we'll only add second obs from
            # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
            # get all observations, but they're not duplicated into "next obs" and
            # "previous obs" (this matters for, e.g., Atari, where observations are
            # really big).
            trajectories_accum.add_step(dict(obs=ob), env_idx)

        # Now, we sample until `sample_until(trajectories)` is true.
        # If we just stopped then this would introduce a bias towards shorter episodes,
        # since longer episodes are more likely to still be active, i.e. in the process
        # of being sampled from. To avoid this, we continue sampling until all epsiodes
        # are complete.
        #
        # To start with, all environments are active.
        active = np.ones(self.venv.num_envs, dtype=bool)
        state = None
        dones = np.ones(self.venv.num_envs, dtype=bool)

        while np.any(active):
            # policy gets unwrapped observations (eg as dict, not dictobs)
            dagger_acts, state = self.bc_trainer.policy.predict(
                obs, state=state, episode_start=dones
            )
            oracle_acts = np.stack(self.venv.env_method("get_oracle_action"))
            mask = self.rng.uniform(0, 1, size=(self.venv.num_envs,)) < beta
            dagger_acts[mask] = oracle_acts[mask]
            obs, rews, dones, infos = self.venv.step(dagger_acts)
            wrapped_obs = types.maybe_wrap_in_dictobs(obs)

            # If an environment is inactive, i.e. the episode completed for that
            # environment after `sample_until(trajectories)` was true, then we do
            # *not* want to add any subsequent trajectories from it. We avoid this
            # by just making it never done.
            dones &= active

            new_trajs = trajectories_accum.add_steps_and_auto_finish(
                oracle_acts,
                wrapped_obs,
                rews,
                dones,
                infos,
            )
            trajectories.extend(new_trajs)

            if sample_until(trajectories):
                # Termination condition has been reached. Mark as inactive any
                # environments where a trajectory was completed this timestep.
                active &= ~dones
        # Note that we just drop partial trajectories. This is not ideal for some
        # algos; e.g. BC can probably benefit from partial trajectories, too.

        # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
        # `trajectories` sooner. Shuffle to avoid bias in order. This is important
        # when callees end up truncating the number of trajectories or transitions.
        # It is also cheap, since we're just shuffling pointers.
        self.rng.shuffle(trajectories)  # type: ignore[arg-type]

        # Sanity checks.
        for trajectory in trajectories:
            n_steps = len(trajectory.acts)
            # extra 1 for the end
            if isinstance(self.venv.observation_space, spaces.Dict):
                exp_obs = {}
                for k, v in self.venv.observation_space.items():
                    assert v.shape is not None
                    exp_obs[k] = (n_steps + 1,) + v.shape
            else:
                obs_space_shape = self.venv.observation_space.shape
                assert obs_space_shape is not None
                exp_obs = (n_steps + 1,) + obs_space_shape  # type: ignore[assignment]
            real_obs = trajectory.obs.shape
            assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
            assert self.venv.action_space.shape is not None
            exp_act = (n_steps,) + self.venv.action_space.shape
            real_act = trajectory.acts.shape
            assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
            exp_rew = (n_steps,)
            real_rew = trajectory.rews.shape
            assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"
        return trajectories

    def extend_and_update(
        self,
        trajectories: Sequence[types.TrajectoryWithRew],
        bc_train_kwargs=None,
    ):
        if bc_train_kwargs is None:
            bc_train_kwargs = {}
        else:
            bc_train_kwargs = dict(bc_train_kwargs)

        user_keys = bc_train_kwargs.keys()

        if "n_epochs" not in user_keys and "n_batches" not in user_keys:
            bc_train_kwargs["n_epochs"] = self.DEFAULT_N_EPOCHS

        self.expert_trajs.extend(trajectories)

        transitions = flatten_trajectories(self.expert_trajs)

        logging.info(
            f"Training at round {self.round_num} with {len(transitions)} transitions"
        )
        self.bc_trainer.set_demonstrations(transitions)
        self.bc_trainer.train(**bc_train_kwargs)
        logging.info(f"Training at round {self.round_num} complete")
        self.round_num += 1
        return self.round_num

    # def save_trainer(self):
    #     super().save_trainer()
    #     vecnormalize_path = self.scratch_dir / f"checkpoint-{self.round_num:03d}_vecnormalize.pkl"
    #     self.venv.save(vecnormalize_path)
