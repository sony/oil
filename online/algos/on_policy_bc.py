import numpy as np
import torch
import torch.nn.functional as F
import warnings
from typing import Dict
from gymnasium import spaces
from typing import Type, Union, Optional
from typing import Any, Dict, Optional, Type, Union
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from typing import Any, Dict, Optional, Type, Union, Tuple
from torch.nn import functional as F
from online.algos.buffers import OracleRolloutBuffer, OracleEpisodeRolloutBuffer
from online.policies.actor import ActorPolicy


class OnPolicyBC(OnPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[ActorPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = OracleRolloutBuffer,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=1.0,
            gae_lambda=1.0,
            ent_coef=ent_coef,
            vf_coef=None,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps

            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        if _init_setup_model:
            self._setup_model()

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer: OracleRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            oracle_actions = env.env_method("get_oracle_action")
            action_lengths = [arr.shape[0] for arr in oracle_actions[:-1]]
            split_indices = np.cumsum(action_lengths)
            clipped_actions = np.split(clipped_actions, split_indices, axis=0)

            if oracle_actions[0].shape == self.action_space.shape:
                oracle_actions = np.stack(oracle_actions)
            else:
                oracle_actions = np.concatenate(oracle_actions)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                torch.zeros(1),  # value predictions are not used
                log_probs,
                oracle_actions,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        entropy_losses = []
        imitation_losses = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                expert_actions = rollout_data.oracle_actions

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                    expert_actions = rollout_data.expert_actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # Compute pred and several hacks to deal with the different shapes
                actions_pred, _ = self.policy(rollout_data.observations)
                actions = actions.reshape(actions_pred.shape)
                actions_pred = actions_pred.reshape(expert_actions.shape)

                # Imitation loss
                if isinstance(self.action_space, spaces.Discrete):
                    imitation_loss = F.cross_entropy(actions_pred, expert_actions)
                else:
                    imitation_loss = F.mse_loss(actions_pred, expert_actions)
                imitation_losses.append(imitation_loss.item())

                # Entropy loss favor exploration
                log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = imitation_loss + self.ent_coef * entropy_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/imitation_loss", np.mean(imitation_losses))
        self.logger.record("train/loss", loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", torch.exp(self.policy.log_std).mean().item()
            )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


class OnPolicyTransformerBC(OnPolicyBC):
    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer: OracleEpisodeRolloutBuffer,
        n_rollout_steps: int,  # In the transformer version, this is the number of episodes
    ) -> bool:

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        n_steps = 0
        for _ in range(n_rollout_steps):
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            episode_log_probs = []
            episode_oracle_actions = []

            done = False
            while not done:
                if (
                    self.use_sde
                    and self.sde_sample_freq > 0
                    and n_steps % self.sde_sample_freq == 0
                ):
                    self.policy.reset_noise(env.num_envs)

                episode_obs.append(self._last_obs)
                with torch.no_grad():
                    obs_tensor = torch.tensor(
                        np.stack(episode_obs, axis=0), device=self.device
                    ).permute(1, 0, 2)
                    actions, log_probs = self.policy(obs_tensor, single_action=True)
                actions = actions.cpu().numpy()
                clipped_actions = actions

                if isinstance(self.action_space, spaces.Box):
                    if self.policy.squash_output:
                        clipped_actions = self.policy.unscale_action(clipped_actions)
                    else:
                        clipped_actions = np.clip(
                            actions, self.action_space.low, self.action_space.high
                        )
                new_obs, rewards, dones, infos = env.step(clipped_actions)
                oracle_actions = np.stack(env.env_method("get_oracle_action"))
                self.num_timesteps += env.num_envs

                callback.update_locals(locals())
                if not callback.on_step():
                    return False

                self._update_info_buffer(infos, dones)

                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.reshape(-1, 1)

                episode_actions.append(actions)
                episode_rewards.append(rewards)
                episode_dones.append(self._last_episode_starts)
                episode_log_probs.append(log_probs)
                episode_oracle_actions.append(oracle_actions)

                self._last_obs = new_obs
                self._last_episode_starts = dones
                done = any(dones)
                n_steps += 1

                if done:
                    assert all(
                        dones
                    ), "Assuming all the environments have the same duration"

            # Convert to tensor
            episode_obs = torch.tensor(
                np.stack(episode_obs, axis=0), device=self.device
            ).permute(1, 0, 2)
            episode_actions = torch.tensor(
                np.stack(episode_actions, axis=0), device=self.device
            ).permute(1, 0, 2)
            episode_rewards = torch.tensor(
                np.stack(episode_rewards, axis=0), device=self.device
            ).permute(1, 0)
            episode_dones = torch.tensor(
                np.stack(episode_dones, axis=0), device=self.device
            ).permute(1, 0)
            episode_log_probs = (
                torch.stack(episode_log_probs, axis=0).to(self.device).permute(1, 0)
            )
            episode_oracle_actions = torch.tensor(
                np.stack(episode_oracle_actions, axis=0), device=self.device
            ).permute(1, 0, 2)
            rollout_buffer.add(
                episode_obs,
                episode_actions,
                episode_rewards,
                episode_dones,
                torch.zeros(1),
                episode_log_probs,
                episode_oracle_actions,
            )
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        single_action: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(
            observation, state, episode_start, deterministic, single_action
        )
