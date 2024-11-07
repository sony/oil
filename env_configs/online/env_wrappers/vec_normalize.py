import numpy as np
from typing import Union, Dict
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from stable_baselines3.common.running_mean_std import RunningMeanStd


class BatchVecNormalize(VecNormalize):
    """
    Same as VecNormalize, but flattens the observations before updating the mean and std.
    """
    
    def step_wait(self) -> VecEnvStepReturn:
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, dones)

        where ``dones`` is a boolean vector indicating whether each element is new.
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        assert isinstance(obs, (np.ndarray, dict))  # for mypy
        self.old_obs = obs
        self.old_reward = rewards

        if self.training and self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key].reshape(-1, obs[key].shape[-1]))
            else:
                self.obs_rms.update(obs.reshape(-1, obs.shape[-1]))

        obs = self.normalize_obs(obs)

        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)

        # Normalize the terminal observations
        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self.normalize_obs(infos[idx]["terminal_observation"])

        self.returns[dones] = 0
        return obs, rewards, dones, infos

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reset all environments
        :return: first observation of the episode
        """
        obs = self.venv.reset()
        assert isinstance(obs, (np.ndarray, dict))
        self.old_obs = obs
        self.returns = np.zeros(self.num_envs)
        if self.training and self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key].reshape(-1, obs[key].shape[-1]))
            else:
                assert isinstance(self.obs_rms, RunningMeanStd)
                self.obs_rms.update(obs.reshape(-1, obs.shape[-1]))
        return self.normalize_obs(obs)