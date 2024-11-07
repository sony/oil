import numpy as np
import torch
from typing import NamedTuple
from stable_baselines3.common.buffers import RolloutBuffer
from typing import Union, Optional
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer


class OracleRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    oracle_actions: torch.Tensor


class OracleRolloutBuffer(RolloutBuffer):
    """Rollout buffer that also stores the expert actions."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        batch_state_subsample: Optional[int] = None,
    ):
        self.batch_state_subsample = batch_state_subsample
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs,
        )
        if batch_state_subsample is not None:
            assert (
                self.action_dim == 1
            ), "Batch state subsample only works with 1D actions"
            assert isinstance(
                self.observation_space, spaces.Box
            ), "Batch state subsample only works with Box observations"
            self.action_dim = batch_state_subsample
            self.obs_shape = (batch_state_subsample, *self.obs_shape)

    def reset(self):
        self.expert_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        super().reset()

    def add(self, obs, action, reward, episode_start, value, log_prob, expert_action):
        if self.batch_state_subsample is not None:
            num_elements = action.shape[1]
            subsample_indices = np.random.choice(
                num_elements, self.batch_state_subsample * self.n_envs, replace=True
            )
            obs = obs[subsample_indices]
            action = action[subsample_indices]
            expert_action = expert_action[subsample_indices]
            log_prob = log_prob[subsample_indices].mean()
        obs = obs.reshape((self.n_envs, *self.obs_shape))
        action = action.reshape((self.n_envs, self.action_dim))
        expert_action = expert_action.reshape((self.n_envs, self.action_dim))
        self.expert_actions[self.pos] = np.array(expert_action)
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size=None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "expert_actions",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env=None,
    ):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.expert_actions[batch_inds],
        )
        return OracleRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class OracleEpisodeRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        ep_len: int = 48,
    ):
        self.ep_len = ep_len
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs,
        )

    def reset(self):
        self.expert_actions = torch.zeros(
            (self.buffer_size, self.n_envs, self.ep_len, self.action_dim),
            dtype=torch.float32,
        )
        self.observations = torch.zeros(
            (self.buffer_size, self.n_envs, self.ep_len, *self.obs_shape),
            dtype=torch.float32,
        )
        self.actions = torch.zeros(
            (self.buffer_size, self.n_envs, self.ep_len, self.action_dim),
            dtype=torch.float32,
        )
        self.rewards = torch.zeros(
            (self.buffer_size, self.n_envs, self.ep_len), dtype=torch.float32
        )
        self.returns = torch.zeros(
            (self.buffer_size, self.n_envs, self.ep_len), dtype=torch.float32
        )
        self.episode_starts = torch.zeros(
            (self.buffer_size, self.n_envs, self.ep_len), dtype=torch.float32
        )
        self.values = torch.zeros(
            (self.buffer_size, self.n_envs, self.ep_len), dtype=torch.float32
        )
        self.log_probs = torch.zeros(
            (self.buffer_size, self.n_envs, self.ep_len), dtype=torch.float32
        )
        self.advantages = torch.zeros(
            (
                self.buffer_size,
                self.n_envs,
                self.ep_len,
            ),
            dtype=torch.float32,
        )
        self.generator_ready = False
        BaseBuffer.reset(self)

    def compute_returns_and_advantage(
        self, last_values: torch.Tensor, dones: np.ndarray
    ) -> None:
        raise NotImplementedError(
            "This method should not be called for this buffer type"
        )

    def add(self, obs, action, reward, episode_start, value, log_prob, expert_action):
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, self.ep_len, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.ep_len, self.action_dim))
        expert_action = expert_action.reshape(
            (self.n_envs, self.ep_len, self.action_dim)
        )

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.expert_actions[self.pos] = expert_action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.clone().flatten()
        self.log_probs[self.pos] = log_prob.clone()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size=None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "expert_actions",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env=None,
    ):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.expert_actions[batch_inds],
        )
        return OracleRolloutBufferSamples(*tuple(map(self.to_torch, data)))
