import numpy as np
import torch
from typing import NamedTuple
from stable_baselines3.common.buffers import RolloutBuffer


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

    def reset(self):
        self.expert_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        super().reset()

    def add(self, obs, action, reward, episode_start, value, log_prob, expert_action):
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
