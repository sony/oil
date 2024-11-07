from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


class BatchSubprocVecEnv(SubprocVecEnv):
    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)  # type: ignore[assignment]
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos  # type: ignore[return-value]

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space)


def _flatten_obs(
    obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space
) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(
        obs, (list, tuple)
    ), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(
            space.spaces, OrderedDict
        ), "Dict space must have ordered subspaces"
        assert isinstance(
            obs[0], dict
        ), "non-dict observation for environment with Dict observation space"
        return OrderedDict(
            [(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()]
        )
    elif isinstance(space, spaces.Tuple):
        assert isinstance(
            obs[0], tuple
        ), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))  # type: ignore[index]
    else:
        if obs[0].shape == space.shape:
            return np.stack(obs)
        else:
            return np.concatenate(obs)  # type: ignore[arg-type]
