import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import os
import numpy as np
import torch
from typing import Iterable, Union
from scipy.signal import savgol_filter
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from definitions import ENV_PATTERN, MODEL_PATTERN, ALGO_CLASS_DICT
from stable_baselines3 import PPO
from scipy.signal import butter, lfilter


def get_number(filename):
    return int(filename.split("_steps.zip")[0].split("_")[-1])


def load_model(
    algo,
    experiment_path,
    checkpoint_number,
):
    if checkpoint_number is None:
        model_file = "best_model"
    else:
        model_file = MODEL_PATTERN.replace("*", str(checkpoint_number))
    model_path = os.path.join(experiment_path, model_file)
    model = ALGO_CLASS_DICT[algo].load(model_path)
    return model


def load_vecnormalize(experiment_path, checkpoint_number, base_env):
    if checkpoint_number is None:
        env_file = "training_env.pkl"
    else:
        env_file = ENV_PATTERN.replace("*", str(checkpoint_number))
    env_path = os.path.join(experiment_path, env_file)
    venv = DummyVecEnv([lambda: base_env])
    print("env path", env_path)
    vecnormalize = VecNormalize.load(env_path, venv)
    return vecnormalize


def get_sorted_checkpoints(steps, rewards, checkpoints, verbose=1, descending=True):
    # Lowpass filter the rewards to avoid choosing a checkpoint at a peak due to noise
    clean_rewards = savgol_filter(rewards, window_length=51, polyorder=3)
    steps = list(steps)

    # Get the list of the closest steps to the checkpoints and the corresponding rewards
    closest_step_list = [
        min(steps, key=lambda x: abs(x - ckpt)) for ckpt in checkpoints
    ]

    # Get the corresponding filtered rewards for these checkpoints
    closest_reward_list = [
        clean_rewards[steps.index(closest_step)] for closest_step in closest_step_list
    ]

    # Combine checkpoints and rewards into tuples and sort by reward (best to worst)
    checkpoint_reward_pairs = sorted(
        zip(checkpoints, closest_reward_list), key=lambda x: x[1], reverse=descending
    )

    # Unpack the sorted list
    sorted_checkpoints = [ckpt for ckpt, _ in checkpoint_reward_pairs]

    if verbose:
        print("Checkpoints sorted by reward (best to worst):")
        for ckpt, reward in checkpoint_reward_pairs:
            print(f"Checkpoint: {ckpt}, Reward: {reward}")

    return sorted_checkpoints


def get_best_checkpoint(steps, rewards, checkpoints, verbose=1, descending=True):
    sorted_checkpoints = get_sorted_checkpoints(
        steps, rewards, checkpoints, verbose, descending
    )
    return sorted_checkpoints[0]


def get_data_from_tb_log(path, y, x="step", tb_config=None):
    if tb_config is None:
        tb_config = {}

    event_acc = EventAccumulator(path, tb_config)
    event_acc.Reload()

    if not isinstance(y, Iterable) or isinstance(y, str):
        y = [y]

    out_dict = {}
    for attr_name in y:
        if attr_name in event_acc.Tags()["scalars"]:
            x_vals, y_vals = np.array(
                [(getattr(el, x), el.value) for el in event_acc.Scalars(attr_name)]
            ).T
            out_dict[attr_name] = (x_vals, y_vals)
        else:
            out_dict[attr_name] = None
    return out_dict


def get_experiment_data(tb_dir_path, attributes, tb_config=None):
    experiment_data = {}
    folder_content = os.listdir(tb_dir_path)
    assert len(folder_content) == 1
    tb_file_name = folder_content[0]
    tb_file_path = os.path.join(tb_dir_path, tb_file_name)
    data_dict = get_data_from_tb_log(tb_file_path, attributes, tb_config=tb_config)
    for key, values in data_dict.items():
        if values is not None:
            x_vals, y_vals = values
            experiment_data_el = experiment_data.get(key)
            if experiment_data_el is None:
                experiment_data[key] = {}
                experiment_data[key]["x"] = [x_vals]
                experiment_data[key]["y"] = [y_vals]
            else:
                experiment_data[key]["x"].append(x_vals)
                experiment_data[key]["y"].append(y_vals)
    return experiment_data


def my_safe_to_tensor(array: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
    """Converts a NumPy array to a PyTorch tensor.

    The data is copied in the case where the array is non-writable. Unfortunately if
    you just use `torch.as_tensor` for this, an ugly warning is logged and there's
    undefined behavior if you try to write to the tensor.

    Args:
        array: The array to convert to a PyTorch tensor.
        kwargs: Additional keyword arguments to pass to `torch.as_tensor`.

    Returns:
        A PyTorch tensor with the same content as `array`.
    """
    return torch.as_tensor(array, dtype=torch.float32, **kwargs)


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
