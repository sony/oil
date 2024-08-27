import numpy as np
from typing import Iterable
from scipy.signal import savgol_filter
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from definitions import ENV_PATTERN, MODEL_PATTERN
from stable_baselines3 import PPO
import os


def get_number(filename):
    return int(filename.split("_steps.zip")[0].split("_")[-1])


def load_model(
    experiment_path,
    checkpoint_number,
):
    if checkpoint_number is None:
        model_file = "best_model"
    else:
        model_file = MODEL_PATTERN.replace("*", str(checkpoint_number))
    model_path = os.path.join(experiment_path, model_file)
    model = PPO.load(model_path)
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


def get_best_checkpoint(steps, rewards, checkpoints, verbose=1):
    # Lowpass filter the rewards to avoid choosing a checkpoint at a peak due to noise
    clean_rewards = savgol_filter(rewards, window_length=51, polyorder=3)
    steps = list(steps)
    # Get the list of the closest steps to the checkpoints and the corresponding rewards
    closest_step_list = [
        min(steps, key=lambda x: abs(x - ckpt)) for ckpt in checkpoints
    ]
    closest_reward_list = [
        clean_rewards[steps.index(closest_step)] for closest_step in closest_step_list
    ]
    reward_ckpt_max = max(closest_reward_list)
    step_ckpt_max_approx = closest_step_list[closest_reward_list.index(reward_ckpt_max)]
    step_ckpt_max = min(checkpoints, key=lambda x: abs(x - step_ckpt_max_approx))
    if verbose:
        print(
            "Best checkpoint:",
            step_ckpt_max,
            ", corresponding reward:",
            reward_ckpt_max,
        )
    return step_ckpt_max


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
