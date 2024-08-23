import argparse
import os
import json
import numpy as np
import pandas as pd
import glob
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from scipy.signal import savgol_filter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Iterable
from torch import nn
import skvideo
import platform
import subprocess


MODEL_PATTERN = "rl_model_*_steps.zip"
ENV_PATTERN = "rl_model_vecnormalize_*_steps.pkl"
TB_DIR_NAME = "PPO_1"  # "RecurrentPPO_1", "SAC_1"
CKPT_CHOICE_CRITERION = "rollout/ep_rew_mean"  # "rollout/ep_rew_mean", "rollout/solved"
VIDEO_DIR = os.path.join(ROOT_DIR, "data", "videos")
HOST = "chiappa@sv-rcp-gateway.intranet.epfl.ch"
HOST_PROJECT_ROOT = "/storage-rcp-pure/upamathis_scratch/alberto/SharedBionicControl"


def get_number(filename):
    return int(filename.split("_steps.zip")[0].split("_")[-1])


def load_model(experiment_path, checkpoint_number=None, action_space=None, observation_space=None, model_config=None):
    custom_objects = {}
    if checkpoint_number is None:
        model_file = "best_model"
    else:
        model_file = MODEL_PATTERN.replace("*", str(checkpoint_number))
    model_path = os.path.join(experiment_path, model_file)
    if not os.path.exists(model_path):
        print("Attempting to fetch remote experiment...")
        get_remote_checkpoint(experiment_path, checkpoint_number)
    model = PPO.load(model_path, custom_objects=custom_objects)
    return model


def load_vecnormalize(experiment_path, checkpoint_number, base_env):
    if checkpoint_number is None:
        env_file = "training_env.pkl"
    else:
        env_file = ENV_PATTERN.replace("*", str(checkpoint_number))
    env_path = os.path.join(experiment_path, env_file)
    if not os.path.exists(env_path):
        print("Attempting to fetch remote experiment...")
        get_remote_checkpoint(experiment_path, checkpoint_number)
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


def get_remote_checkpoint(experiment_path, checkpoint_num):
    if checkpoint_num is None:
        raise NotImplementedError("Selection of best checkpoint from the remote not implemented")
    file_names = [
        "args.json",
        "env_config.json",
        "*_config.json",
        f"rl_model_{checkpoint_num}_steps.zip",
        f"rl_model_vecnormalize_{checkpoint_num}_steps.pkl"
    ]
    file_paths = [os.path.join(f"{HOST}:{HOST_PROJECT_ROOT}", experiment_path, f) for f in file_names]
    os.makedirs(os.path.join(ROOT_DIR, experiment_path), exist_ok=True)
    subprocess.run(["rsync",  *file_paths, os.path.join(ROOT_DIR, experiment_path)])


def main(args):
    if args.experiment_path is None:
        env = EnvironmentFactory.create(args.env_name)
        model = PPO(policy="MultiInputPolicy", env=env)
        venv = DummyVecEnv([lambda: env])
        vecnormalize = VecNormalize(venv)
        sde_sample_freq = 1
    else:
        config_path = os.path.join(args.experiment_path, f"env_config.json")
        if not os.path.exists(config_path):
            print("Attempting to fetch remote experiment...")
            get_remote_checkpoint(args.experiment_path, args.checkpoint)
        env_config = json.load(open(config_path, "r"))
        ## TODO: remove
        if args.env_name is not None:
            env_config = {"env_name": args.env_name}
        env = EnvironmentFactory.create(**env_config)
        if args.checkpoint is None:
            # First get the training data from the tensorboard log
            tb_dir_path = os.path.join(args.experiment_path, TB_DIR_NAME)
            experiment_data = get_experiment_data(tb_dir_path, CKPT_CHOICE_CRITERION)
            steps = experiment_data[CKPT_CHOICE_CRITERION]["x"][0]
            rewards = experiment_data[CKPT_CHOICE_CRITERION]["y"][0]

            # Get the list of checkpoints
            model_list = sorted(
                glob.glob(os.path.join(args.experiment_path, MODEL_PATTERN)),
                key=get_number,
            )
            checkpoints = [
                get_number(el)
                for el in model_list
                if get_number(el) < args.max_checkpoint
            ]
            if len(checkpoints):
                # Select the checkpoint corresponding to the best reward
                checkpoint = get_best_checkpoint(steps, rewards, checkpoints)
            else:
                checkpoint = None
        else:
            checkpoint = args.checkpoint
        model_config_path = os.path.join(args.experiment_path, "model_config.json")
        model_config = json.load(open(model_config_path, "r"))
        model = load_model(args.experiment_path, checkpoint, action_space=env.action_space, observation_space=env.observation_space)
        vecnormalize = load_vecnormalize(args.experiment_path, checkpoint, env)

    # Collect rollouts and store them
    vecnormalize.training = False
    # if args.render:
    #     env.mujoco_render_frames = True
    if args.save_video:
        env.mujoco_render_frames = False
        frames = []
    for i in range(args.num_episodes):
        lstm_states = None
        cum_rew = 0
        step = 0
        obs, _ = env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        while not done:
            if args.render:
                # env.sim.renderer.render_to_window()
                env.render()
            if args.save_video:
                curr_frame = env.sim.renderer.render_offscreen(
                        # cameras=[None],
                        width=640,
                        height=480,
                        camera_id=1,
                        device_id=0
                    )
                frames.append(curr_frame)
            if model.use_sde and not args.deterministic and step % sde_sample_freq == 0:
                model.policy.reset_noise()

            action, lstm_states = model.predict(
                vecnormalize.normalize_obs(obs),
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=args.deterministic,
            )
            next_obs, rewards, terminated, truncated, _ = env.step(action)
            print(rewards)
            if(env.data.time == 0):
                print("Done should be true", terminated or truncated)
            # print("qpos", next_obs["qpos"][-6:-3])
            # print("tip err", np.linalg.norm(next_obs["error_pos"]))
            import time
            time.sleep(0.05)
            print(env.reward_dict)
            done = terminated or truncated
            episode_starts = done
            cum_rew += rewards
            step += 1
        print("Episode", i, ", len:", step, ", cum rew: ", cum_rew)
        
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main script to create a dataset of episodes with a trained agent"
    )

    parser.add_argument(
        "--experiment_path",
        type=str,
        default=None,
        help="Path to the folder where the experiment results are stored",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Number of the checkpoint to select. Otherwise the checkpoint corresponding to the highest reward is selected.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        help="Name of the environment where to test the agent",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100, help="Number of episodes to collect"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Flag to use the deterministic policy",
    )

    parser.add_argument(
        "--max_checkpoint",
        type=float,
        default=float("inf"),
        help="Do not consider checkpoints past this number (to be fair across trainings)",
    )
    parser.add_argument(
        "--no_save_df",
        action="store_true",
        default=False,
        help="Flag to not save the dataframe",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Flag to render at the screen",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=False,
        help="Flag to save a video",
    )
    args = parser.parse_args()
    main(args)
    
    """Example:
    # mjpython src/main_dataset_recurrent_ppo.py --experiment_path=output/training/2023-09-17/15-29-45_CustomChaseTag_sde_False_lattice_True_freq_1_log_std_init_0.0_ppo_seed_0_xrange_-1_1_yrange_-5_5_static_max_1000_steps \
    #     --num_episodes=100 --no_save_df --render --deterministic    
    #         mjpython src/main_dataset_recurrent_ppo.py --experiment_path=output/training/ongoing/CustomChaseTag_seed_8_x_-1.0_1.0_y_-5.0_0.0_dist_0.05_hip_0.001_period_100.0_alive_0.0_solved_0.0_early_solved_0.1_joints_0.005_lose_0.0_ref_0.002_heel_0_gait_l_0.8_gait_c_1.0_fix_0.1_ran_0.9_mov_0.0_job_60 \
    # --num_episodes=100 --no_save_df --render --deterministic --checkpoint=211986432
    """
