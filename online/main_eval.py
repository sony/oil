import pathlib
import sys

from online.helpers import (
    get_best_checkpoint,
    get_experiment_data,
    get_number,
    load_model,
    load_vecnormalize,
)

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import argparse
import os
import json
import numpy as np
import glob
from definitions import ROOT_DIR, MODEL_PATTERN, ENV_PATTERN
from envs.environment_factory import EnvironmentFactory
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import subprocess


TB_DIR_NAME = "PPO_0"  # "RecurrentPPO_1", "SAC_1"
CKPT_CHOICE_CRITERION = "rollout/ep_rew_mean"  # "rollout/ep_rew_mean", "rollout/solved"


def main(args):
    if args.experiment_path is None:
        env_config = json.load(open(args.eval_config_path, "r"))
        env = EnvironmentFactory.create(**env_config)
        model = PPO(policy="MlpPolicy", env=env)
        venv = DummyVecEnv([lambda: env])
        vecnormalize = VecNormalize(venv)
    else:
        experiment_path = ROOT_DIR / args.experiment_path
        env_config = json.load(open(args.eval_config_path, "r"))
        env = EnvironmentFactory.create(**env_config)
        baseline_env = EnvironmentFactory.create(**env_config)
        if args.checkpoint is None:
            # First get the training data from the tensorboard log
            tb_dir_path = os.path.join(experiment_path, TB_DIR_NAME)
            experiment_data = get_experiment_data(tb_dir_path, CKPT_CHOICE_CRITERION)
            steps = experiment_data[CKPT_CHOICE_CRITERION]["x"][0]
            rewards = experiment_data[CKPT_CHOICE_CRITERION]["y"][0]

            # Get the list of checkpoints
            model_list = sorted(
                glob.glob(os.path.join(experiment_path, MODEL_PATTERN)),
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

        model = load_model(
            experiment_path,
            checkpoint,
        )
        vecnormalize = load_vecnormalize(experiment_path, checkpoint, env)

    # Collect rollouts and store them
    vecnormalize.training = False
    mean_ep_rew = 0
    mean_baseline_ep_rew = 0
    for i in range(args.num_episodes):
        lstm_states = None
        ep_rew = 0
        baseline_ep_rew = 0
        step = 0
        obs, _ = env.reset()
        baseline_env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        while not done:
            action, _ = model.predict(
                vecnormalize.normalize_obs(obs),
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=args.deterministic,
            )
            obs, rewards, terminated, truncated, _ = env.step(action)

            baseline_action = env.get_baseline_action()
            _, baseline_rewards, _, _, _ = baseline_env.step(baseline_action)

            done = terminated or truncated
            episode_starts = done
            ep_rew += rewards
            baseline_ep_rew += baseline_rewards
            step += 1
        mean_ep_rew = (mean_ep_rew * i + ep_rew) / (i + 1)
        mean_baseline_ep_rew = (mean_baseline_ep_rew * i + baseline_ep_rew) / (i + 1)
        print(
            "Ep:",
            i,
            "ep rew:",
            ep_rew,
            "avg score:",
            mean_ep_rew,
            "avg_baseline_score:",
            mean_baseline_ep_rew,
        )

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
        "--eval_config_path",
        type=str,
        default=ROOT_DIR / "data" / "env_configs" / "eval_config.json",
        help="Path to the eval config",
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
    args = parser.parse_args()
    main(args)

"""Example:
# mjpython src/main_dataset_recurrent_ppo.py --experiment_path=output/training/2023-09-17/15-29-45_CustomChaseTag_sde_False_lattice_True_freq_1_log_std_init_0.0_ppo_seed_0_xrange_-1_1_yrange_-5_5_static_max_1000_steps \
#     --num_episodes=100 --no_save_df --render --deterministic    
#         mjpython src/main_dataset_recurrent_ppo.py --experiment_path=output/training/ongoing/CustomChaseTag_seed_8_x_-1.0_1.0_y_-5.0_0.0_dist_0.05_hip_0.001_period_100.0_alive_0.0_solved_0.0_early_solved_0.1_joints_0.005_lose_0.0_ref_0.002_heel_0_gait_l_0.8_gait_c_1.0_fix_0.1_ran_0.9_mov_0.0_job_60 \
# --num_episodes=100 --no_save_df --render --deterministic --checkpoint=211986432

python online/main_eval.py --experiment_path=output/training/ongoing/008_ppo_seed_0 \
    --checkpoint=5250000 --num_episodes=100 --no_save_df
    
python online/main_eval.py --experiment_path=output/training/ongoing/013_ppo_seed_0_old_action \
    --checkpoint=27000000 --num_episodes=100 --no_save_df

python online/main_eval.py \
    --num_episodes=100 --no_save_df
"""
