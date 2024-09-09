import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import argparse
import os
import json
import numpy as np
import glob
import torch
from definitions import ROOT_DIR, MODEL_PATTERN, ENV_CONFIG_NAME
from envs.environment_factory import EnvironmentFactory
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from helpers import (
    get_best_checkpoint,
    get_experiment_data,
    get_number,
    load_model,
    load_vecnormalize,
)

torch.manual_seed(0)

TB_DIR_NAME = "PPO_0"  # "RecurrentPPO_1", "SAC_1"
CKPT_CHOICE_CRITERION = "score"  # "rollout/ep_rew_mean", "rollout/solved"


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
        baseline_env = EnvironmentFactory.create(**env_config)
        if args.compute_topline:
            assert env_config["simplified_bidding"]
            topline_config = env_config.copy()
            topline_config["new_action"] = False
            topline_config["multi_action"] = False
            topline_config["exp_action"] = False
            topline_env = EnvironmentFactory.create(**topline_config)

        # We need to use the observation and action defined in the training config
        train_config = json.load(open(experiment_path / ENV_CONFIG_NAME, "r"))
        env_config["obs_keys"] = train_config["obs_keys"]
        if "act_keys" in train_config:
            env_config["act_keys"] = train_config["act_keys"]
        env_config["new_action"] = train_config.get("new_action", False)
        env_config["multi_action"] = train_config.get("multi_action", False)
        env_config["exp_action"] = train_config.get("exp_action", False)

        env = EnvironmentFactory.create(**env_config)

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
    mean_topline_ep_rew = 0
    for i in range(args.num_episodes):
        lstm_states = None
        ep_rew = 0
        baseline_ep_rew = 0
        topline_ep_rew = 0
        step = 0
        obs, _ = env.reset(seed=i)
        baseline_env.reset(
            budget=env.unwrapped.total_budget,
            target_cpa=env.unwrapped.target_cpa,
            advertiser=env.unwrapped.advertiser,
            period=env.unwrapped.period,
        )
        if args.compute_topline:
            topline_env.reset(
                budget=env.unwrapped.total_budget,
                target_cpa=env.unwrapped.target_cpa,
                advertiser=env.unwrapped.advertiser,
                period=env.unwrapped.period,
            )
            topline_action = topline_env.unwrapped.get_simplified_topline_action()
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

            baseline_action = baseline_env.unwrapped.get_baseline_action()
            _, baseline_rewards, _, _, _ = baseline_env.step(baseline_action)
            baseline_ep_rew += baseline_rewards

            if args.compute_topline:
                _, topline_rewards, _, _, _ = topline_env.step(topline_action)
                topline_ep_rew += topline_rewards

            done = terminated or truncated
            episode_starts = done
            ep_rew += rewards
            step += 1
        mean_ep_rew = (mean_ep_rew * i + ep_rew) / (i + 1)
        mean_baseline_ep_rew = (mean_baseline_ep_rew * i + baseline_ep_rew) / (i + 1)
        if args.compute_topline:
            mean_topline_ep_rew = (mean_topline_ep_rew * i + topline_ep_rew) / (i + 1)
        str_out = (
            "Ep: {} ep rew: {:.2f} avg score: {:.2f} avg_baseline_score: {:.2f}".format(
                i, ep_rew, mean_ep_rew, mean_baseline_ep_rew
            )
        )
        if args.compute_topline:
            str_out += " avg_topline_score: {:.2f}".format(mean_topline_ep_rew)
        print(str_out)

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
    parser.add_argument(
        "--compute_topline",
        action="store_true",
        default=False,
        help="Flag to compute the topline",
    )
    args = parser.parse_args()
    main(args)

"""Example:
python online/main_eval.py --experiment_path=output/training/ongoing/008_ppo_seed_0 \
    --checkpoint=5250000 --num_episodes=100 --no_save_df
    
python online/main_eval.py --experiment_path=output/training/ongoing/013_ppo_seed_0_old_action \
    --checkpoint=28000000 --num_episodes=100 --no_save_df

python online/main_eval.py --experiment_path=output/training/ongoing/017_ppo_seed_0_new_action_test \
    --num_episodes=100 --no_save_df --checkpoint=6500000

python online/main_eval.py --experiment_path=output/training/ongoing/016_ppo_seed_0_new_action_test \
    --num_episodes=100 --no_save_df --checkpoint=10250000 --deterministic 
    
python online/main_eval.py --experiment_path=output/training/ongoing/027_ppo_seed_0_dense_base_ranges_19_obs_exp_single_action_simplified \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 13250000

# Best: 0.41
python online/main_eval.py --experiment_path=output/training/ongoing/029_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_simplified \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=10750000
    
# Only 0.35 despite very strong offline performance?
python online/main_eval.py --experiment_path=output/training/ongoing/031_ppo_seed_0_dense_base_ranges_19_obs_exp_multi_action_simplified \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=8000000
    
# Very good and still improving a little
python online/main_eval.py --experiment_path=output/training/ongoing/032_ppo_seed_0_dense_base_ranges_29_obs_exp_multi_action_simplified \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=8500000

# Trained with realistic bidding, can keep up with the best in the simplified bidding
python online/main_eval.py --experiment_path=output/training/ongoing/033_ppo_seed_0_dense_base_ranges_29_obs_exp_multi_action \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=3500000

# 0.4164
python online/main_eval.py --experiment_path=output/training/ongoing/035_ppo_seed_0_dense_larger_ranges_29_obs_exp_single_action_simplified_resume_029 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=12250000

# New best! 0.4199
python online/main_eval.py --experiment_path=output/training/ongoing/036_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_resume_029 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=12000000

# New best!!! 0.4312, local: 562.65 (so high)
python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=5000000


# Also very good
python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=5500000

# Also very good
python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=5250000

# 0.4205
python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=10000000
    
python online/main_eval.py --experiment_path=output/training/ongoing/038_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_noisy_env \
    --num_episodes=100 --no_save_df --deterministic

python online/main_eval.py --experiment_path=output/training/ongoing/038_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_noisy_env \
    --num_episodes=100 --no_save_df --deterministic --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_stochastic.json


python online/main_eval.py --experiment_path=output/training/ongoing/038_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_noisy_env \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 4000000 --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_realistic_stochastic.json
    
python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=5000000 --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_realistic_stochastic.json

python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=5000000 --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_stochastic.json

python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=5000000 --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_realistic.json

# Only 0.3539
python online/main_eval.py --experiment_path=output/training/ongoing/037_ppo_seed_0_dense_base_ranges_29_obs_exp_3_actions \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=7750000 \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config.json

python online/main_eval.py --experiment_path=output/training/ongoing/037_ppo_seed_0_dense_base_ranges_29_obs_exp_3_actions \
    --num_episodes=100 --no_save_df --deterministic \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config.json


python online/main_eval.py --experiment_path=output/training/ongoing/039_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_stoch_exposure \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 5000000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config.json

python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=5000000 --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config.json \
        --compute_topline
    


"""
