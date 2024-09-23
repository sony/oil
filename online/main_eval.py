import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import argparse
import os
import json
import numpy as np
import glob
import torch
import logging
from definitions import ROOT_DIR, MODEL_PATTERN, ENV_CONFIG_NAME, ALGO_TB_DIR_NAME_DICT
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
            if env_config["simplified_bidding"]:
                logging.warning(
                    "Using the oracle action computed for simplified auction for relistic auction"
                )
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
        # env_config["deterministic_conversion"] = True

        env = EnvironmentFactory.create(**env_config)

        if args.checkpoint is None:
            # First get the training data from the tensorboard log
            tb_dir_path = os.path.join(
                experiment_path, ALGO_TB_DIR_NAME_DICT[args.algo]
            )
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
            args.algo,
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
        obs, _ = env.reset(seed=i)  # , advertiser=0)
        baseline_env.reset(
            budget=env.unwrapped.total_budget,
            target_cpa=env.unwrapped.target_cpa,
            advertiser=env.unwrapped.advertiser,
            period=env.unwrapped.period,
        )
        baseline_env.unwrapped.episode_pvalues_df = env.unwrapped.episode_pvalues_df
        baseline_env.unwrapped.episode_bids_df = env.unwrapped.episode_bids_df

        if args.compute_topline:
            topline_env.reset(
                budget=env.unwrapped.total_budget,
                target_cpa=env.unwrapped.target_cpa,
                advertiser=env.unwrapped.advertiser,
                period=env.unwrapped.period,
            )
            topline_env.unwrapped.episode_pvalues_df = env.unwrapped.episode_pvalues_df
            topline_env.unwrapped.episode_bids_df = env.unwrapped.episode_bids_df
            topline_action = topline_env.unwrapped.get_oracle_action()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        if args.algo == "onbc_transformer":
            obs_list = []
        while not done:
            norm_obs = vecnormalize.normalize_obs(obs)
            if args.algo == "onbc_transformer":
                obs_list.append(norm_obs)
                norm_obs = np.stack(obs_list)
                action, _ = model.predict(
                    norm_obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=args.deterministic,
                    single_action=True,
                )
            else:
                action, _ = model.predict(
                    norm_obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=args.deterministic,
                )
            obs, rewards, terminated, truncated, _ = env.step(action)

            baseline_action = baseline_env.unwrapped.get_baseline_action()
            _, baseline_rewards, _, _, _ = baseline_env.step(baseline_action)
            baseline_ep_rew += baseline_rewards

            if args.compute_topline:
                # topline_action = topline_env.unwrapped.get_oracle_action()
                topline_action = topline_env.unwrapped.get_simplified_oracle_action()
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
        "--algo",
        type=str,
        default="ppo",
        help="Algorithm used to train the agent.",
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
        default=ROOT_DIR / "env_configs" / "eval_config.json",
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

# New best!!! 0.4446, local: 573.87
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --no_save_df --deterministic --compute_topline --checkpoint 1500000

# New best!!! 0.4485, local: 569.95
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --no_save_df --deterministic --compute_topline --checkpoint 2250000

# 583.44
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --no_save_df --deterministic --compute_topline --checkpoint 2500000

# New best!!! 0.4543, local: 590.59
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --no_save_df --deterministic --compute_topline --checkpoint 2750000

local: 579.30
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --no_save_df --deterministic --compute_topline --checkpoint 3000000

# Submission: 0.4403 local: 588.15
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --no_save_df --deterministic --compute_topline --checkpoint 3500000
    
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --no_save_df --deterministic --compute_topline --checkpoint 3750000
    
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 4500000

# Submission: 0.4396, local: 591.46
python online/main_eval.py --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_040 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2850000
    
python online/main_eval.py --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_040 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 4220000
    
python online/main_eval.py --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_040 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 4730000
    
python online/main_eval.py --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_040 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 5000000

# Local: 588.33
python online/main_eval.py --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_040 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 4190000

python online/main_eval.py --experiment_path=output/training/ongoing/029_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_simplified \
    --num_episodes=100 --no_save_df --deterministic --checkpoint=6000000

# Submission: 0.4531, local: 591.95
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/002_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3150000

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/002_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3200000
    
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/002_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2600000

# local:  586.67
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/004_onbc_seed_0_transformer \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2400000

# local:  590.52
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/004_onbc_seed_0_transformer \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2500000
        
# local: 587.09
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/004_onbc_seed_0_transformer \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2850000

# local: 594.21
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/004_onbc_seed_0_transformer \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3750000

# New best: 0.4572 local: 592.57
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/005_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_002 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3350000
    
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/005_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_002 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3350000 --compute_topline \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_stochastic.json
        
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/005_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_002 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3350000 --compute_topline \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_stochastic.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/009_onbc_seed_0_small_pvals_auction_noise_simplified \
    --num_episodes=100 --no_save_df --deterministic --compute_topline \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_stochastic.json

# avg score: 36.95 avg_baseline_score: 30.65 avg_topline_score: 42.78
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/013_onbc_seed_0_new_data_simplified \
    --num_episodes=100 --no_save_df --deterministic --compute_topline --checkpoint 420000

# Best!!! 0.4598, avg score: 37.42, Score: 71.42
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/013_onbc_seed_0_new_data_simplified \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3650000

# Best!!! 0.4712, avg score: 38.29, score: 72.67
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/016_onbc_seed_0_new_data_simplified_small_lr_resume_013 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3710000

avg score: 27.45 avg_baseline_score: 21.24 avg_topline_score: 29.48
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/016_onbc_seed_0_new_data_simplified_small_lr_resume_013 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3710000 --compute_topline \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

25.86
python online/main_eval.py --algo ppo --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_29_obs_exp_single_action_realistic_auction_new_data \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 4350000 \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

python online/main_eval.py --algo ppo --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_29_obs_exp_single_action_realistic_auction_new_data \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 5950000 \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json
        
python online/main_eval.py --algo ppo --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_29_obs_exp_single_action_realistic_auction_new_data \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 7000000 \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json
        
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/010_onbc_seed_0_all_pvals_auction_noise_simplified \
    --num_episodes=100 --no_save_df --deterministic --compute_topline \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_stochastic.json

# avg score: 36.63 avg_baseline_score: 30.65 avg_topline_score: 42.78
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/014_onbc_seed_0_transformer_new_data \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2400000
    
avg score: 38.41, test: 72.42 (baseline 39.05)
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/014_onbc_seed_0_transformer_new_data \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2700000
        
avg score: 36.74
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/017_onbc_seed_0_stoch_exposure_simplified_new_data \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3700000
        
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/017_onbc_seed_0_stoch_exposure_simplified_new_data \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3700000 --compute_topline\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

# 27.62
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/018_onbc_seed_0_new_data_realistic \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 750000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

score: 27.90
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/018_onbc_seed_0_new_data_realistic \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3800000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

score 28.08
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/018_onbc_seed_0_new_data_realistic \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 5300000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

submission: 0.4665, local: 28.29
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/018_onbc_seed_0_new_data_realistic \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2000000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/018_onbc_seed_0_new_data_realistic \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2050000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json


python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/018_onbc_seed_0_new_data_realistic \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 1950000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

local: 28.80
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3240000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

New best!!! 48.84, local: 28.88
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2840000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

local: 28.18
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2850000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

local: 28.71
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2710000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

28.79
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3860000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

28.85
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3870000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

28.72
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3640000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

28.54
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3650000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

local: 28.39
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2040000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

28.78
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 3840000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

local: 28.68
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 4630000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

local: 28.58
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 4640000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

28.58
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 4650000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

28.95
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 4840000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

28.73
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 4850000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

28.44
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 5300000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

28.42
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 5310000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

28.62
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 6330000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

29.17, score: 72.25
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 6590000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

28.73
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --no_save_df --deterministic --checkpoint 2910000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

28.75, score: 61.33
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/020_onbc_seed_0_transformer_new_data_realistic \
    --num_episodes=100 --no_save_df --deterministic --4050000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json
"""
