import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import argparse
import os
import json
import numpy as np
import glob
import torch
import time
import pandas as pd
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
    get_sorted_checkpoints,
)

torch.manual_seed(0)

CKPT_CHOICE_CRITERION = "score"  # "rollout/ep_rew_mean", "rollout/solved"
descending = False


def main(args):
    start_ts = int(time.time())
    if args.experiment_path is None:
        env_config = json.load(open(args.eval_config_path, "r"))
        env = EnvironmentFactory.create(**env_config)
        model = PPO(policy="MlpPolicy", env=env)
        venv = DummyVecEnv([lambda: env])
        vecnormalize = VecNormalize(venv)
        checkpoint_list = [None]
    else:
        experiment_path = ROOT_DIR / args.experiment_path
        train_config = json.load(open(experiment_path / ENV_CONFIG_NAME, "r"))

        if args.eval_config_path is None:
            env_config = {
                key: val for key, val in train_config.items() if key != "seed"
            }
            env_config["pvalues_df_path"] = (
                "data/online_rl_data_sparse/period-27_pvalues.parquet"
            )
            env_config["bids_df_path"] = (
                "data/online_rl_data_sparse/period-27_bids.parquet",
            )
        else:
            env_config = json.load(open(args.eval_config_path, "r"))
        if args.compute_baseline:
            baseline_env = EnvironmentFactory.create(**env_config)
        if args.compute_topline:
            topline_env = EnvironmentFactory.create(**env_config)
        if args.compute_flex_topline:
            flex_topline_config = env_config.copy()
            flex_topline_config["oracle_upgrade"] = True
            flex_topline_config["two_slopes_action"] = args.two_slopes_action
            flex_topline_config["oracle_upgrade_cost_weight"] = (
                args.oracle_upgrade_cost_weight
            )
            flex_topline_env = EnvironmentFactory.create(**flex_topline_config)

        env_config["obs_keys"] = train_config["obs_keys"]
        if "act_keys" in train_config:
            env_config["act_keys"] = train_config["act_keys"]
        env_config["deterministic_conversion"] = args.deterministic_conversion
        env_config["two_slopes_action"] = train_config.get("two_slopes_action", False)
        env_config["single_io_bid"] = train_config.get("single_io_bid", False)
        env_config["batch_state"] = True

        env = EnvironmentFactory.create(**env_config)
        if args.checkpoint is None:
            # Get the list of checkpoints
            model_list = sorted(
                glob.glob(os.path.join(experiment_path, MODEL_PATTERN)),
                key=get_number,
            )
            checkpoints = [
                get_number(el)
                for el in model_list
                if args.min_checkpoint < get_number(el) < args.max_checkpoint
            ]
            # First get the training data from the tensorboard log
            tb_dir_path = os.path.join(
                experiment_path, ALGO_TB_DIR_NAME_DICT[args.algo]
            )
            experiment_data = get_experiment_data(tb_dir_path, CKPT_CHOICE_CRITERION)
            steps = experiment_data[CKPT_CHOICE_CRITERION]["x"][0]
            rewards = experiment_data[CKPT_CHOICE_CRITERION]["y"][0]

            if len(checkpoints):
                if args.all_checkpoints:
                    checkpoint_list = get_sorted_checkpoints(
                        steps, rewards, checkpoints, descending
                    )
                else:
                    # Select the checkpoint corresponding to the best reward
                    checkpoint = get_best_checkpoint(
                        steps, rewards, checkpoints, descending
                    )
                    checkpoint_list = [checkpoint]
            else:
                checkpoint_list = [None]
        else:
            checkpoint_list = [args.checkpoint]

    best_score = -np.inf
    best_checkpoint = None
    score_list = []
    dataset_list = []

    for checkpoint in checkpoint_list:
        model = load_model(
            args.algo,
            experiment_path,
            checkpoint,
        )
        vecnormalize = load_vecnormalize(experiment_path, checkpoint, env)

        # Collect rollouts and store them
        vecnormalize.training = False
        ep_rew_list = []
        ep_cost_over_budget_list = []
        ep_target_cpa_over_cpa_list = []
        ep_total_conversions_list = []
        ep_baseline_rew_list = []
        ep_topline_rew_list = []
        ep_topline_cost_over_budget_list = []
        ep_topline_target_cpa_over_cpa_list = []
        ep_topline_total_conversions_list = []
        ep_flex_topline_rew_list = []
        ep_flex_topline_cost_over_budget_list = []
        ep_flex_topline_target_cpa_over_cpa_list = []
        ep_flex_topline_total_conversions_list = []

        for i in range(args.num_episodes):
            lstm_states = None
            ep_rew = 0
            baseline_ep_rew = 0
            topline_ep_rew = 0
            flex_topline_ep_rew = 0
            step = 0
            obs, _ = env.reset(seed=i + args.seed, advertiser=args.advertiser)
            if args.compute_baseline:
                baseline_env.reset(
                    budget=env.unwrapped.total_budget,
                    target_cpa=env.unwrapped.target_cpa,
                    advertiser=env.unwrapped.advertiser,
                    period=env.unwrapped.period,
                )
                baseline_env.unwrapped.episode_pvalues_df = (
                    env.unwrapped.episode_pvalues_df
                )
                baseline_env.unwrapped.episode_bids_df = env.unwrapped.episode_bids_df

            if args.compute_topline:
                topline_env.reset(
                    budget=env.unwrapped.total_budget,
                    target_cpa=env.unwrapped.target_cpa,
                    advertiser=env.unwrapped.advertiser,
                    period=env.unwrapped.period,
                )
                topline_env.unwrapped.episode_pvalues_df = (
                    env.unwrapped.episode_pvalues_df
                )
                topline_env.unwrapped.episode_bids_df = env.unwrapped.episode_bids_df

            if args.compute_flex_topline:
                flex_topline_env.reset(
                    budget=env.unwrapped.total_budget,
                    target_cpa=env.unwrapped.target_cpa,
                    advertiser=env.unwrapped.advertiser,
                    period=env.unwrapped.period,
                )
                flex_topline_env.unwrapped.episode_pvalues_df = (
                    env.unwrapped.episode_pvalues_df
                )
                flex_topline_env.unwrapped.episode_bids_df = (
                    env.unwrapped.episode_bids_df
                )

            episode_starts = np.ones((1,), dtype=bool)
            done = False
            if args.algo == "oil_transformer":
                obs_list = []
            while not done:
                norm_obs = vecnormalize.normalize_obs(obs)
                if args.algo == "oil_transformer":
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

                if args.create_dataset:
                    oracle_action = env.unwrapped.get_oracle_upgrade_action()
                    pvalues, pvalues_sigma = env.unwrapped.get_pvalues_mean_and_std()
                    dataset_list.append(
                        {
                            "episode": i,
                            "step": step,
                            "obs": obs.tolist(),
                            "norm_obs": norm_obs.tolist(),
                            "action": action.item(),
                            "oracle_action": oracle_action.flatten().tolist(),
                            "pvalues": pvalues.tolist(),
                            "pvalues_sigma": pvalues_sigma.tolist(),
                        }
                    )
                obs, rewards, terminated, truncated, info = env.step(action)

                if args.compute_baseline:
                    baseline_action = baseline_env.unwrapped.get_baseline_action()
                    _, baseline_rewards, _, _, _ = baseline_env.step(baseline_action)
                    baseline_ep_rew += baseline_rewards

                if args.compute_topline:
                    topline_action = topline_env.unwrapped.get_oracle_action()
                    _, topline_rewards, _, _, topline_info = topline_env.step(
                        topline_action
                    )
                    topline_ep_rew += topline_rewards

                if args.compute_flex_topline:
                    flex_topline_action = (
                        flex_topline_env.unwrapped.get_oracle_upgrade_action()
                    )
                    _, flex_topline_rewards, _, _, flex_topline_info = (
                        flex_topline_env.step(flex_topline_action)
                    )
                    flex_topline_ep_rew += flex_topline_rewards

                done = terminated or truncated
                episode_starts = done
                ep_rew += rewards
                step += 1
            ep_rew_list.append(ep_rew)
            ep_cost_over_budget_list.append(info["cost_over_budget"])
            ep_target_cpa_over_cpa_list.append(info["target_cpa_over_cpa"])
            ep_total_conversions_list.append(info["conversions"])

            if args.compute_baseline:
                ep_baseline_rew_list.append(baseline_ep_rew)
            if args.compute_topline:
                ep_topline_rew_list.append(topline_ep_rew)
                ep_topline_cost_over_budget_list.append(
                    topline_info["cost_over_budget"]
                )
                ep_topline_target_cpa_over_cpa_list.append(
                    topline_info["target_cpa_over_cpa"]
                )
                ep_topline_total_conversions_list.append(topline_info["conversions"])

            if args.compute_flex_topline:
                ep_flex_topline_rew_list.append(flex_topline_ep_rew)
                ep_flex_topline_cost_over_budget_list.append(
                    flex_topline_info["cost_over_budget"]
                )
                ep_flex_topline_target_cpa_over_cpa_list.append(
                    flex_topline_info["target_cpa_over_cpa"]
                )
                ep_flex_topline_total_conversions_list.append(
                    flex_topline_info["conversions"]
                )

            mean_ep_rew = np.mean(ep_rew_list)
            sem_ep_rew = np.std(ep_rew_list) / np.sqrt(len(ep_rew_list))
            mean_ep_cost_over_budget = np.mean(ep_cost_over_budget_list)
            sem_ep_cost_over_budget = np.std(ep_cost_over_budget_list) / np.sqrt(
                len(ep_cost_over_budget_list)
            )
            mean_ep_target_cpa_over_cpa = np.mean(ep_target_cpa_over_cpa_list)
            sem_ep_target_cpa_over_cpa = np.std(ep_target_cpa_over_cpa_list) / np.sqrt(
                len(ep_target_cpa_over_cpa_list)
            )

            if args.compute_baseline:
                mean_baseline_ep_rew = np.mean(ep_baseline_rew_list)
                sem_baseline_ep_rew = np.std(ep_baseline_rew_list) / np.sqrt(
                    len(ep_baseline_rew_list)
                )
            if args.compute_topline:
                mean_topline_ep_rew = np.mean(ep_topline_rew_list)
                sem_topline_ep_rew = np.std(ep_topline_rew_list) / np.sqrt(
                    len(ep_topline_rew_list)
                )
            if args.compute_flex_topline:
                mean_flex_topline_ep_rew = np.mean(ep_flex_topline_rew_list)
                sem_flex_topline_ep_rew = np.std(ep_flex_topline_rew_list) / np.sqrt(
                    len(ep_flex_topline_rew_list)
                )

            str_out = "Ep: {} ep rew: {:.2f}, avg score: {:.2f} ± {:.2f}, avg c/b: {:.2f} ± {:.2f}, avg t_cpa/cpa: {:.2f} ± {:.2f},".format(
                i,
                ep_rew,
                mean_ep_rew,
                sem_ep_rew,
                mean_ep_cost_over_budget,
                sem_ep_cost_over_budget,
                mean_ep_target_cpa_over_cpa,
                sem_ep_target_cpa_over_cpa,
            )
            if args.compute_baseline:
                str_out += " avg_baseline_score: {:.2f} ± {:.2f}".format(
                    mean_baseline_ep_rew, sem_baseline_ep_rew
                )
            if args.compute_topline:
                str_out += " avg_topline_score: {:.2f} ± {:.2f}".format(
                    mean_topline_ep_rew, sem_topline_ep_rew
                )
            if args.compute_flex_topline:
                str_out += " avg_flex_topline_score: {:.2f} ± {:.2f}".format(
                    mean_flex_topline_ep_rew, sem_flex_topline_ep_rew
                )
            print(str_out)

        env.close()
        score_list.append(mean_ep_rew)
        if mean_ep_rew > best_score:
            best_score = mean_ep_rew
            best_checkpoint = checkpoint
        print("Best score: {:.2f} (checkpoint {})".format(best_score, best_checkpoint))
        if args.save_dict:
            experiment_name = experiment_path.name
            results = [
                {
                    "experiment": experiment_name,
                    "score": {
                        "mean": mean_ep_rew,
                        "sem": sem_ep_rew,
                    },
                    "cost_over_budget": {
                        "mean": mean_ep_cost_over_budget,
                        "sem": sem_ep_cost_over_budget,
                    },
                    "target_cpa_over_cpa": {
                        "mean": mean_ep_target_cpa_over_cpa,
                        "sem": sem_ep_target_cpa_over_cpa,
                    },
                    "total_conversions": {
                        "mean": np.mean(ep_total_conversions_list),
                        "sem": np.std(ep_total_conversions_list)
                        / np.sqrt(len(ep_total_conversions_list)),
                    },
                }
            ]
            if args.compute_topline:
                results.append(
                    {
                        "experiment": "oracle",
                        "score": {
                            "mean": mean_topline_ep_rew,
                            "sem": sem_topline_ep_rew,
                        },
                        "cost_over_budget": {
                            "mean": np.mean(ep_topline_cost_over_budget_list),
                            "sem": np.std(ep_topline_cost_over_budget_list)
                            / np.sqrt(len(ep_topline_cost_over_budget_list)),
                        },
                        "target_cpa_over_cpa": {
                            "mean": np.mean(ep_topline_target_cpa_over_cpa_list),
                            "sem": np.std(ep_topline_target_cpa_over_cpa_list)
                            / np.sqrt(len(ep_topline_target_cpa_over_cpa_list)),
                        },
                        "total_conversions": {
                            "mean": np.mean(ep_topline_total_conversions_list),
                            "sem": np.std(ep_topline_total_conversions_list)
                            / np.sqrt(len(ep_topline_total_conversions_list)),
                        },
                    }
                )
            if args.compute_flex_topline:
                results.append(
                    {
                        "experiment": "oracle_upgrade",
                        "score": {
                            "mean": mean_flex_topline_ep_rew,
                            "sem": sem_flex_topline_ep_rew,
                        },
                        "cost_over_budget": {
                            "mean": np.mean(ep_flex_topline_cost_over_budget_list),
                            "sem": np.std(ep_flex_topline_cost_over_budget_list)
                            / np.sqrt(len(ep_flex_topline_cost_over_budget_list)),
                        },
                        "target_cpa_over_cpa": {
                            "mean": np.mean(ep_flex_topline_target_cpa_over_cpa_list),
                            "sem": np.std(ep_flex_topline_target_cpa_over_cpa_list)
                            / np.sqrt(len(ep_flex_topline_target_cpa_over_cpa_list)),
                        },
                        "total_conversions": {
                            "mean": np.mean(ep_flex_topline_total_conversions_list),
                            "sem": np.std(ep_flex_topline_total_conversions_list)
                            / np.sqrt(len(ep_flex_topline_total_conversions_list)),
                        },
                    }
                )
            out_path = ROOT_DIR / "output" / "testing" / experiment_name
            out_path.mkdir(parents=True, exist_ok=True)
            with open(out_path / f"results_{start_ts}.json", "w") as f:
                json.dump(results, f, indent=4)

        if args.create_dataset:
            experiment_name = experiment_path.name
            out_path = ROOT_DIR / "output" / "testing" / experiment_name
            out_path.mkdir(parents=True, exist_ok=True)
            dataset_df = pd.DataFrame(dataset_list)
            dataset_df.to_parquet(
                out_path / f"dataset_seed_{args.seed}.parquet", index=False
            )
            print("Dataset saved at", out_path / "dataset.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main script to create a dataset of episodes with a trained agent"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to initialize the environment",
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
        default=None,
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
        "--min_checkpoint",
        type=float,
        default=0,
        help="Do not consider checkpoints before this number (to be fair across trainings)",
    )
    parser.add_argument(
        "--max_checkpoint",
        type=float,
        default=float("inf"),
        help="Do not consider checkpoints past this number (to be fair across trainings)",
    )
    parser.add_argument(
        "--compute_topline",
        action="store_true",
        default=False,
        help="Flag to compute the topline",
    )
    parser.add_argument(
        "--compute_flex_topline",
        action="store_true",
        default=False,
        help="Flag to compute the flex topline",
    )
    parser.add_argument(
        "--deterministic_conversion",
        action="store_true",
        default=False,
        help="Flag to use the deterministic conversion",
    )
    parser.add_argument(
        "--advertiser",
        type=int,
        default=None,
        help="Advertiser to evaluate",
    )
    parser.add_argument(
        "--compute_baseline",
        action="store_true",
        default=False,
        help="Flag to compute the baseline",
    )
    parser.add_argument(
        "--all_checkpoints",
        action="store_true",
        default=False,
        help="Flag to evaluate all checkpoints",
    )
    parser.add_argument(
        "--cpa_multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--two_slopes_action",
        action="store_true",
        default=False,
        help="Flag to use the two slopes action",
    )
    parser.add_argument(
        "--oracle_upgrade_cost_weight",
        type=float,
        default=0.5,
        help="Weight of the upper and lower cost in the flex oracle action",
    )
    parser.add_argument(
        "--create_dataset",
        action="store_true",
        default=False,
        help="Flag to create a dataset of episodes",
    )
    parser.add_argument(
        "--save_dict",
        action="store_true",
        default=False,
        help="Flag to save the results as a dictionary",
    )
    args = parser.parse_args()
    main(args)

"""Example:
python online/main_eval.py --experiment_path=output/training/ongoing/008_ppo_seed_0 \
    --checkpoint=5250000 --num_episodes=100
    
python online/main_eval.py --experiment_path=output/training/ongoing/013_ppo_seed_0_old_action \
    --checkpoint=28000000 --num_episodes=100

python online/main_eval.py --experiment_path=output/training/ongoing/017_ppo_seed_0_new_action_test \
    --num_episodes=100 --checkpoint=6500000

python online/main_eval.py --experiment_path=output/training/ongoing/016_ppo_seed_0_new_action_test \
    --num_episodes=100 --checkpoint=10250000 --deterministic 
    
python online/main_eval.py --experiment_path=output/training/ongoing/027_ppo_seed_0_dense_base_ranges_19_obs_exp_single_action_simplified \
    --num_episodes=100 --deterministic --checkpoint 13250000

# Best: 0.41
python online/main_eval.py --experiment_path=output/training/ongoing/029_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_simplified \
    --num_episodes=100 --deterministic --checkpoint=10750000
    
# Only 0.35 despite very strong offline performance?
python online/main_eval.py --experiment_path=output/training/ongoing/031_ppo_seed_0_dense_base_ranges_19_obs_exp_multi_action_simplified \
    --num_episodes=100 --deterministic --checkpoint=8000000
    
# Very good and still improving a little
python online/main_eval.py --experiment_path=output/training/ongoing/032_ppo_seed_0_dense_base_ranges_29_obs_exp_multi_action_simplified \
    --num_episodes=100 --deterministic --checkpoint=8500000

# Trained with realistic bidding, can keep up with the best in the simplified bidding
python online/main_eval.py --experiment_path=output/training/ongoing/033_ppo_seed_0_dense_base_ranges_29_obs_exp_multi_action \
    --num_episodes=100 --deterministic --checkpoint=3500000

# 0.4164
python online/main_eval.py --experiment_path=output/training/ongoing/035_ppo_seed_0_dense_larger_ranges_29_obs_exp_single_action_simplified_resume_029 \
    --num_episodes=100 --deterministic --checkpoint=12250000

# New best! 0.4199
python online/main_eval.py --experiment_path=output/training/ongoing/036_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_resume_029 \
    --num_episodes=100 --deterministic --checkpoint=12000000

# New best!!! 0.4312, local: 562.65 (so high)
python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --deterministic --checkpoint=5000000

# Also very good
python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --deterministic --checkpoint=5500000

# Also very good
python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --deterministic --checkpoint=5250000

# 0.4205
python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --deterministic --checkpoint=10000000
    
python online/main_eval.py --experiment_path=output/training/ongoing/038_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_noisy_env \
    --num_episodes=100 --deterministic

python online/main_eval.py --experiment_path=output/training/ongoing/038_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_noisy_env \
    --num_episodes=100 --deterministic --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_stochastic.json


python online/main_eval.py --experiment_path=output/training/ongoing/038_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_noisy_env \
    --num_episodes=100 --deterministic --checkpoint 4000000 --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse_stochastic.json
    
python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --deterministic --checkpoint=5000000 --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse_stochastic.json

python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --deterministic --checkpoint=5000000 --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_stochastic.json

python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --deterministic --checkpoint=5000000 --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json

# Only 0.3539
python online/main_eval.py --experiment_path=output/training/ongoing/037_ppo_seed_0_dense_base_ranges_29_obs_exp_3_actions \
    --num_episodes=100 --deterministic --checkpoint=7750000 \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config.json

python online/main_eval.py --experiment_path=output/training/ongoing/037_ppo_seed_0_dense_base_ranges_29_obs_exp_3_actions \
    --num_episodes=100 --deterministic \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config.json


python online/main_eval.py --experiment_path=output/training/ongoing/039_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_stoch_exposure \
    --num_episodes=100 --deterministic --checkpoint 5000000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config.json

python online/main_eval.py --experiment_path=output/training/ongoing/034_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action \
    --num_episodes=100 --deterministic --checkpoint=5000000 --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config.json \
        --compute_topline

# New best!!! 0.4446, local: 573.87
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --deterministic --compute_topline --checkpoint 1500000

# New best!!! 0.4485, local: 569.95
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --deterministic --compute_topline --checkpoint 2250000

# 583.44
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --deterministic --compute_topline --checkpoint 2500000

# New best!!! 0.4543, local: 590.59
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --deterministic --compute_topline --checkpoint 2750000

local: 579.30
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --deterministic --compute_topline --checkpoint 3000000

# Submission: 0.4403 local: 588.15
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --deterministic --compute_topline --checkpoint 3500000
    
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --deterministic --compute_topline --checkpoint 3750000
    
python online/main_eval.py --experiment_path=output/training/ongoing/040_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --deterministic --checkpoint 4500000

# Submission: 0.4396, local: 591.46
python online/main_eval.py --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_040 \
    --num_episodes=100 --deterministic --checkpoint 2850000
    
python online/main_eval.py --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_040 \
    --num_episodes=100 --deterministic --checkpoint 4220000
    
python online/main_eval.py --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_040 \
    --num_episodes=100 --deterministic --checkpoint 4730000
    
python online/main_eval.py --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_040 \
    --num_episodes=100 --deterministic --checkpoint 5000000

# Local: 588.33
python online/main_eval.py --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_040 \
    --num_episodes=100 --deterministic --checkpoint 4190000

python online/main_eval.py --experiment_path=output/training/ongoing/029_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_simplified \
    --num_episodes=100 --deterministic --checkpoint=6000000

# Submission: 0.4531, local: 591.95
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/002_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --deterministic --checkpoint 3150000

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/002_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --deterministic --checkpoint 3200000
    
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/002_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
    --num_episodes=100 --deterministic --checkpoint 2600000

# local:  586.67
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/004_onbc_seed_0_transformer \
    --num_episodes=100 --deterministic --checkpoint 2400000

# local:  590.52
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/004_onbc_seed_0_transformer \
    --num_episodes=100 --deterministic --checkpoint 2500000
        
# local: 587.09
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/004_onbc_seed_0_transformer \
    --num_episodes=100 --deterministic --checkpoint 2850000

# local: 594.21
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/004_onbc_seed_0_transformer \
    --num_episodes=100 --deterministic --checkpoint 3750000

# New best: 0.4572 local: 592.57
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/005_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_002 \
    --num_episodes=100 --deterministic --checkpoint 3350000
    
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/005_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_002 \
    --num_episodes=100 --deterministic --checkpoint 3350000 --compute_topline \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_stochastic.json
        
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/005_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_002 \
    --num_episodes=100 --deterministic --checkpoint 3350000 --compute_topline \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_stochastic.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/009_onbc_seed_0_small_pvals_auction_noise_simplified \
    --num_episodes=100 --deterministic --compute_topline \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_stochastic.json

# avg score: 36.95 avg_baseline_score: 30.65 avg_topline_score: 42.78
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/013_onbc_seed_0_new_data_simplified \
    --num_episodes=100 --deterministic --compute_topline --checkpoint 420000

# Best!!! 0.4598, avg score: 37.42, Score: 71.42
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/013_onbc_seed_0_new_data_simplified \
    --num_episodes=100 --deterministic --checkpoint 3650000

# Best!!! 0.4712, avg score: 38.29, score: 72.67
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/016_onbc_seed_0_new_data_simplified_small_lr_resume_013 \
    --num_episodes=100 --deterministic --checkpoint 3710000

avg score: 27.45 avg_baseline_score: 21.24 avg_topline_score: 29.48
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/016_onbc_seed_0_new_data_simplified_small_lr_resume_013 \
    --num_episodes=100 --deterministic --checkpoint 3710000 --compute_topline \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

25.86
python online/main_eval.py --algo ppo --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_29_obs_exp_single_action_realistic_auction_new_data \
    --num_episodes=1000 --deterministic --checkpoint 4350000 \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo ppo --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_29_obs_exp_single_action_realistic_auction_new_data \
    --num_episodes=100 --deterministic --checkpoint 5950000 \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
python online/main_eval.py --algo ppo --experiment_path=output/training/ongoing/042_ppo_seed_0_dense_29_obs_exp_single_action_realistic_auction_new_data \
    --num_episodes=100 --deterministic --checkpoint 7000000 \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/010_onbc_seed_0_all_pvals_auction_noise_simplified \
    --num_episodes=100 --deterministic --compute_topline \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_stochastic.json

# avg score: 36.63 avg_baseline_score: 30.65 avg_topline_score: 42.78
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/014_onbc_seed_0_transformer_new_data \
    --num_episodes=100 --deterministic --checkpoint 2400000
    
avg score: 38.41, test: 72.42 (baseline 39.05)
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/014_onbc_seed_0_transformer_new_data \
    --num_episodes=100 --deterministic --checkpoint 2700000
        
avg score: 36.74
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/017_onbc_seed_0_stoch_exposure_simplified_new_data \
    --num_episodes=100 --deterministic --checkpoint 3700000

avg score: 27.48 avg_baseline_score: 21.24 avg_topline_score: 30.49
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/017_onbc_seed_0_stoch_exposure_simplified_new_data \
    --num_episodes=100 --deterministic --checkpoint 3700000 --compute_topline\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

# 27.62
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/018_onbc_seed_0_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 750000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

score: 27.90
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/018_onbc_seed_0_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 3800000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

score 28.08
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/018_onbc_seed_0_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 5300000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

submission: 0.4665, local: 28.29
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/018_onbc_seed_0_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 2000000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/018_onbc_seed_0_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 2050000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json


python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/018_onbc_seed_0_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 1950000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

local: 28.80
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 3240000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

New best!!! 48.84, local: 28.88
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 2840000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

local: 28.18
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 2850000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

local: 28.71
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 2710000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.79
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 3860000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.85
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 3870000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.72
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 3640000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.54
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 3650000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

local: 28.39
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 2040000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.78
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 3840000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

local: 28.68
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 4630000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

local: 28.58
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 4640000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.58
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 4650000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.95
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 4840000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.73
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 4850000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.44
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 5300000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.42
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 5310000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.62
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 6330000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.73
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 2910000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.75, score: 61.33
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/020_onbc_seed_0_transformer_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 4050000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
28.94  
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/020_onbc_seed_0_transformer_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 4100000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
29.31
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/020_onbc_seed_0_transformer_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 4200000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

29.37
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/020_onbc_seed_0_transformer_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 4150000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.60     
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/020_onbc_seed_0_transformer_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 4450000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

29.33
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/020_onbc_seed_0_transformer_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 4500000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

        
New best!!! Submission: 0.4901, 29.17, local: 63.06
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 6590000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

To submit: 29.07
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 6560000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.78
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 6570000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

Submission: 0.4891,  29.10
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 6580000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.57
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 6350000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.71
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 6600000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.83
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 6610000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.40
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 8890000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
28.76    
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 8900000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.57
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 7360000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.88
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 7370000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.55
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 7710000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.45 
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 7720000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
28.38
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 7940000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
28.71   
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 7950000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

29.02
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 9730000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.57
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 9140000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.91
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 9150000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
28.40
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 9720000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.53  
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 9970000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.62
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --checkpoint 10000000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

29.03
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/022_onbc_seed_0_transformer_new_data_realistic_resume_020 \
    --num_episodes=100 --deterministic --checkpoint 4030000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

29.51
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/022_onbc_seed_0_transformer_new_data_realistic_resume_020 \
    --num_episodes=100 --deterministic --checkpoint 4450000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
29.35
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/022_onbc_seed_0_transformer_new_data_realistic_resume_020 \
    --num_episodes=100 --deterministic --checkpoint 5070000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

29.41
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/022_onbc_seed_0_transformer_new_data_realistic_resume_020 \
    --num_episodes=100 --deterministic --checkpoint 5080000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

29.11
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/022_onbc_seed_0_transformer_new_data_realistic_resume_020 \
    --num_episodes=100 --deterministic --checkpoint 5090000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

29.70, local: 59.22
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/022_onbc_seed_0_transformer_new_data_realistic_resume_020 \
    --num_episodes=100 --deterministic --checkpoint 5100000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

29.47      
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/022_onbc_seed_0_transformer_new_data_realistic_resume_020 \
    --num_episodes=100 --deterministic --checkpoint 5180000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

29.38
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/022_onbc_seed_0_transformer_new_data_realistic_resume_020 \
    --num_episodes=100 --deterministic --checkpoint 5190000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

28.92
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/022_onbc_seed_0_transformer_new_data_realistic_resume_020 \
    --num_episodes=100 --deterministic --checkpoint 5870000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

29.59
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/022_onbc_seed_0_transformer_new_data_realistic_resume_020 \
    --num_episodes=100 --deterministic --checkpoint 5880000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/023_onbc_seed_0_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 3100000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --checkpoint 4000000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --checkpoint 4010000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --checkpoint 4020000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --checkpoint 4810000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --checkpoint 4820000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --checkpoint 4830000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --checkpoint 4600000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --all_checkpoints --min_checkpoint 3310000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/022_onbc_seed_0_transformer_new_data_realistic_resume_020 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/021_onbc_seed_0_new_data_realistic_resume_018 \
    --num_episodes=100 --deterministic --all_checkpoints --min_checkpoint 2170000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
python online/main_eval.py --algo onbc_transformer --experiment_path=output/training/ongoing/020_onbc_seed_0_transformer_new_data_realistic \
    --num_episodes=100 --deterministic --checkpoint 4150000 --deterministic_conversion --advertiser 0\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_test.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --checkpoint 3850000 --deterministic_conversion --advertiser 0\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_test.json

# Best! Submission: 0.4948, offline: 29.52
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --checkpoint 4600000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json \
            --compute_baseline --compute_topline --compute_flex_topline
        
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --checkpoint 3850000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/031_onbc_seed_0_new_data_realistic_no_self_id \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/032_onbc_seed_0_new_data_realistic_no_self_id_resume_031 \
    --num_episodes=100 --deterministic --checkpoint 2320000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/032_onbc_seed_0_new_data_realistic_no_self_id_resume_031 \
    --num_episodes=100 --deterministic --all_checkpoints --min_checkpoint 4120000 --cpa_multiplier 0.95\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/039_onbc_seed_0_flex_two_slopes_oracle_145_keys_4_layers \
    --num_episodes=100 --deterministic\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

# Best! Submission: 0.4948, offline: 29.52
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --checkpoint 4600000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json \
            --compute_baseline --compute_topline --compute_flex_topline --two_slopes_action

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/046_onbc_seed_0_single_io_bid \
    --num_episodes=100 --deterministic --checkpoint 59500000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

# This seems to be better for low vals
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/052_onbc_seed_0_flex_two_slopes_oracle_60_obs_fix_oracle_resume_051 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

# Best submission all checkpoints
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
# Bug fix all checkpoints
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --all_checkpoints  --min_checkpoint 5000000 --max_checkpoint 10000000 \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

# Noisy bids all checkpoints       
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/054_onbc_seed_1_new_data_realistic_60_obs_resume_050_with_bid_noise \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/050_onbc_seed_0_new_data_realistic_60_obs_fix_oracle \
    --num_episodes=100 --deterministic --all_checkpoints --deterministic_conversion\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

# 29.46 stochastic, 29.32 deterministic
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/052_onbc_seed_0_flex_two_slopes_oracle_60_obs_fix_oracle_resume_051 \
    --num_episodes=100 --deterministic --checkpoint 7040000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

# 29.68 stochastic, 29.35 deterministic
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/052_onbc_seed_0_flex_two_slopes_oracle_60_obs_fix_oracle_resume_051 \
    --num_episodes=100 --deterministic --checkpoint 5580000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/052_onbc_seed_0_flex_two_slopes_oracle_60_obs_fix_oracle_resume_051 \
    --num_episodes=100 --deterministic --checkpoint 3580000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/052_onbc_seed_0_flex_two_slopes_oracle_60_obs_fix_oracle_resume_051 \
    --num_episodes=100 --deterministic --checkpoint 8520000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/052_onbc_seed_0_flex_two_slopes_oracle_60_obs_fix_oracle_resume_051 \
    --num_episodes=100 --deterministic --checkpoint 9820000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
# 29.62 stoch, 29.28 det  --  stochastic result not reproducible?
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=1000 --deterministic --checkpoint 13170000 --compute_flex_topline --two_slopes_action\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --checkpoint 10850000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

# Test specialists
# 0 0
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/059_onbc_seed_1_specialize_050_1000_3000_40_90 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_50_80_1500_2500.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --checkpoint 3270000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_50_80_1500_2500.json

# 0 1
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/060_onbc_seed_1_specialize_050_2000_4000_40_90 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_50_80_2500_3500.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --checkpoint 3270000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_50_80_2500_3500.json

# 0 2
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/061_onbc_seed_1_specialize_050_3000_5000_40_90 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_50_80_3500_5000.json
        
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --checkpoint 3270000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_50_80_3500_5000.json

# 1 0
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/062_onbc_seed_1_specialize_050_1000_3000_70_120 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_80_110_1500_2500.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --checkpoint 3270000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_80_110_1500_2500.json

# 1 1
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/063_onbc_seed_1_specialize_050_2000_4000_70_120 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_80_110_2500_3500.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --checkpoint 3270000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_80_110_2500_3500.json

# 1 2
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/065_onbc_seed_1_specialize_050_3000_5000_70_120 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_80_110_3500_5000.json
        
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --checkpoint 3270000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_80_110_3500_5000.json

# 2 0
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/064_onbc_seed_1_specialize_050_1000_3000_100_150 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_110_140_1500_2500.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --checkpoint 3270000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_110_140_1500_2500.json
        
# 2 1
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/067_onbc_seed_1_specialize_050_2000_4000_100_150 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_110_140_2500_3500.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --checkpoint 3270000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_110_140_2500_3500.json
  
# 2 2
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/066_onbc_seed_1_specialize_050_3000_5000_100_150 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_110_140_3500_5000.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --checkpoint 3270000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_110_140_3500_5000.json


python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
   
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
    --num_episodes=100 --deterministic --checkpoint 16868520\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
    --num_episodes=100 --deterministic --checkpoint 16338732\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=saved_model/ONBC/055_onbc_seed_0_single_io_bid_batch_state_20_envs_1000_subs \
    --num_episodes=100 --deterministic --checkpoint 3650000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/075_onbc_seed_0_new_data_realistic_60_obs_resume_055 \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/076_onbc_seed_42_resume_075_stoch_exp \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/081_onbc_seed_0_single_io_bid_100_subs_resume_077_stoch_exp \
    --num_episodes=100 --deterministic --all_checkpoints\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
    --num_episodes=100 --deterministic --all_checkpoints --min_checkpoint 21400000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/076_onbc_seed_42_resume_075_stoch_exp \
    --num_episodes=100 --deterministic --checkpoint 19280000 --seed 1\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/082_onbc_seed_42_resume_053_positive_noise \
    --num_episodes=100 --deterministic --all_checkpoints --min_checkpoint 14320000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse_positive_noise.json

# avg score: 24.36, avg c/b: 0.89, avg t_cpa/cpa: 0.92, avg_topline_score: 27.25 avg_flex_topline_score: 28.71
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --checkpoint 13170000 --compute_topline --compute_flex_topline\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse_positive_noise.json

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/083_onbc_seed_42_resume_053_negative_noise \
    --num_episodes=100 --deterministic --all_checkpoints --min_checkpoint 14320000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse_negative_noise.json

# avg score: 31.83, avg c/b: 0.96, avg t_cpa/cpa: 1.03, avg_topline_score: 36.34 avg_flex_topline_score: 37.38
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --checkpoint 13170000 --compute_topline --compute_flex_topline\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse_negative_noise.json


python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/720_onbc_seed_42_resume_053_advertiser_0_exclude_self \
    --num_episodes=50 --deterministic --all_checkpoints

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/721_onbc_seed_42_resume_053_advertiser_41_exclude_self \
    --num_episodes=50 --deterministic --all_checkpoints

python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/421_onbc_seed_42_resume_053_advertiser_1_exclude_self \
    --num_episodes=50 --deterministic --all_checkpoints

# Test max bid
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
    --num_episodes=100 --deterministic --checkpoint 13170000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse_max_bid.json

# Test if same result as online agent
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/052_onbc_seed_0_flex_two_slopes_oracle_60_obs_fix_oracle_resume_051 \
    --num_episodes=1 --deterministic --checkpoint 7040000 --deterministic_conversion --advertiser 0\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_test.json

# Create dataset
python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
    --num_episodes=100 --deterministic --checkpoint 4600000 --create_dataset --seed=42424242\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_sparse.json
        
python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/005_oil_seed_2_sparse_dataset_oracle_upgrade_two_slopes_medium_bids_lin_lr \
    --num_episodes=100 --deterministic \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json

python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/013_oil_seed_2_sparse_dataset_realistic_oracle_medium_bids_lin_lr \
    --num_episodes=100 --deterministic \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json

python online/main_eval.py --algo ppo --experiment_path=output/training/ongoing/015_ppo_seed_0_sparse_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr \
    --num_episodes=1000 --deterministic \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json

python online/main_eval.py --algo ppo --experiment_path=output/training/ongoing/016_ppo_seed_1_sparse_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr \
    --num_episodes=1000 --deterministic \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json

python online/main_eval.py --algo ppo --experiment_path=output/training/ongoing/017_ppo_seed_2_sparse_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr \
    --num_episodes=1000 --deterministic \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json

python online/main_eval.py --algo ppo --experiment_path=output/training/ongoing/022_ppo_seed_1_sparse_dataset_dense_reward_medium_bids_smaller_lin_lr \
    --num_episodes=1000 --deterministic \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json

python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/046_oil_seed_0_sparse_dataset_oracle_upgrade_two_slopes_medium_bids_lin_lr_cw_0_52 \
    --num_episodes=1000 --deterministic \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json

python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/046_oil_seed_0_sparse_dataset_oracle_upgrade_two_slopes_medium_bids_lin_lr_cw_0_52 \
    --num_episodes=1000 --deterministic --checkpoint 10000000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json

python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/046_oil_seed_0_sparse_dataset_oracle_upgrade_two_slopes_medium_bids_lin_lr_cw_0_52 \
    --num_episodes=1000 --deterministic --checkpoint 10000000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json

python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/039_oil_seed_0_dense_dataset_realistic_oracle_medium_bids_lin_lr \
    --num_episodes=1000 --deterministic\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_dense.json \
            --compute_flex_topline --two_slopes_action

python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/049_oil_seed_0_sparse_dataset_oracle_upgrade_single_io_bid_medium_bids_lin_lr \
    --num_episodes=1000 --deterministic --checkpoint 1220000\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_dense.json


python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/022_ppo_seed_1_sparse_dataset_dense_reward_medium_bids_smaller_lin_lr \
    --num_episodes=1000 --deterministic --checkpoint 10000000 --save_dict --compute_topline\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json
        
python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/023_ppo_seed_2_sparse_dataset_dense_reward_medium_bids_smaller_lin_lr \
    --num_episodes=1000 --deterministic --checkpoint 10000000 --save_dict --compute_flex_topline\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json

python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/016_ppo_seed_1_sparse_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr \
    --num_episodes=1000 --deterministic --checkpoint 10000000 --save_dict \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json
        
python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/015_ppo_seed_0_sparse_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr \
    --num_episodes=1000 --deterministic --checkpoint 10000000 --save_dict --compute_flex_topline --two_slopes_action\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json

python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/039_oil_seed_0_dense_dataset_realistic_oracle_medium_bids_lin_lr \
    --num_episodes=1000 --deterministic --checkpoint 10000000 --save_dict  --compute_topline\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_dense.json
        
python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/042_oil_seed_0_dense_dataset_oracle_upgrade_two_slopes_medium_bids_lin_lr \
    --num_episodes=1000 --deterministic --checkpoint 10000000 --save_dict --compute_flex_topline\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_dense.json

python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/017_ppo_seed_2_sparse_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr \
    --num_episodes=1000 --deterministic --checkpoint 10000000 --save_dict \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json
        
python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/018_ppo_seed_0_dense_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr \
    --num_episodes=1000 --deterministic --checkpoint 10000000 --save_dict \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_dense.json

python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/018_ppo_seed_0_dense_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr \
    --num_episodes=1000 --deterministic --checkpoint 10000000 --save_dict \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_dense.json

python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/069_ppo_seed_1_dense_dataset_dense_reward_medium_bids_smaller_lin_lr \
    --num_episodes=1000 --deterministic --checkpoint 10000000 --save_dict\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_dense.json

python online/main_eval.py --algo oil --experiment_path=saved_model/paper/051_oil_seed_2_sparse_dataset_flex_oracle_single_io_bid_medium_bids_lin_lr \
    --num_episodes=1000 --deterministic --checkpoint 10000000 --save_dict\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_sparse.json

python online/main_eval.py --algo oil --experiment_path=output/training/ongoing/052_oil_seed_0_dense_dataset_oracle_upgrade_single_io_bid_medium_bids_lin_lr \
    --num_episodes=1000 --deterministic --checkpoint 10000000 --save_dict\
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/env_configs/eval_config_dense.json

"""
