import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import torch
import pandas as pd
from definitions import ROOT_DIR, ENV_CONFIG_NAME
from envs.environment_factory import EnvironmentFactory
from helpers import (
    load_model,
    load_vecnormalize,
)

torch.manual_seed(0)

CKPT_CHOICE_CRITERION = "score"  # "rollout/ep_rew_mean", "rollout/solved"

budget_splits = [2500, 3500]
cpa_splits = [80, 110]

model_list = [
    {
        "category": 0,
        "budget_range": 0,
        "cpa_range": 0,
        "path": "059_onbc_seed_1_specialize_050_1000_3000_40_90",
        "checkpoint": 3530000,
    },
    {
        "category": 0,
        "budget_range": 0,
        "cpa_range": 1,
        "path": "062_onbc_seed_1_specialize_050_1000_3000_70_120",
        "checkpoint": 5900000,
    },
    {
        "category": 0,
        "budget_range": 0,
        "cpa_range": 2,
        "path": "064_onbc_seed_1_specialize_050_1000_3000_100_150",
        "checkpoint": 4450000,
    },
    {
        "category": 0,
        "budget_range": 1,
        "cpa_range": 0,
        "path": "026_onbc_seed_0_new_data_realistic_60_obs_resume_023",
        "checkpoint": 4600000,
    },
    {
        "category": 0,
        "budget_range": 1,
        "cpa_range": 1,
        "path": "063_onbc_seed_1_specialize_050_2000_4000_70_120",
        "checkpoint": 3310000,
    },
    {
        "category": 0,
        "budget_range": 1,
        "cpa_range": 2,
        "path": "067_onbc_seed_1_specialize_050_2000_4000_100_150",
        "checkpoint": 4570000,
    },
    {
        "category": 0,
        "budget_range": 2,
        "cpa_range": 0,
        "path": "026_onbc_seed_0_new_data_realistic_60_obs_resume_023",
        "checkpoint": 4600000,
    },
    {
        "category": 0,
        "budget_range": 2,
        "cpa_range": 1,
        "path": "065_onbc_seed_1_specialize_050_3000_5000_70_120",
        "checkpoint": 3240000,
    },
    {
        "category": 0,
        "budget_range": 2,
        "cpa_range": 2,
        "path": "066_onbc_seed_1_specialize_050_3000_5000_100_150",
        "checkpoint": 7880000,
    },
    {
        "category": 1,
        "budget_range": 0,
        "cpa_range": 0,
        "path": "059_onbc_seed_1_specialize_050_1000_3000_40_90",
        "checkpoint": 3530000,
    },
    {
        "category": 1,
        "budget_range": 0,
        "cpa_range": 1,
        "path": "062_onbc_seed_1_specialize_050_1000_3000_70_120",
        "checkpoint": 5900000,
    },
    {
        "category": 1,
        "budget_range": 0,
        "cpa_range": 2,
        "path": "064_onbc_seed_1_specialize_050_1000_3000_100_150",
        "checkpoint": 4450000,
    },
    {
        "category": 1,
        "budget_range": 1,
        "cpa_range": 0,
        "path": "053_onbc_seed_0_new_data_realistic_60_obs_resume_050",
        "checkpoint": 13170000,
    },
    {
        "category": 1,
        "budget_range": 1,
        "cpa_range": 1,
        "path": "063_onbc_seed_1_specialize_050_2000_4000_70_120",
        "checkpoint": 3310000,
    },
    {
        "category": 1,
        "budget_range": 1,
        "cpa_range": 2,
        "path": "067_onbc_seed_1_specialize_050_2000_4000_100_150",
        "checkpoint": 4570000,
    },
    {
        "category": 1,
        "budget_range": 2,
        "cpa_range": 0,
        "path": "053_onbc_seed_0_new_data_realistic_60_obs_resume_050",
        "checkpoint": 13170000,
    },
    {
        "category": 1,
        "budget_range": 2,
        "cpa_range": 1,
        "path": "065_onbc_seed_1_specialize_050_3000_5000_70_120",
        "checkpoint": 3240000,
    },
    {
        "category": 1,
        "budget_range": 2,
        "cpa_range": 2,
        "path": "066_onbc_seed_1_specialize_050_3000_5000_100_150",
        "checkpoint": 7880000,
    },
    {
        "category": 2,
        "budget_range": 0,
        "cpa_range": 0,
        "path": "059_onbc_seed_1_specialize_050_1000_3000_40_90",
        "checkpoint": 3530000,
    },
    {
        "category": 2,
        "budget_range": 0,
        "cpa_range": 1,
        "path": "062_onbc_seed_1_specialize_050_1000_3000_70_120",
        "checkpoint": 5900000,
    },
    {
        "category": 2,
        "budget_range": 0,
        "cpa_range": 2,
        "path": "064_onbc_seed_1_specialize_050_1000_3000_100_150",
        "checkpoint": 4450000,
    },
    {
        "category": 2,
        "budget_range": 1,
        "cpa_range": 0,
        "path": "053_onbc_seed_0_new_data_realistic_60_obs_resume_050",
        "checkpoint": 10850000,
    },
    {
        "category": 2,
        "budget_range": 1,
        "cpa_range": 1,
        "path": "063_onbc_seed_1_specialize_050_2000_4000_70_120",
        "checkpoint": 3310000,
    },
    {
        "category": 2,
        "budget_range": 1,
        "cpa_range": 2,
        "path": "067_onbc_seed_1_specialize_050_2000_4000_100_150",
        "checkpoint": 4570000,
    },
    {
        "category": 2,
        "budget_range": 2,
        "cpa_range": 0,
        "path": "053_onbc_seed_0_new_data_realistic_60_obs_resume_050",
        "checkpoint": 10850000,
    },
    {
        "category": 2,
        "budget_range": 2,
        "cpa_range": 1,
        "path": "065_onbc_seed_1_specialize_050_3000_5000_70_120",
        "checkpoint": 3240000,
    },
    {
        "category": 2,
        "budget_range": 2,
        "cpa_range": 2,
        "path": "066_onbc_seed_1_specialize_050_3000_5000_100_150",
        "checkpoint": 7880000,
    },
    {
        "category": 3,
        "budget_range": 0,
        "cpa_range": 0,
        "path": "059_onbc_seed_1_specialize_050_1000_3000_40_90",
        "checkpoint": 3530000,
    },
    {
        "category": 3,
        "budget_range": 0,
        "cpa_range": 1,
        "path": "062_onbc_seed_1_specialize_050_1000_3000_70_120",
        "checkpoint": 5900000,
    },
    {
        "category": 3,
        "budget_range": 0,
        "cpa_range": 2,
        "path": "064_onbc_seed_1_specialize_050_1000_3000_100_150",
        "checkpoint": 4450000,
    },
    {
        "category": 3,
        "budget_range": 1,
        "cpa_range": 0,
        "path": "053_onbc_seed_0_new_data_realistic_60_obs_resume_050",
        "checkpoint": 13170000,
    },
    {
        "category": 3,
        "budget_range": 1,
        "cpa_range": 1,
        "path": "063_onbc_seed_1_specialize_050_2000_4000_70_120",
        "checkpoint": 3310000,
    },
    {
        "category": 3,
        "budget_range": 1,
        "cpa_range": 2,
        "path": "067_onbc_seed_1_specialize_050_2000_4000_100_150",
        "checkpoint": 4570000,
    },
    {
        "category": 3,
        "budget_range": 2,
        "cpa_range": 0,
        "path": "053_onbc_seed_0_new_data_realistic_60_obs_resume_050",
        "checkpoint": 13170000,
    },
    {
        "category": 3,
        "budget_range": 2,
        "cpa_range": 1,
        "path": "065_onbc_seed_1_specialize_050_3000_5000_70_120",
        "checkpoint": 3240000,
    },
    {
        "category": 3,
        "budget_range": 2,
        "cpa_range": 2,
        "path": "066_onbc_seed_1_specialize_050_3000_5000_100_150",
        "checkpoint": 7880000,
    },
    {
        "category": 4,
        "budget_range": 0,
        "cpa_range": 0,
        "path": "059_onbc_seed_1_specialize_050_1000_3000_40_90",
        "checkpoint": 3530000,
    },
    {
        "category": 4,
        "budget_range": 0,
        "cpa_range": 1,
        "path": "062_onbc_seed_1_specialize_050_1000_3000_70_120",
        "checkpoint": 5900000,
    },
    {
        "category": 4,
        "budget_range": 0,
        "cpa_range": 2,
        "path": "064_onbc_seed_1_specialize_050_1000_3000_100_150",
        "checkpoint": 4450000,
    },
    {
        "category": 4,
        "budget_range": 1,
        "cpa_range": 0,
        "path": "053_onbc_seed_0_new_data_realistic_60_obs_resume_050",
        "checkpoint": 13170000,
    },
    {
        "category": 4,
        "budget_range": 1,
        "cpa_range": 1,
        "path": "063_onbc_seed_1_specialize_050_2000_4000_70_120",
        "checkpoint": 3310000,
    },
    {
        "category": 4,
        "budget_range": 1,
        "cpa_range": 2,
        "path": "067_onbc_seed_1_specialize_050_2000_4000_100_150",
        "checkpoint": 4570000,
    },
    {
        "category": 4,
        "budget_range": 2,
        "cpa_range": 0,
        "path": "053_onbc_seed_0_new_data_realistic_60_obs_resume_050",
        "checkpoint": 13170000,
    },
    {
        "category": 4,
        "budget_range": 2,
        "cpa_range": 1,
        "path": "065_onbc_seed_1_specialize_050_3000_5000_70_120",
        "checkpoint": 3240000,
    },
    {
        "category": 4,
        "budget_range": 2,
        "cpa_range": 2,
        "path": "066_onbc_seed_1_specialize_050_3000_5000_100_150",
        "checkpoint": 7880000,
    },
    {
        "category": 5,
        "budget_range": 0,
        "cpa_range": 0,
        "path": "059_onbc_seed_1_specialize_050_1000_3000_40_90",
        "checkpoint": 3530000,
    },
    {
        "category": 5,
        "budget_range": 0,
        "cpa_range": 1,
        "path": "062_onbc_seed_1_specialize_050_1000_3000_70_120",
        "checkpoint": 5900000,
    },
    {
        "category": 5,
        "budget_range": 0,
        "cpa_range": 2,
        "path": "064_onbc_seed_1_specialize_050_1000_3000_100_150",
        "checkpoint": 4450000,
    },
    {
        "category": 5,
        "budget_range": 1,
        "cpa_range": 0,
        "path": "053_onbc_seed_0_new_data_realistic_60_obs_resume_050",
        "checkpoint": 3270000,
    },
    {
        "category": 5,
        "budget_range": 1,
        "cpa_range": 1,
        "path": "063_onbc_seed_1_specialize_050_2000_4000_70_120",
        "checkpoint": 3310000,
    },
    {
        "category": 5,
        "budget_range": 1,
        "cpa_range": 2,
        "path": "067_onbc_seed_1_specialize_050_2000_4000_100_150",
        "checkpoint": 4570000,
    },
    {
        "category": 5,
        "budget_range": 2,
        "cpa_range": 0,
        "path": "053_onbc_seed_0_new_data_realistic_60_obs_resume_050",
        "checkpoint": 3270000,
    },
    {
        "category": 5,
        "budget_range": 2,
        "cpa_range": 1,
        "path": "065_onbc_seed_1_specialize_050_3000_5000_70_120",
        "checkpoint": 3240000,
    },
    {
        "category": 5,
        "budget_range": 2,
        "cpa_range": 2,
        "path": "066_onbc_seed_1_specialize_050_3000_5000_100_150",
        "checkpoint": 7880000,
    },
]


# Helper function to map budget to budget range
def get_budget_range(budget):
    if budget < 2500:
        return 0
    elif 2500 <= budget < 3500:
        return 1
    else:
        return 2


# Helper function to map CPA to CPA range
def get_cpa_range(cpa):
    if cpa < 80:
        return 0
    elif 80 <= cpa < 110:
        return 1
    else:
        return 2


# Function to retrieve the correct model
def get_model_vecnormalize_train_env(model_data_dict, model_df, category, budget, cpa):
    budget_range = get_budget_range(budget)
    cpa_range = get_cpa_range(cpa)
    model_row = model_df[
        (model_df["category"] == category)
        & (model_df["budget_range"] == budget_range)
        & (model_df["cpa_range"] == cpa_range)
    ]
    assert len(model_row) == 1
    path = model_row["path"].values[0]
    checkpoint = model_row["checkpoint"].values[0]
    model = model_data_dict[path][checkpoint]["model"]
    vecnormalize = model_data_dict[path][checkpoint]["vecnormalize"]
    train_env = model_data_dict[path][checkpoint]["train_env"]
    return model, vecnormalize, train_env


def main(args):
    env_config = json.load(open(args.eval_config_path, "r"))
    if args.compute_baseline:
        baseline_env = EnvironmentFactory.create(**env_config)
    if args.compute_topline:
        topline_env = EnvironmentFactory.create(**env_config)
    if args.compute_flex_topline:
        flex_topline_config = env_config.copy()
        flex_topline_config["flex_oracle"] = True
        flex_topline_config["two_slopes_action"] = args.two_slopes_action
        flex_topline_config["flex_oracle_cost_weight"] = args.flex_oracle_cost_weight
        flex_topline_env = EnvironmentFactory.create(**flex_topline_config)

    global model_list
    model_df = pd.DataFrame(model_list)
    unique_models_df = model_df.drop_duplicates(subset=["path", "checkpoint"])

    model_data_dict = {}
    for path, checkpoint in zip(
        unique_models_df["path"], unique_models_df["checkpoint"]
    ):
        experiment_path = ROOT_DIR / "saved_model" / "ONBC" / path
        model = load_model(
            args.algo,
            experiment_path,
            checkpoint,
        )
        train_env_config = json.load(open(experiment_path / ENV_CONFIG_NAME, "r"))
        train_env_config["bids_df_path"] = None
        train_env_config["pvalues_df_path"] = None
        train_env = EnvironmentFactory.create(**train_env_config)
        vecnormalize = load_vecnormalize(experiment_path, checkpoint, train_env)
        vecnormalize.training = False
        if path not in model_data_dict:
            model_data_dict[path] = {}
        model_data_dict[path][checkpoint] = {
            "model": model,
            "vecnormalize": vecnormalize,
            "train_env": train_env,
        }

    env = EnvironmentFactory.create(**env_config)

    # Collect rollouts and store them
    mean_ep_rew = 0
    mean_ep_cost_over_budget = 0
    mean_ep_target_cpa_over_cpa = 0
    mean_baseline_ep_rew = 0
    mean_topline_ep_rew = 0
    mean_flex_topline_ep_rew = 0
    for i in range(args.num_episodes):
        lstm_states = None
        ep_rew = 0
        baseline_ep_rew = 0
        topline_ep_rew = 0
        flex_topline_ep_rew = 0
        step = 0
        env.reset(seed=i + args.seed, advertiser=args.advertiser)
        if args.compute_baseline:
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
            flex_topline_env.unwrapped.episode_bids_df = env.unwrapped.episode_bids_df

        category = env.unwrapped.advertiser_category_dict[env.unwrapped.advertiser]
        budget = env.unwrapped.total_budget
        cpa = env.unwrapped.target_cpa
        model, vecnormalize, train_env = get_model_vecnormalize_train_env(
            model_data_dict, model_df, category, budget, cpa
        )

        episode_starts = np.ones((1,), dtype=bool)
        done = False
        while not done:
            pvalues, pvalue_sigmas = env.unwrapped.get_pvalues_mean_and_std()
            state_dict = env.unwrapped.get_state_dict(pvalues, pvalue_sigmas)
            state = train_env.unwrapped.get_state(state_dict)
            norm_obs = vecnormalize.normalize_obs(state)
            action, _ = model.predict(
                norm_obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=args.deterministic,
            )
            bid_coef = train_env.unwrapped.compute_bid_coef(
                action, pvalues, pvalue_sigmas
            )
            bids = bid_coef * env.unwrapped.target_cpa
            _, reward, terminated, truncated, info = env.unwrapped.place_bids(
                bids, pvalues, pvalue_sigmas
            )

            if args.compute_baseline:
                baseline_action = baseline_env.unwrapped.get_baseline_action()
                _, baseline_rewards, _, _, _ = baseline_env.step(baseline_action)
                baseline_ep_rew += baseline_rewards

            if args.compute_topline:
                topline_action = topline_env.unwrapped.get_oracle_action()
                # topline_action = (
                #     topline_env.unwrapped.get_simplified_oracle_action()
                # )
                _, topline_rewards, _, _, _ = topline_env.step(topline_action)
                topline_ep_rew += topline_rewards

            if args.compute_flex_topline:
                flex_topline_action = (
                    flex_topline_env.unwrapped.get_flex_oracle_action()
                )
                _, flex_topline_rewards, _, _, _ = flex_topline_env.step(
                    flex_topline_action
                )
                flex_topline_ep_rew += flex_topline_rewards

            done = terminated or truncated
            episode_starts = done
            ep_rew += reward
            step += 1
        mean_ep_rew = (mean_ep_rew * i + ep_rew) / (i + 1)
        mean_ep_cost_over_budget = (
            mean_ep_cost_over_budget * i + info["cost_over_budget"]
        ) / (i + 1)
        mean_ep_target_cpa_over_cpa = (
            mean_ep_target_cpa_over_cpa * i + info["target_cpa_over_cpa"]
        ) / (i + 1)
        if args.compute_baseline:
            mean_baseline_ep_rew = (mean_baseline_ep_rew * i + baseline_ep_rew) / (
                i + 1
            )
        if args.compute_topline:
            mean_topline_ep_rew = (mean_topline_ep_rew * i + topline_ep_rew) / (i + 1)

        if args.compute_flex_topline:
            mean_flex_topline_ep_rew = (
                mean_flex_topline_ep_rew * i + flex_topline_ep_rew
            ) / (i + 1)
        str_out = "Ep: {} ep rew: {:.2f}, avg score: {:.2f}, avg c/b: {:.2f}, avg t_cpa/cpa: {:.2f},".format(
            i,
            ep_rew,
            mean_ep_rew,
            mean_ep_cost_over_budget,
            mean_ep_target_cpa_over_cpa,
        )
        if args.compute_baseline:
            str_out += " avg_baseline_score: {:.2f}".format(mean_baseline_ep_rew)
        if args.compute_topline:
            str_out += " avg_topline_score: {:.2f}".format(mean_topline_ep_rew)
        if args.compute_flex_topline:
            str_out += " avg_flex_topline_score: {:.2f}".format(
                mean_flex_topline_ep_rew
            )
        print(str_out)
    env.close()

    out_name = f"score_{mean_ep_rew:.4f}".replace(".", "_")
    out_dir = ROOT_DIR / "output" / "testing" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save model_list, budget_range, cpa_range and eval_config as json
    with open(out_dir / "model_list.json", "w") as f:
        json.dump(model_list, f, indent=4)
    with open(out_dir / "budget_range.json", "w") as f:
        json.dump(budget_splits, f, indent=4)
    with open(out_dir / "cpa_range.json", "w") as f:
        json.dump(cpa_splits, f, indent=4)
    with open(out_dir / "eval_config.json", "w") as f:
        json.dump(env_config, f, indent=4)


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
        "--eval_config_path",
        type=str,
        default=ROOT_DIR / "env_configs" / "eval_config_realistic.json",
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
        "--flex_oracle_cost_weight",
        type=float,
        default=0.5,
        help="Weight of the upper and lower cost in the flex oracle action",
    )
    args = parser.parse_args()
    main(args)

"""Example:
python online/main_eval_ensemble.py --num_episodes=100 --deterministic \
    --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json

"""
