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
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from bidding_train_env.strategy import PlayerBiddingStrategy

torch.manual_seed(0)

CKPT_CHOICE_CRITERION = "score"  # "rollout/ep_rew_mean", "rollout/solved"


def main(args):

    env_config = json.load(open(args.eval_config_path, "r"))
    env = EnvironmentFactory.create(**env_config, detailed_info=True)
    agent = PlayerBiddingStrategy()

    # Collect rollouts and store them
    mean_ep_rew = 0
    mean_ep_cost_over_budget = 0
    mean_ep_target_cpa_over_cpa = 0
    for i in range(args.num_episodes):
        ep_rew = 0
        step = 0
        historyPValueInfo = []
        historyBid = []
        historyAuctionResult = []
        historyImpressionResult = []
        historyLeastWinningCost = []

        _, info = env.reset(seed=i + args.seed, advertiser=args.advertiser)
        agent.budget = env.unwrapped.total_budget
        agent.cpa = env.unwrapped.target_cpa
        agent.category = env.unwrapped.advertiser_category_dict[env.unwrapped.advertiser]
        agent.reset()

        done = False

        while not done:
            pValue = info["pvalues"]
            pValueSigma = info["pvalue_sigmas"]
            agent.remaining_budget = env.unwrapped.remaining_budget
            bids = agent.bidding(
                step,
                pValue,
                pValueSigma,
                historyPValueInfo,
                historyBid,
                historyAuctionResult,
                historyImpressionResult,
                historyLeastWinningCost,
            )
            _, reward, terminated, truncated, info = env.unwrapped.place_bids(
                bids, pValue, pValueSigma
            )
            info["cost"][np.logical_and(info["success"] == 1, info["exposure"]==0)] == 1
            pvalue_info = np.stack([pValue, pValueSigma]).T
            historyPValueInfo.append(pvalue_info)
            historyBid.append(bids)
            auction_result = np.stack(
                [
                    info["success"],
                    info["slot"],
                    info["cost"],
                ]
            ).T
            historyAuctionResult.append(auction_result)
            impression_result = np.stack(
                [
                    info["exposure"],
                    info["conversion"],
                ]
            ).T
            historyImpressionResult.append(impression_result)
            historyLeastWinningCost.append(info["least_winning_cost"])
            done = terminated or truncated
            ep_rew += reward
            step += 1
        mean_ep_rew = (mean_ep_rew * i + ep_rew) / (i + 1)
        mean_ep_cost_over_budget = (
            mean_ep_cost_over_budget * i + info["cost_over_budget"]
        ) / (i + 1)
        mean_ep_target_cpa_over_cpa = (
            mean_ep_target_cpa_over_cpa * i + info["target_cpa_over_cpa"]
        ) / (i + 1)

        str_out = "Ep: {} ep rew: {:.2f}, avg score: {:.2f}, avg c/b: {:.2f}, avg t_cpa/cpa: {:.2f},".format(
            i,
            ep_rew,
            mean_ep_rew,
            mean_ep_cost_over_budget,
            mean_ep_target_cpa_over_cpa,
        )
        print(str_out)
    env.close()


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
        "--advertiser_categories",
        type=int,
        nargs="+",
        default=None,
        help="Advertiser categories where to sample from, if None all are used",
    )
    args = parser.parse_args()
    main(args)

"""Example:
python online/main_strategy.py --seed 0 --algo ppo --eval_config_path env_configs/eval_config_realistic.json \
    --num_episodes 100 --deterministic
"""
