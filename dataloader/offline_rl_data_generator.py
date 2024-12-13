# Append root directory to path
import sys
import pathlib

from online.envs.helpers import safe_mean

import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import json
import pandas as pd
from definitions import ROOT_DIR
from online.envs.bidding_env import BiddingEnv


dataset = "dense"  # "dense", "sparse"
obs_type = "obs_60_keys"
num_advertisers = 48
period_start = 7
period_end = 26  # included


def compute_alpha(ad_df, ts, target_cpa):
    bids = ad_df[ad_df.timeStepIndex == ts].bid.to_numpy()
    pvalues = ad_df[ad_df.timeStepIndex == ts].pValue.to_numpy()
    alpha = safe_mean(bids[pvalues > 0] / pvalues[pvalues > 0]) / target_cpa
    return np.log(alpha) if alpha > np.exp(-10) else -10


if __name__ == "__main__":
    training_data_rows = []
    for period in range(period_start, period_end + 1):
        print(f"Processing period {period}")
        pvalues_path = (
            ROOT_DIR
            / "data"
            / "traffic"
            / f"online_rl_data_{dataset}"
            / f"period-{period}_pvalues.parquet"
        )
        bids_path = (
            ROOT_DIR
            / "data"
            / "traffic"
            / f"online_rl_data{dataset}"
            / f"period-{period}_bids.parquet"
        )
        data_path = (
            ROOT_DIR
            / "data"
            / "traffic"
            / f"raw_traffic_{dataset}_parquet"
            / f"period-{period}.parquet"
        )
        raw_df = pd.read_parquet(data_path)
        raw_df["real_cost"] = raw_df["cost"] * raw_df["isExposed"]
        info_df = (
            raw_df.groupby("advertiserNumber")
            .agg(
                {
                    "budget": "mean",
                    "CPAConstraint": "mean",
                    "advertiserCategoryIndex": "mean",
                }
            )
            .reset_index()
        )

        with open(ROOT_DIR / "data" / "obs_configs" / f"{obs_type}.json") as f:
            obs_keys = json.load(f)

        env = BiddingEnv(
            pvalues_path,
            bids_path,
            obs_keys=obs_keys,
            exclude_self_bids=True,
            deterministic_conversion=False,
            single_io_bid=False,
            oracle_upgrade=False,
            stochastic_exposure=False,
        )

        # Impersonate one advertiser at a time and collect the transitions
        for advertiser_id in range(num_advertisers):
            print(f"Processing advertiser {advertiser_id} from period {period}")
            ad_df = raw_df[raw_df["advertiserNumber"] == advertiser_id]
            ad_info = info_df[info_df.advertiserNumber == advertiser_id]
            budget = ad_info["budget"].item()
            target_cpa = ad_info["CPAConstraint"].item()
            category = ad_info["advertiserCategoryIndex"].item()

            obs, _ = env.reset(
                budget=budget, target_cpa=target_cpa, advertiser=advertiser_id
            )
            done = False
            while not done:
                alpha = compute_alpha(ad_df, env.time_step, target_cpa)
                next_obs, reward, terminated, truncated, info = env.step(alpha)
                done = terminated or truncated
                training_data_rows.append(
                    {
                        "deliveryPeriodIndex": period,
                        "advertiserNumber": advertiser_id,
                        "advertiserCategoryIndex": category,
                        "budget": budget,
                        "CPAConstraint": target_cpa,
                        "realAllCost": env.history_info["cost_sum"][-1],
                        "realAllConversion": env.history_info["conversion_count"][-1],
                        "timeStepIndex": env.time_step - 1,
                        "state": obs.tolist(),
                        "action": alpha,
                        "reward": info["sparse"],
                        "reward_continuous": info["dense"],
                        "done": done,
                    }
                )
                obs = next_obs
        training_data = pd.DataFrame(training_data_rows)
        training_data = training_data.sort_values(
            by=["deliveryPeriodIndex", "advertiserNumber", "timeStepIndex"]
        )

        training_data["next_state"] = training_data.groupby(
            ["deliveryPeriodIndex", "advertiserNumber"]
        )["state"].shift(-1)
        training_data.loc[training_data["done"] == 1, "next_state"] = None

    out_dir = ROOT_DIR / "data" / "traffic" / f"offline_rl_data_{dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    training_data.to_parquet(
        out_dir / f"period-{period_start}_{period_end}_offline_rl_data.parquet"
    )
