import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import pandas as pd
import os
import time
import json
import numpy as np
from online.envs.environment_factory import EnvironmentFactory
from definitions import ROOT_DIR


def compute_alpha(df, ts, budget, cpa, category):
    filtered_df = df[
        (df["timeStepIndex"] == ts)
        & (df["advertiserCategoryIndex"] == category)
        & (df["cum_cost"] > budget)
    ]
    if not filtered_df.empty:
        alpha = filtered_df.iloc[0]["realCPA"] / cpa
    else:
        alpha = 1
    return np.log(min(1.5, alpha))


if __name__ == "__main__":

    algo = "online_lp"
    num_episodes = 1000
    seed = 0
    dataset = "dense"  # "dense", "sparse"
    experiment_name = f"online_lp_{dataset}_seed_{seed}"
    df_path = ROOT_DIR / "output" / "offline" / experiment_name / "data.csv"
    eval_config_path = ROOT_DIR / "data" / "env_configs" / f"eval_config_{dataset}.json"

    lp_df = pd.read_csv(df_path)

    start_ts = int(time.time())

    with open(eval_config_path, "r") as f:
        eval_config = json.load(f)

    env = EnvironmentFactory.create(**eval_config)

    ep_rew_list = []
    ep_cost_over_budget_list = []
    ep_target_cpa_over_cpa_list = []
    ep_total_conversions_list = []
    for i in range(num_episodes):
        ep_rew = 0
        step = 0
        obs, _ = env.reset(seed=i)
        done = False
        cpa = env.target_cpa
        category = env.category
        while not done:
            budget_left = env.remaining_budget
            action = compute_alpha(
                df=lp_df, ts=step, budget=budget_left, cpa=cpa, category=category
            )
            obs, rew, terminated, truncated, info = env.step(action)
            ep_rew += rew
            step += 1
            done = terminated or truncated
        ep_rew_list.append(ep_rew)
        ep_cost_over_budget_list.append(info["cost_over_budget"])
        ep_target_cpa_over_cpa_list.append(info["target_cpa_over_cpa"])
        ep_total_conversions_list.append(info["conversions"])
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
        print(str_out)
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
    out_path = ROOT_DIR / "output" / "testing" / experiment_name
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / f"results_{start_ts}.json", "w") as f:
        json.dump(results, f, indent=4)
    info_dict = {
        "num_episodes": num_episodes,
        "start_ts": start_ts,
        "algo": algo,
        "df_path": str(df_path),
        "eval_config_path": str(eval_config_path),
    }
    with open(out_path / f"info_{start_ts}.json", "w") as f:
        json.dump(info_dict, f, indent=4)
