import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import torch
import pickle
import json
import time
import numpy as np
from definitions import ROOT_DIR
from online.envs.environment_factory import EnvironmentFactory
from bidding_train_env.common.utils import apply_norm_state


if __name__ == "__main__":

    algo = "bc"
    num_episodes = 1000
    experiment_name = "bc_training_2_dataset_final"
    exp_path = (
        ROOT_DIR
        / "output"
        / "offline"
        / experiment_name
        / "model_final"
        / f"{algo}_model.pth"
    )
    normalize_path = (
        ROOT_DIR / "output" / "offline" / experiment_name / "normalize_dict.pkl"
    )
    eval_config_path = ROOT_DIR / "data" / "env_configs" / "eval_config_realistic.json"

    start_ts = int(time.time())
    model = torch.jit.load(exp_path)
    with open(normalize_path, "rb") as f:
        normalize_dict = pickle.load(f)

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
        while not done:
            obs = apply_norm_state(obs, normalize_dict)
            with torch.no_grad():
                obs = torch.tensor(obs, dtype=torch.float32)
                action = model(obs).cpu().numpy()
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
