import pathlib
import json
from stable_baselines3 import SAC, TD3, PPO
from sb3_contrib import RecurrentPPO
from online.algos.ppo import BCPPO
from online.algos.oil import OnPolicyBC, OnPolicyTransformerBC


ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATTERN = "rl_model_*_steps.zip"
ENV_PATTERN = "rl_model_vecnormalize_*_steps.pkl"
ENV_CONFIG_NAME = "env_config.json"
ACT_CONFIG_PATH = ROOT_DIR / "data" / "act_configs"
OBS_CONFIG_PATH = ROOT_DIR / "data" / "obs_configs"

ALGO_CLASS_DICT = {
    "ppo": PPO,
    "bc_ppo": BCPPO,
    "recurrent_ppo": RecurrentPPO,
    "sac": SAC,
    "td3": TD3,
    "oil": OnPolicyBC,
    "oil_transformer": OnPolicyTransformerBC,
}

ALGO_TB_DIR_NAME_DICT = {
    "ppo": "PPO_0",
    "oil": "OnPolicyAlgorithm_0",
    "oil_transformer": "OnPolicyAlgorithm_0",
}

DEFAULT_ACT_KEYS = json.load(open(ACT_CONFIG_PATH / "act_1_key.json", "r"))
DEFAULT_OBS_KEYS = json.load(open(OBS_CONFIG_PATH / "obs_60_keys.json", "r"))
DEFAULT_RWD_WEIGHTS = {
    "dense": 0.0,
    "sparse": 1.0,
}

NO_HISTORY_KEYS = [
    "time_left",
    "budget_left",
    "budget",
    "cpa",
    "category",
    "total_conversions",
    "total_cost",
    "total_cpa",
    "current_pvalues_mean",
    "current_pvalues_90_pct",
    "current_pvalues_99_pct",
    "current_pv_num",
]

HISTORY_KEYS = [
    "least_winning_cost_mean",
    "least_winning_cost_10_pct",
    "least_winning_cost_01_pct",
    "cpa_exceedence_rate",
    "pvalues_mean",
    "conversion_mean",
    "conversion_count",
    "bid_success_mean",
    "successful_bid_position_mean",
    "bid_over_lwc_mean",
    "pv_over_lwc_mean",
    "pv_over_lwc_90_pct",
    "pv_over_lwc_99_pct",
    "pv_num",
    "exposure_count",
    "cost_sum",
]

HISTORY_AND_SLOT_KEYS = [
    "bid_mean",
    "cost_mean",
    "bid_success_count",
    "exposure_mean",
]