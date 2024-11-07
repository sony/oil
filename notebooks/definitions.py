import pathlib
from stable_baselines3 import SAC, TD3, PPO
from sb3_contrib import RecurrentPPO
from online.train.ppo import BCPPO
from online.algos.on_policy_bc import OnPolicyBC, OnPolicyTransformerBC


ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATTERN = "rl_model_*_steps.zip"
ENV_PATTERN = "rl_model_vecnormalize_*_steps.pkl"
ENV_CONFIG_NAME = "env_config.json"

ALGO_CLASS_DICT = {
    "ppo": PPO,
    "bc_ppo": BCPPO,
    "recurrent_ppo": RecurrentPPO,
    "sac": SAC,
    "td3": TD3,
    "oil": OnPolicyBC,
    "oil_transformer": OnPolicyTransformerBC
}

ALGO_TB_DIR_NAME_DICT = {
    "ppo": "PPO_0",
    "oil": "OnPolicyAlgorithm_0",
    "oil_transformer": "OnPolicyAlgorithm_0",
}