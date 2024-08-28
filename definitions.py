import pathlib


ROOT_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATTERN = "rl_model_*_steps.zip"
ENV_PATTERN = "rl_model_vecnormalize_*_steps.pkl"
ENV_CONFIG_NAME = "env_config.json"