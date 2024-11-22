import pathlib
import sys
import wandb

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import torch
import pandas as pd
from bidding_train_env.common.utils import (
    normalize_state,
    normalize_reward,
    save_normalize_dict,
    apply_norm_state,
)
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.bc.behavior_clone import BC
from definitions import ROOT_DIR
import logging
import pickle

np.set_printoptions(suppress=True, precision=4)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_bc(only_eval=False):
    """
    Run bc model training and evaluation.
    """
    if only_eval:
        eval_model()
    else:
        train_model()


def eval_model():
    seed = 0
    batch_size = 1
    dataset_name = "final"  # official, final
    algo = "bc"
    experiment_name = f"bc_training_{seed}_dataset_{dataset_name}"
    model_path = (
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

    if dataset_name == "official":
        df_str = ""
    elif dataset_name == "final":
        df_str = "_final"
    else:
        raise ValueError("Invalid dataset name")
    test_data_path = f"/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/traffic/offline_rl_data{df_str}/period-27_27_offline_rl_data.parquet"

    test_data = pd.read_parquet(test_data_path)

    test_data["state"] = test_data["state"].apply(np.array)
    test_data["next_state"] = test_data["next_state"].apply(np.array)

    model = torch.jit.load(model_path).actor

    with open(normalize_path, "rb") as f:
        normalize_dict = pickle.load(f)

    test_loss = 0
    total_reward = 0
    num_samples = len(test_data)
    for start_idx in range(0, num_samples, batch_size):
        batch = test_data.iloc[start_idx : start_idx + batch_size]
        states = torch.tensor(np.stack(batch["state"].values), dtype=torch.float32)
        actions = torch.tensor(np.stack(batch["action"].values), dtype=torch.float32)[
            :, None
        ]
        states = apply_norm_state(states, normalize_dict)
        with torch.no_grad():
            pred_actions = model(states)
            loss = torch.nn.functional.mse_loss(pred_actions, actions).item()
            test_loss += loss * len(batch)
            total_reward += batch["reward_continuous"].sum()

    logger.info(
        f"Test Loss: {test_loss / num_samples}, Total Reward: {total_reward / 48}"
    )


def train_model():
    """
    Train BC model and log losses.
    """
    seed = 2
    dataset_name = "official"  # official, final

    # Initialize Weights & Biases
    wandb.init(
        project="baselines",
        name=f"bc_training_{seed}_dataset_{dataset_name}",
        config={
            "batch_size": 100,
            "step_num": 100000,
            "state_dim": 60,
            "log_interval": 100,
            "test_interval": 100,
            "lr": 1e-4,
            "save_every": 10000,
            "seed": seed,
            "dataset": dataset_name,
        },
    )

    experiment_name = wandb.run.name
    if dataset_name == "official":
        df_str = ""
    elif dataset_name == "final":
        df_str = "_final"
    else:
        raise ValueError("Invalid dataset name")
    train_data_path = f"/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/traffic/offline_rl_data{df_str}/period-7_26_offline_rl_data.parquet"
    test_data_path = f"/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/traffic/offline_rl_data{df_str}/period-27_27_offline_rl_data.parquet"

    # Load training and test data
    training_data = pd.read_parquet(train_data_path)
    test_data = pd.read_parquet(test_data_path)

    training_data["state"] = training_data["state"].apply(np.array)
    training_data["next_state"] = training_data["next_state"].apply(np.array)
    test_data["state"] = test_data["state"].apply(np.array)
    test_data["next_state"] = test_data["next_state"].apply(np.array)

    normalize_indices = list(range(wandb.config.state_dim))
    is_normalize = True

    normalize_dic = normalize_state(
        training_data, wandb.config.state_dim, normalize_indices
    )
    normalize_reward(training_data, "reward_continuous")
    save_normalize_dict(normalize_dic, f"output/offline/{experiment_name}")

    model = BC(
        dim_obs=wandb.config.state_dim,
        actor_lr=wandb.config.lr,
        network_random_seed=wandb.config.seed,
    )
    replay_buffer = ReplayBuffer(device="cuda" if model.use_cuda else "cpu")
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)

    logger.info(f"Replay buffer size: {len(replay_buffer.memory)}")

    step_num = wandb.config.step_num
    batch_size = wandb.config.batch_size
    log_interval = wandb.config.log_interval
    test_interval = wandb.config.test_interval

    for step in range(step_num):
        # Sample a batch and perform a training step
        states, actions, _, _, _ = replay_buffer.sample(batch_size)
        train_loss = model.step(states, actions)
        wandb.log({"training_loss": np.mean(train_loss)}, step=step)

        if step % log_interval == 0:
            logger.info(f"Step: {step} Training Loss: {np.mean(train_loss)}")

        # Test periodically
        if step % test_interval == 0:
            test_loss = evaluate_model(
                model, test_data, batch_size, is_normalize, normalize_dic
            )
            wandb.log({"test_loss": test_loss}, step=step)
            logger.info(f"Step: {step} Test Loss: {test_loss}")

        if step % wandb.config.save_every == 0:
            model.save_jit(
                ROOT_DIR / "output" / "offline" / experiment_name / f"model_{step}"
            )

    # Save the model
    model.save_jit(ROOT_DIR / "output" / "offline" / experiment_name / "model_final")
    wandb.finish()


def evaluate_model(model, test_data, batch_size, is_normalize, normalize_dic):
    """
    Evaluate model on test data.
    """
    test_loss = 0
    num_samples = len(test_data)
    for start_idx in range(0, num_samples, batch_size):
        batch = test_data.iloc[start_idx : start_idx + batch_size]
        states = torch.tensor(np.stack(batch["state"].values), dtype=torch.float32)
        actions = torch.tensor(np.stack(batch["action"].values), dtype=torch.float32)[
            :, None
        ]
        if is_normalize:
            states = apply_norm_state(states, normalize_dic)
        with torch.no_grad():
            pred_actions = model.take_actions(states)
            loss = torch.nn.functional.mse_loss(pred_actions, actions).item()
            test_loss += loss * len(batch)
    return test_loss / num_samples


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = (
            row.state if not is_normalize else row.normalize_state,
            row.action,
            row.reward if not is_normalize else row.normalize_reward,
            row.next_state if not is_normalize else row.normalize_nextstate,
            row.done,
        )
        # Remove all rows with done == 1
        if done != 1:
            replay_buffer.push(
                np.array(state),
                np.array([action]),
                np.array([reward]),
                np.array(next_state),
                np.array([done]),
            )
        else:
            replay_buffer.push(
                np.array(state),
                np.array([action]),
                np.array([reward]),
                np.zeros_like(state),
                np.array([done]),
            )


if __name__ == "__main__":
    run_bc(only_eval=False)
