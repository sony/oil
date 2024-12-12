import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import wandb
import torch
import numpy as np
import logging
import pandas as pd
import ast
import wandb
from offline.common.utils import (
    normalize_state,
    normalize_reward,
    save_normalize_dict,
    apply_norm_state,
)
from offline.iql.replay_buffer import ReplayBuffer
from offline.iql.iql import IQL
from definitions import ROOT_DIR


np.set_printoptions(suppress=True, precision=4)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_iql():
    """
    Run IQL model training and evaluation.
    """
    train_iql_model()


def train_iql_model():
    """
    Train the IQL model with periodic testing and wandb logging.
    """
    seed = 2
    reward_continuous = False
    dataset_name = "sparse"  # dense, sparse
    # Initialize wandb
    wandb.init(
        project="baselines",
        name=f"iql_training_{seed}_reward_continuous_{reward_continuous}_dataset_{dataset_name}",
        config={
            "batch_size": 100,
            "step_num": 100000,
            "state_dim": 60,
            "log_interval": 100,
            "test_interval": 100,
            "lr": 1e-4,
            "save_every": 10000,
            "seed": seed,
            "reward_continuous": reward_continuous,
            "dataset": dataset_name,
        },
    )

    experiment_name = wandb.run.name
    if dataset_name == "dense":
        df_str = ""
    elif dataset_name == "sparse":
        df_str = "_sparse"
    else:
        raise ValueError("Invalid dataset name")
    train_data_path = f"/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/traffic/offline_rl_data{df_str}/period-7_26_offline_rl_data.parquet"
    test_data_path = f"/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/traffic/offline_rl_data{df_str}/period-27_27_offline_rl_data.parquet"

    # Load and preprocess training and test data
    training_data = pd.read_parquet(train_data_path)
    test_data = pd.read_parquet(test_data_path)

    training_data["state"] = training_data["state"].apply(np.array)
    training_data["next_state"] = training_data["next_state"].apply(np.array)

    test_data["state"] = test_data["state"].apply(np.array)
    test_data["next_state"] = test_data["next_state"].apply(np.array)

    # Normalize data
    normalize_indices = list(range(wandb.config.state_dim))
    is_normalize = True

    normalize_dic = normalize_state(
        training_data, wandb.config.state_dim, normalize_indices
    )
    reward_type = "reward_continuous" if reward_continuous else "reward"
    normalize_reward(training_data, reward_type)
    save_normalize_dict(normalize_dic, f"output/offline/{experiment_name}")

    # Initialize model
    model = IQL(
        dim_obs=wandb.config.state_dim,
        V_lr=wandb.config.lr,
        critic_lr=wandb.config.lr,
        actor_lr=wandb.config.lr,
        network_random_seed=wandb.config.seed,
    )

    # Build replay buffer
    replay_buffer = ReplayBuffer(device="cuda" if model.use_cuda else "cpu")
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)

    logger.info(f"Replay buffer size: {len(replay_buffer.memory)}")

    step_num = wandb.config.step_num
    batch_size = wandb.config.batch_size
    log_interval = wandb.config.log_interval
    test_interval = wandb.config.test_interval

    for step in range(step_num):
        # Sample batch and train
        states, actions, rewards, next_states, terminals = replay_buffer.sample(
            batch_size
        )
        q_loss, v_loss, a_loss = model.step(
            states, actions, rewards, next_states, terminals
        )

        # Log training losses
        wandb.log(
            {
                "train/q_loss": q_loss,
                "train/v_loss": v_loss,
                "train/a_loss": a_loss,
                "step": step,
            }
        )
        if step % log_interval == 0:
            logger.info(
                f"Step: {step} Q_loss: {q_loss} V_loss: {v_loss} A_loss: {a_loss}"
            )

        # Periodic testing
        if step % test_interval == 0 or step == step_num - 1:
            test_loss = evaluate_model(
                model, test_data, batch_size, is_normalize, normalize_dic
            )
            wandb.log({"test/loss": test_loss, "step": step})
            logger.info(f"Step: {step} Test Loss: {test_loss}")

        if step % wandb.config.save_every == 0:
            model.save_jit(
                ROOT_DIR / "output" / "offline" / experiment_name / f"model_{step}"
            )

    # Save model
    model.save_jit(ROOT_DIR / "output" / "offline" / experiment_name / "model_sparse")
    wandb.finish()


def add_to_replay_buffer(replay_buffer, data, is_normalize):
    for row in data.itertuples():
        state, action, reward, next_state, done = (
            row.state if not is_normalize else row.normalize_state,
            row.action,
            row.reward if not is_normalize else row.normalize_reward,
            row.next_state if not is_normalize else row.normalize_nextstate,
            row.done,
        )
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


# def evaluate_model(model, test_data):
#     """
#     Evaluate the IQL model on the test dataset and return the average loss.
#     """
#     total_loss = 0.0
#     num_samples = len(test_data)
#     for row in test_data.itertuples():
#         state = np.array(row.state)
#         action = np.array([row.action])
#         pred_action = model.take_actions(
#             torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#         )
#         loss = ((pred_action.cpu().detach().numpy() - action) ** 2).mean()
#         total_loss += loss
#     return total_loss / num_samples


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


if __name__ == "__main__":
    run_iql()
