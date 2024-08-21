import os
import numpy as np
import logging
import sys
import pandas as pd
import ast
import torch
import pathlib
import json
from bidding_train_env.common.utils import (
    normalize_state,
    normalize_reward,
    save_normalize_dict,
)
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.iql.iql import IQL
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from run.run_evaluate import run_test
from definitions import ROOT_DIR

np.set_printoptions(suppress=True, precision=4)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

STATE_DIM = 16
ACTION_DIM = 1
IDX_NORM = [13, 14, 15]  # list(range(STATE_DIM))  # [13, 14, 15]
# train_data_path = ROOT_DIR / "data/traffic/custom_training_data/training_data_all-rlData.csv"
train_data_path = (
    ROOT_DIR
    / "data/traffic_top_regression/training_data_16/training_data_all-rlData.csv"
)
experiment_name = "IQL/train_regression_16_003"
iql_params = {
    "gamma": 0.99,
    "tau": 0.01,
    "V_lr": 0.0001,
    "critic_lr": 0.0001,
    "actor_lr": 1e-4,
    "network_random_seed": 1,
    "expectile": 0.5,
    "temperature": 1.0,
}
step_num = 200_000
batch_size = 100
print_every = 300
eval_every = 1_000
data_path = ROOT_DIR / "data/traffic/all_periods.parquet"

eval_params = {
    "budget_list": [500, 3000, 7000, 11000],
    "target_cpa_list": [4, 8, 12],
    "category_list": [0],
}


def train_iql_model():
    """
    Train the IQL model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_data = pd.read_csv(train_data_path)

    out_path = ROOT_DIR / "output" / experiment_name

    # Save the iql parametes and a copy of the current script to the output path
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "iql_params.json", "w") as f:
        json.dump(iql_params, f, indent=4)
    with open(out_path / "run_iql.py", "w") as f:
        f.write(open("run/run_iql.py").read())

    def safe_literal_eval(val):
        if pd.isna(val):
            return val
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val

    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
    is_normalize = True

    if is_normalize:
        normalize_dic = normalize_state(
            training_data, STATE_DIM, normalize_indices=IDX_NORM
        )
        # select use continuous reward
        training_data["reward"] = normalize_reward(training_data, "reward_continuous")
        # select use sparse reward
        # training_data['reward'] = normalize_reward(training_data, "reward")
        # save_normalize_dict(normalize_dic, out_path)

    # Build replay buffer
    replay_buffer = ReplayBuffer(device=device)
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))

    # Train model
    model = IQL(
        dim_obs=STATE_DIM,
        dim_actions=ACTION_DIM,
        device=device,
        **iql_params,
    )

    # Load the test data once and keep them in memory as it takes some time
    logger.info(f"Loading data from {data_path}")
    dataloader = TestDataLoader(file_path=data_path)
    logger.info(f"Data loaded successfully")

    # Save untrained model
    checkpoint_num = 0
    logger.info(f"Saving model at step {checkpoint_num}")
    checkpoint_path = out_path / f"checkpoint_{checkpoint_num}"
    model.save_jit(checkpoint_path)
    save_normalize_dict(normalize_dic, checkpoint_path)

    model.eval()
    model_name = os.path.join(experiment_name, f"checkpoint_{checkpoint_num}")
    logger.info(f"Evaluating model at step {checkpoint_num}")
    run_test(
        saved_model_name=model_name,
        strategy_name="iql",
        data_path_or_dataloader=dataloader,
        **eval_params,
    )
    model.train()

    num_runs = step_num // eval_every
    for i in range(num_runs):
        model.train()
        train_model_steps(
            model,
            replay_buffer,
            step_num=eval_every,
            batch_size=batch_size,
            print_every=print_every,
        )

        # Save model
        checkpoint_num = (i + 1) * eval_every
        logger.info(f"Saving model at step {checkpoint_num}")
        checkpoint_path = out_path / f"checkpoint_{checkpoint_num}"
        model.save_jit(checkpoint_path)
        save_normalize_dict(normalize_dic, checkpoint_path)

        # Evaluate checkpoint
        logger.info(f"Evaluating model at step {checkpoint_num}")
        model_name = os.path.join(experiment_name, f"checkpoint_{checkpoint_num}")

        model.eval()
        run_test(
            saved_model_name=model_name,
            strategy_name="iql",
            data_path_or_dataloader=dataloader,
            **eval_params,
        )
        model.train()

    # Test trained model
    test_trained_model(model, replay_buffer)


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = (
            row.state if not is_normalize else row.normalize_state,
            row.action,
            row.reward if not is_normalize else row.normalize_reward,
            row.next_state if not is_normalize else row.normalize_nextstate,
            row.done,
        )
        # ! 去掉了所有的done==1的数据
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


def train_model_steps(
    model, replay_buffer, step_num=200_000, batch_size=100, print_every=100
):
    for i in range(step_num):
        cum_q_loss = 0
        cum_v_loss = 0
        cum_a_loss = 0
        states, actions, rewards, next_states, terminals = replay_buffer.sample(
            batch_size
        )
        q_loss, v_loss, a_loss = model.step(
            states, actions, rewards, next_states, terminals
        )
        cum_q_loss += q_loss
        cum_v_loss += v_loss
        cum_a_loss += a_loss
        if i % print_every == 0:
            logger.info(
                f"Step: {i} Q_loss: {cum_q_loss / 100} V_loss: {cum_v_loss / 100} A_loss: {cum_a_loss / 100}"
            )
            cum_q_loss = 0
            cum_v_loss = 0
            cum_a_loss = 0


def test_trained_model(model, replay_buffer):
    states, actions, rewards, next_states, terminals = replay_buffer.sample(100)
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred action:", tem)


def run_iql():
    print(sys.path)
    """
    Run IQL model training and evaluation.
    """
    train_iql_model()


if __name__ == "__main__":
    run_iql()
