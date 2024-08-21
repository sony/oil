import numpy as np
import torch
import pandas as pd
import logging
import ast
import json
import os
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.bc.behavior_clone import BC
from definitions import ROOT_DIR
from run.run_evaluate import run_test
from bidding_train_env.common.utils import (
    normalize_state,
    normalize_reward,
    save_normalize_dict,
)
from bidding_train_env.dataloader.test_dataloader import TestDataLoader

np.set_printoptions(suppress=True, precision=4)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

STATE_DIM = 29
ACTION_DIM = 1
IDX_NORM = [24, 25, 26, 27, 28] # [13, 14, 15] for 16, [16, 17, 18] for 19, [24, 25, 26, 27, 28] for 29
# train_data_path = ROOT_DIR / "data/traffic/custom_training_data/training_data_all-rlData.csv"
train_data_path = (
    ROOT_DIR / "data/traffic_top_regression/training_data_29/training_data_all-rlData.csv"
)
experiment_name = "BC/026_train_regression_29"
bc_params = {}
step_num = 50_000
batch_size = 100
actor_lr = 1e-4
actor_train_iter = 3
print_every = 300
eval_every = 1_000
data_path = ROOT_DIR / "data/traffic/all_periods.parquet"

eval_params = {
    "budget_list": [500, 3000, 7000, 11000],
    "target_cpa_list": [4, 8, 12],
    "category_list": [0],
}


def run_bc():
    """
    Run bc model training and evaluation.
    """
    train_model(
        train_data_path=train_data_path, step_num=step_num, batch_size=batch_size
)
    # load_model()


def train_model(train_data_path, step_num, batch_size):
    """
    train BC model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_data = pd.read_csv(train_data_path)

    out_path = ROOT_DIR / "output" / experiment_name

    # Save the bc parametes and a copy of the current script to the output path
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "bc_params.json", "w") as f:
        json.dump(bc_params, f, indent=4)
    with open(out_path / "run_bc.py", "w") as f:
        f.write(open("run/run_bc.py").read())

    def safe_literal_eval(val):
        if pd.isna(val):
            return val  # 如果是NaN，返回NaN
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val  # 如果解析出错，返回原值

    # 使用apply方法应用上述函数
    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)

    is_normalize = True

    normalize_dic = normalize_state(training_data, STATE_DIM, IDX_NORM)
    normalize_reward(training_data, "reward_continuous")
    # save_normalize_dict(normalize_dic, "saved_model/BCtest")

    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))

    logger.info(f"Replay buffer size: {len(replay_buffer.memory)}")

    model = BC(dim_obs=STATE_DIM, actor_lr=actor_lr, actor_train_iter=actor_train_iter)

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
        strategy_name="bc",
        data_path_or_dataloader=dataloader,
        **eval_params,
    )
    model.train()

    num_runs = step_num // eval_every
    for i in range(num_runs):
        model.train()
        for j in range(eval_every):
            states, actions, _, _, _ = replay_buffer.sample(batch_size)
            a_loss = model.step(states, actions)
            cum_loss = np.mean(a_loss)
            if j % print_every == 0:
                logger.info(
                    f"Step: {i * eval_every + j} Action loss: {np.mean(cum_loss):.4f}"
                )
                cum_loss = 0

        # Save model
        checkpoint_num = (i + 1) * eval_every
        logger.info(f"Saving model at step {checkpoint_num}")
        checkpoint_path = out_path / f"checkpoint_{checkpoint_num}"
        model.save_jit(checkpoint_path)
        save_normalize_dict(normalize_dic, checkpoint_path)

        # Evaluate checkpoint
        logger.info(f"Evaluating model at step {(i + 1) * eval_every}")
        model_name = os.path.join(experiment_name, f"checkpoint_{checkpoint_num}")

        model.eval()
        run_test(
            saved_model_name=model_name,
            strategy_name="bc",
            data_path_or_dataloader=dataloader,
            **eval_params,
        )
        model.train()

    # test_trained_model(model, replay_buffer)


def load_model():
    """
    load model
    """
    model = BC(dim_obs=STATE_DIM)
    model.load_net("saved_model/BCtest")
    test_state = np.ones(STATE_DIM, dtype=np.float32)
    test_state_tensor = torch.tensor(test_state, dtype=torch.float)
    logger.info(f"Test action: {model.take_actions(test_state_tensor)}")


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


def test_trained_model(model, replay_buffer):
    states, actions, rewards, next_states, terminals = replay_buffer.sample(100)
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred_action:", tem)


if __name__ == "__main__":
    run_bc()
