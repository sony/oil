import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import torch
import pandas as pd
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.bc.behavior_clone import BC
import logging
import ast

np.set_printoptions(suppress=True, precision=4)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_bc():
    """
    Run bc model training and evaluation.
    """
    train_model()
    # load_model()


def train_model():
    """
    train BC model
    """
    experiment_name = "test"
    train_data_path = "/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/traffic/offline_rl_data_final/period-7_26_offline_rl_data.parquet"
    training_data = pd.read_parquet(train_data_path)

    # 使用apply方法应用上述函数
    training_data["state"] = training_data["state"].apply(np.array)
    training_data["next_state"] = training_data["next_state"].apply(np.array)

    state_dim = 60
    normalize_indices = list(range(state_dim))
    is_normalize = True

    normalize_dic = normalize_state(training_data, state_dim, normalize_indices)
    normalize_reward(training_data, "reward_continuous")
    save_normalize_dict(normalize_dic, f"output/offline/{experiment_name}")

    model = BC(dim_obs=state_dim)
    replay_buffer = ReplayBuffer(device="cuda" if model.use_cuda else "cpu")
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))

    logger.info(f"Replay buffer size: {len(replay_buffer.memory)}")

    step_num = 20000
    batch_size = 100
    for i in range(step_num):
        states, actions, _, _, _ = replay_buffer.sample(batch_size)
        a_loss = model.step(states, actions)
        logger.info(f"Step: {i} Action loss: {np.mean(a_loss)}")

    # model.save_net("saved_model/BCtest")
    model.save_jit("saved_model/BCtest")
    test_trained_model(model, replay_buffer)


def load_model():
    """
    load model
    """
    model = BC(dim_obs=60)
    model.load_net("saved_model/BCtest")
    test_state = np.ones(16, dtype=np.float32)
    test_state_tensor = torch.tensor(test_state, dtype=torch.float)
    logger.info(f"Test action: {model.take_actions(test_state_tensor)}")


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = row.state if not is_normalize else row.normalize_state, row.action, row.reward if not is_normalize else row.normalize_reward, row.next_state if not is_normalize else row.normalize_nextstate, row.done
        # ! 去掉了所有的done==1的数据
        if done != 1:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.array(next_state),
                               np.array([done]))
        else:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state),
                               np.array([done]))


def test_trained_model(model, replay_buffer):
    states, actions, rewards, next_states, terminals = replay_buffer.sample(100)
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred_action:", tem)


if __name__ == "__main__":
    run_bc()