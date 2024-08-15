import numpy as np
import torch
import os
import sys
import argparse
import pathlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run.run_evaluate import run_test

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=pathlib.Path, default="./data/traffic/period-12.csv"
    )
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--target_cpa", type=int, default=8)
    parser.add_argument("--category", type=int, default=4)
    parser.add_argument(
        "--experiment_path",
        type=pathlib.Path,
        default="./saved_model/onlineLpTest",
    )
    parser.add_argument("--strategy_name", type=str, default="default")
    args = parser.parse_args()
    run_test(
        data_path=args.data_path,
        budget=args.budget,
        target_cpa=args.target_cpa,
        category=args.category,
        experiment_path=args.experiment_path,
        strategy_name=args.strategy_name,
    )

"""Examples
# Best so far
python main/main_test.py --data_path ./data/traffic/period-12.csv --budget 500 --target_cpa 8 --category 0 \
    --experiment_path saved_model/IQL/train_003/checkpoint_5000 --strategy_name iql

# Also good
python main/main_test.py --data_path ./data/traffic/period-10.csv --budget 500 --target_cpa 8 --category 0 \
    --experiment_path saved_model/IQL/train_002/checkpoint_5000 --strategy_name iql

python main/main_test.py --data_path ./data/traffic/period-7.csv --budget 500 --target_cpa 12 --category 0 \
--experiment_path saved_model/IQL/train_003/checkpoint_5000 --strategy_name iql

python main/main_test.py --data_path ./data/traffic/period-7.csv --budget 500 --target_cpa 12 --category 0 \
    --experiment_path saved_model/onlineLpTest --strategy_name onlineLp

"""
