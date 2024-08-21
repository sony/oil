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
    parser.add_argument("--budget", nargs="+", type=int, default=[500])
    parser.add_argument("--target_cpa", nargs="+", type=int, default=[8])
    parser.add_argument("--category", nargs="+", type=int, default=[4])
    parser.add_argument(
        "--experiment",
        type=str,
        default="onlineLpTest",
    )
    parser.add_argument("--strategy_name", type=str, default="onlineLp")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run_test(
        data_path_or_dataloader=args.data_path,
        budget_list=args.budget,
        target_cpa_list=args.target_cpa,
        category_list=args.category,
        saved_model_name=args.experiment,
        strategy_name=args.strategy_name,
        device=args.device,
    )

"""Examples
# Best so far
python main/main_test.py --data_path ./data/traffic/period-12.csv --budget 500 --target_cpa 8 --category 0 \
    --experiment IQL/train_003/checkpoint_5000 --strategy_name iql --device cuda

# Also good
python main/main_test.py --data_path ./data/traffic/period-10.csv --budget 500 --target_cpa 8 --category 0 \
    --experiment IQL/train_002/checkpoint_5000 --strategy_name iql --device cuda

python main/main_test.py --data_path ./data/traffic/period-7.csv --budget 6000 --target_cpa 8 --category 0 \
--experiment IQL/train_004/checkpoint_18000 --strategy_name iql --device cpu

python main/main_test.py --data_path ./data/traffic/period-7.csv --budget 3000 --target_cpa 8 --category 0 \
    --experiment onlineLpTest --strategy_name onlineLp --device cuda
    
python main/main_test.py --data_path data/traffic/all_periods.parquet --budget 100 500 1_000 3_000 5_000 7_000 9_000 11_000 13_000\
    --target_cpa 2 4 6 8 10 12 --category 0 \
    --experiment onlineLpTest --strategy_name onlineLp --device cpu
    
python main/main_test.py --data_path data/traffic/all_periods.parquet --budget 100 500 1_000 3_000 5_000 7_000 9_000 11_000 13_000\
    --target_cpa 2 4 6 8 10 12 --category 0 \
    --experiment BC/train_004/checkpoint_100000 --strategy_name bc --device cpu

python main/main_test.py --data_path data/traffic/all_periods.parquet --budget 500 3000 7000 11000 \
    --target_cpa 4 8 12 --category 0 \
    --experiment onlineLpTest --strategy_name onlineLp --device cpu
    
python main/main_test.py --data_path data/traffic/all_periods.parquet --budget 500 3000 7000 11000 \
    --target_cpa 4 8 12 --category 0 \
    --experiment IQL/train_full_dataset_001/checkpoint_50000 --strategy_name iql --device cpu
"""
