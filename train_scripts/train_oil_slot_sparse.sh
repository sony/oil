#!/bin/bash

cd "$(dirname "$0")/../"

python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 \
    --seed 0 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix _oracle_slot_sparse_dataset --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --data_folder_name online_rl_data_sparse