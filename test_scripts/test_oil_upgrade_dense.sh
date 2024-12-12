#!/bin/bash

cd "$(dirname "$0")/../"

python online/main_eval.py --experiment_path=pretrained/oil_seed_0_oracle_upgrade_dense_dataset \
    --num_episodes=1000 --deterministic --checkpoint 10000000 --save_dict\
        --eval_config_path=data/env_configs/eval_config_dense.json
