import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import os
import shutil
import argparse
import json
import torch.nn as nn
import wandb
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from metrics.custom_callbacks import TensorboardCallback
from train.trainer import SingleEnvTrainer
from envs.helpers import get_model_and_env_path

torch.manual_seed(0)


parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Seed for random number generator",
)
parser.add_argument(
    "--log_std_init",
    type=float,
    default=0.0,
    help="Initial log standard deviation",
)
parser.add_argument(
    "--env_name",
    type=str,
    default="BiddingEnv",
    help="Name of the environment",
)
parser.add_argument(
    "--load_path",
    type=str,
    help="Path to the experiment to load",
)
parser.add_argument(
    "--checkpoint_num",
    type=int,
    default=None,
    help="Number of the checkpoint to load",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device, cuda or cpu",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=100_000_000,
    help="Number of training steps",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="Batch size",
)
parser.add_argument(
    "--out_prefix",
    type=str,
    default="",
    help="Prefix to prepend to the training run name",
)
parser.add_argument(
    "--out_suffix",
    type=str,
    default="",
    help="Suffix to append to the training run name",
)
parser.add_argument(
    "--net_arch",
    type=int,
    nargs="*",
    default=[256, 256],
    help="Layer sizes for the policy and value networks",
)
parser.add_argument(
    "--project_name",
    type=str,
    default="alibaba",
    help="Project name for wandb",
)
parser.add_argument(
    "--save_every",
    type=int,
    default=250_000,
    help="Save a checkpoint every N number of steps",
)
parser.add_argument(
    "--budget_min",
    type=float,
    default=200,
    help="Minimum budget",
)
parser.add_argument(
    "--budget_max",
    type=float,
    default=12_000,
    help="Maximum budget",
)
parser.add_argument(
    "--target_cpa_min",
    type=float,
    default=6,
    help="Minimum target CPA",
)
parser.add_argument(
    "--target_cpa_max",
    type=float,
    default=12,
    help="Maximum target CPA",
)
parser.add_argument(
    "--new_action",
    action="store_true",
    help="Use the new action transformation",
)
parser.add_argument(
    "--multi_action",
    action="store_true",
    help="Use an action vec of 5 elements",
)
parser.add_argument(
    "--exp_action",
    action="store_true",
    help="Use the exponential action transformation",
)
parser.add_argument(
    "--dense_weight",
    type=float,
    default=1,
    help="Weight for dense reward",
)
parser.add_argument(
    "--sparse_weight",
    type=float,
    default=0,
    help="Weight for sparse reward",
)
parser.add_argument(
    "--obs_type",
    type=str,
    default="obs_16_keys",
    help="Type of observation",
)
parser.add_argument(
    "--act_type",
    type=str,
    default="act_1_key",
    help="Type of action",
)
parser.add_argument(
    "--sample_log_budget",
    action="store_true",
    help="Sample log budget",
)
parser.add_argument(
    "--simplified_bidding",
    action="store_true",
    help="Use simplified bidding",
)
parser.add_argument(
    "--stochastic_exposure",
    action="store_true",
    help="Stochastic exposure",
)
parser.add_argument(
    "--cost_noise",
    type=float,
    default=0.0,
    help="Cost noise",
)
parser.add_argument(
    "--bid_noise",
    type=float,
    default=0.0,
    help="Bid noise",
)

args = parser.parse_args()

run_name = f"{args.out_prefix}ppo_seed_{args.seed}{args.out_suffix}"
TENSORBOARD_LOG = os.path.join(ROOT_DIR, "output", "training", "ongoing", run_name)

# Reward structure and task parameters:
with open(ROOT_DIR / "data" / "env_configs" / f"{args.obs_type}.json", "r") as f:
    obs_keys = json.load(f)
with open(ROOT_DIR / "data" / "env_configs" / f"{args.act_type}.json", "r") as f:
    act_keys = json.load(f)

config_list = []
for period in range(7, 7 + args.num_envs):  # one period per env
    assert os.path.exists(
        ROOT_DIR / "data" / "online_rl_data" / f"period-{period}_bids.parquet"
    )
    assert os.path.exists(
        ROOT_DIR / "data" / "online_rl_data" / f"period-{period}_pvalues.parquet"
    )
    pvalues_df_path = (
        ROOT_DIR / "data" / "online_rl_data" / f"period-{period}_pvalues.parquet"
    )
    bids_df_path = (
        ROOT_DIR / "data" / "online_rl_data" / f"period-{period}_bids.parquet"
    )

    rwd_weights = {
        "dense": args.dense_weight,
        "sparse": args.sparse_weight,
    }

    config_list.append(
        {
            "env_name": args.env_name,
            "pvalues_df_path": pvalues_df_path,
            "bids_df_path": bids_df_path,
            "budget_range": (args.budget_min, args.budget_max),
            "target_cpa_range": (args.target_cpa_min, args.target_cpa_max),
            "rwd_weights": rwd_weights,
            "new_action": args.new_action,
            "multi_action": args.multi_action,
            "exp_action": args.exp_action,
            "obs_keys": obs_keys,
            "act_keys": act_keys,
            "sample_log_budget": args.sample_log_budget,
            "simplified_bidding": args.simplified_bidding,
            "stochastic_exposure": args.stochastic_exposure,
            "cost_noise": args.cost_noise,
            "competitor_bid_noise": args.bid_noise,
            "seed": args.seed,
        }
    )

model_config = dict(
    policy="MlpPolicy",
    device=args.device,
    batch_size=args.batch_size,
    n_steps=128,
    learning_rate=2e-05,
    ent_coef=3e-06,
    clip_range=0.3,
    gamma=0.99,
    gae_lambda=0.9,
    max_grad_norm=0.7,
    vf_coef=0.5,
    n_epochs=10,
    use_sde=False,
    policy_kwargs=dict(
        log_std_init=args.log_std_init,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=args.net_arch, vf=args.net_arch),
    ),
    seed=args.seed,
)


# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_config_list):
    def make_env(env_config):
        def _thunk():
            env = EnvironmentFactory.create(**env_config)
            env = Monitor(env, TENSORBOARD_LOG)
            return env

        return _thunk

    return SubprocVecEnv([make_env(c) for c in env_config_list])


if __name__ == "__main__":
    run = wandb.init(
        project=args.project_name,
        name=run_name,
        config=model_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    # ensure tensorboard log directory exists and copy this file and the args
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)
    with open(os.path.join(TENSORBOARD_LOG, "args.json"), "w") as file:
        json.dump(args.__dict__, file, indent=4, default=lambda _: "<not serializable>")
    with open(os.path.join(TENSORBOARD_LOG, "env_config.json"), "w") as file:
        json.dump(config_list[0], file, indent=4, default=lambda _: "<not serializable")

    checkpoint_callback = CheckpointCallback(
        save_freq=int(args.save_every / args.num_envs),
        save_path=TENSORBOARD_LOG,
        save_vecnormalize=True,
        verbose=1,
    )

    tensorboard_callback = TensorboardCallback(
        info_keywords=(
            "conversions",
            "cost",
            "cpa",
            "target_cpa",
            "budget",
            "avg_pvalues",
            "score_over_pvalue",
            "score_over_budget",
            "score_over_cpa",
            "cost_over_budget",
            "target_cpa_over_cpa",
            "score",
            "sparse",
            "dense",
            "action",
            "bid",
        ),
    )

    model_path, env_path = get_model_and_env_path(
        TENSORBOARD_LOG, args.load_path, args.checkpoint_num
    )

    # Create and wrap the training and evaluations environments
    envs = make_parallel_envs(config_list)
    if env_path is not None:
        envs = VecNormalize.load(env_path, envs)
    else:
        envs = VecNormalize(envs)

    # Define trainer
    trainer = SingleEnvTrainer(
        algo="ppo",
        envs=envs,
        load_model_path=model_path,
        log_dir=TENSORBOARD_LOG,
        model_config=model_config,
        callbacks=[tensorboard_callback, checkpoint_callback],
        timesteps=args.num_steps,
    )

    # Train agent
    trainer.train()
    trainer.save()

"""
python online/main_train.py --num_envs 1 --batch_size 8 --num_steps 1_000_000 --out_suffix _test_004

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 10_000_000 --out_suffix _test_005 \
    --budget_min 6000 --budget_max 6000 --target_cpa_min 8 --target_cpa_max 8

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 10_000_000 --out_prefix 006_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 8 --target_cpa_max 8

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 10_000_000 --out_prefix 007_ \
    --budget_min 200 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 10_000_000 --out_prefix 010_ \
    --budget_min 6000 --budget_max 6000 --target_cpa_min 8 --target_cpa_max 8 \
        --load_path output/training/ongoing/008_ppo_seed_0 --checkpoint_num 5250000
        
python online/main_train.py --num_envs 1 --batch_size 256 --num_steps 10_000_000 --out_prefix 012_ \
    --budget_min 6000 --budget_max 6000 --target_cpa_min 8 --target_cpa_max 8 \
        --load_path output/training/ongoing/008_ppo_seed_0 --checkpoint_num 5250000

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 50_000_000 --out_prefix 013_ \
    --budget_min 200 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --load_path output/training/ongoing/008_ppo_seed_0  --out_suffix=_old_action
        
python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 50_000_000 --out_prefix 014_ \
    --budget_min 200 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --out_suffix=_new_action_from_scratch
        
python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 50_000_000 --out_prefix 016_ \
    --budget_min 200 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --out_suffix=_new_action_test
        
python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 50_000_000 --out_prefix 017_ \
    --budget_min 200 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --out_suffix=_new_action_test
        
python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 50_000_000 --out_prefix 018_ \
    --budget_min 200 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --multi_action --out_suffix=_new_action_vec

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 019_ \
    --budget_min 50 --budget_max 15000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --out_suffix=_resume_016_sparse_wider_ranges \
            --dense_weight 0 --sparse_weight 1 \
                --load_path output/training/ongoing/016_ppo_seed_0_new_action_test --checkpoint_num 10250000
                
python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 021_ \
    --budget_min 50 --budget_max 50000 --target_cpa_min 1 --target_cpa_max 20 \
        --new_action --out_suffix=_dense_very_wide_ranges \
            --dense_weight 1 --sparse_weight 0

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 022_ \
    --budget_min 50 --budget_max 50000 --target_cpa_min 1 --target_cpa_max 20 \
        --new_action --out_suffix=_sparse_very_wide_ranges \
            --dense_weight 0 --sparse_weight 1
            

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 023_ \
    --budget_min 50 --budget_max 50000 --target_cpa_min 1 --target_cpa_max 20 \
        --new_action --out_suffix=_dense_very_wide_ranges_60_obs \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_60_keys

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 023_ \
    --budget_min 50 --budget_max 50000 --target_cpa_min 1 --target_cpa_max 20 \
        --new_action --out_suffix=_dense_very_wide_ranges_60_obs_multi_act \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_60_keys --multi_action

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 024_ \
    --budget_min 50 --budget_max 50000 --target_cpa_min 1 --target_cpa_max 20 \
        --new_action --out_suffix=_dense_very_wide_ranges_60_obs \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_60_keys
            
python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 025_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --multi_action --out_suffix=_dense_base_ranges_60_obs \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_60_keys

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 026_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_dense_base_ranges_60_obs_exp_single_action \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_60_keys
            
python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 027_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_dense_base_ranges_19_obs_exp_single_action_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_19_keys --simplified_bidding

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 028_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_dense_base_ranges_16_obs_exp_single_action_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_16_keys --simplified_bidding
            
python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 029_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_dense_base_ranges_29_obs_exp_single_action_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys --simplified_bidding
            
python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 030_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --multi_action --out_suffix=_dense_base_ranges_16_obs_exp_multi_action_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_16_keys --simplified_bidding

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 031_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --multi_action --out_suffix=_dense_base_ranges_19_obs_exp_multi_action_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_19_keys --simplified_bidding
            
python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 032_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --multi_action --out_suffix=_dense_base_ranges_29_obs_exp_multi_action_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys --simplified_bidding
            
python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 033_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --multi_action --out_suffix=_dense_base_ranges_29_obs_exp_multi_action \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys
            
python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 034_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_dense_base_ranges_29_obs_exp_single_action \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 035_ \
    --budget_min 100 --budget_max 15000 --target_cpa_min 4 --target_cpa_max 16 \
        --new_action --exp_action --out_suffix=_dense_larger_ranges_29_obs_exp_single_action_simplified_resume_029 \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys --simplified_bidding \
                --load_path output/training/ongoing/029_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_simplified --checkpoint_num 10750000

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 036_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_dense_base_ranges_29_obs_exp_single_action_resume_029 \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys \
                --load_path output/training/ongoing/029_ppo_seed_0_dense_base_ranges_29_obs_exp_single_action_simplified --checkpoint_num 10750000
                
python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 037_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_dense_base_ranges_29_obs_exp_3_actions \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys --act_type act_3_keys 

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 038_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_dense_base_ranges_29_obs_exp_single_action_noisy_env \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys \
                --stochastic_exposure --cost_noise 0.01 --bid_noise 0.01

python online/main_train.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 039_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_dense_base_ranges_29_obs_exp_single_action_stoch_exposure \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys --stochastic_exposure
            
            
python online/main_train.py --num_envs 1 --batch_size 16 --num_steps 20_000_000 --out_prefix 000_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_test_seed_2 \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys

"""
