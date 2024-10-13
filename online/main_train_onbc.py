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
from online.env_wrappers.subproc_vec_env import BatchSubprocVecEnv
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from metrics.custom_callbacks import TensorboardCallback
from train.trainer import SingleEnvTrainer
from envs.helpers import get_model_and_env_path
from online.policies.actor import ActorPolicy, TransformerActorPolicy
from algos.buffers import OracleRolloutBuffer, OracleEpisodeRolloutBuffer
from online.env_wrappers.vec_normalize import BatchVecNormalize

torch.manual_seed(0)


parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument(
    "--algo",
    type=str,
    default="onbc",
    help="Algorithm to use",
)
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
    "--use_transformer",
    action="store_true",
    help="Use transformer policy",
)
parser.add_argument(
    "--embed_size",
    type=int,
    default=64,
    help="Embedding size for transformer",
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=4,
    help="Number of heads for transformer",
)
parser.add_argument(
    "--num_layers",
    type=int,
    default=2,
    help="Number of layers for mlp or transformer",
)
parser.add_argument(
    "--dim_feedforward",
    type=int,
    default=256,
    help="Feedforward dimension for mlp or transformer",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.1,
    help="Dropout for transformer",
)
parser.add_argument(
    "--layer_norm_eps",
    type=float,
    default=1e-5,
    help="Layer norm epsilon for transformer",
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
    "--simplified_oracle",
    action="store_true",
    help="Use simplified oracle",
)
parser.add_argument(
    "--stochastic_exposure",
    action="store_true",
    help="Stochastic exposure",
)
parser.add_argument(
    "--auction_noise_min",
    type=float,
    default=0.0,
    help="Cost and bid noise",
)
parser.add_argument(
    "--auction_noise_max",
    type=float,
    default=0.0,
    help="Cost and bid noise",
)
parser.add_argument(
    "--pvalues_rescale_min",
    type=float,
    default=1.0,
    help="Minimum pvalues rescale",
)
parser.add_argument(
    "--pvalues_rescale_max",
    type=float,
    default=1.0,
    help="Maximum pvalues rescale",
)
parser.add_argument(
    "--exposure_prob_min",
    type=float,
    default=1.0,
    help="Minimum exposure probability",
)
parser.add_argument(
    "--exposure_prob_max",
    type=float,
    default=1.0,
    help="Maximum exposure probability",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=2e-05,
    help="Learning rate",
)
parser.add_argument(
    "--n_rollout_steps",
    type=int,
    default=128,
    help="Number of steps to run for each environment",
)
parser.add_argument(
    "--episode_len",
    type=int,
    default=48,
    help="Length of each episode",
)
parser.add_argument(
    "--exclude_self_bids",
    action="store_true",
    help="Exclude self bids from the auction",
)
parser.add_argument(
    "--flex_oracle",
    action="store_true",
    help="Use flexible oracle",
)
parser.add_argument(
    "--two_slopes_action",
    action="store_true",
    help="Use two slopes for the action transformation",
)
parser.add_argument(
    "--detailed_bid",
    action="store_true",
    help="Use detailed prediction",
)
parser.add_argument(
    "--batch_state",
    action="store_true",
    help="Use batched states",
)
parser.add_argument(
    "--batch_state_subsample",
    type=int,
    default=1000,
    help="Subsample for batched states",
)
args = parser.parse_args()

run_name = f"{args.out_prefix}onbc_seed_{args.seed}{args.out_suffix}"
TENSORBOARD_LOG = os.path.join(ROOT_DIR, "output", "training", "ongoing", run_name)

# Reward structure and task parameters:
with open(ROOT_DIR / "env_configs" / f"{args.obs_type}.json", "r") as f:
    obs_keys = json.load(f)
with open(ROOT_DIR / "env_configs" / f"{args.act_type}.json", "r") as f:
    act_keys = json.load(f)

config_list = []
for period in range(7, 7 + args.num_envs):  # one period per env
    assert os.path.exists(
        ROOT_DIR
        / "data"
        / "online_rl_data_final_with_ad_idx"
        / f"period-{period}_bids.parquet"
    )
    assert os.path.exists(
        ROOT_DIR
        / "data"
        / "online_rl_data_final_with_ad_idx"
        / f"period-{period}_pvalues.parquet"
    )
    pvalues_df_path = (
        ROOT_DIR
        / "data"
        / "online_rl_data_final_with_ad_idx"
        / f"period-{period}_pvalues.parquet"
        # / f"pvalues_periods_7_26.parquet"
    )
    bids_df_path = (
        ROOT_DIR
        / "data"
        / "online_rl_data_final_with_ad_idx"
        / f"period-{period}_bids.parquet"
        # / f"bids_periods_7_26.parquet"
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
            "auction_noise": (
                args.auction_noise_min,
                args.auction_noise_max,
            ),
            "pvalues_rescale_range": (
                args.pvalues_rescale_min,
                args.pvalues_rescale_max,
            ),
            "simplified_exposure_prob_range": (
                args.exposure_prob_min,
                args.exposure_prob_max,
            ),
            "simplified_oracle": args.simplified_oracle,
            "exclude_self_bids": args.exclude_self_bids,
            "flex_oracle": args.flex_oracle or args.detailed_bid,
            "two_slopes_action": args.two_slopes_action,
            "detailed_bid": args.detailed_bid,
            "batch_state": args.batch_state,
            "seed": args.seed,
        }
    )

if args.use_transformer:
    policy = TransformerActorPolicy
    net_arch = {
        "embed_size": args.embed_size,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "layer_norm_eps": args.layer_norm_eps,
    }
    # n_steps in transformer is actually number of episodes
    n_steps = args.n_rollout_steps // args.episode_len
    rollout_buffer_class = OracleEpisodeRolloutBuffer
    rollout_buffer_kwargs = {"ep_len": args.episode_len}
else:
    policy = ActorPolicy
    net_arch = [args.dim_feedforward for _ in range(args.num_layers)]
    n_steps = args.n_rollout_steps
    rollout_buffer_class = OracleRolloutBuffer
    rollout_buffer_kwargs = {
        "batch_state_subsample": (
            args.batch_state_subsample if args.batch_state else None
        ),
    }

model_config = dict(
    policy=policy,
    device=args.device,
    batch_size=args.batch_size,
    n_steps=n_steps,
    learning_rate=args.learning_rate,
    ent_coef=3e-06,
    max_grad_norm=0.7,
    n_epochs=10,
    use_sde=False,
    policy_kwargs=dict(
        log_std_init=args.log_std_init,
        activation_fn=nn.ReLU,
        net_arch=net_arch,
    ),
    rollout_buffer_class=rollout_buffer_class,
    rollout_buffer_kwargs=rollout_buffer_kwargs,
    seed=args.seed,
)


# Maybe hacky, but it avoids the deprecation warning
class OracleMonitor(Monitor):
    def get_oracle_action(self):
        return self.unwrapped.get_oracle_action()


# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_config_list):
    def make_env(env_config):
        def _thunk():
            env = EnvironmentFactory.create(**env_config)
            env = OracleMonitor(env, TENSORBOARD_LOG)
            return env

        return _thunk

    return BatchSubprocVecEnv([make_env(c) for c in env_config_list])


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
        envs = BatchVecNormalize.load(env_path, envs)
    else:
        envs = BatchVecNormalize(envs)

    # Define trainer
    trainer = SingleEnvTrainer(
        algo=args.algo,
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
python online/main_train_onbc.py --num_envs 20 --batch_size 256 --num_steps 20_000_000 --out_prefix 001_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys \
                --simplified_bidding --learning_rate 1e-3 --save_every 250000
                
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 002_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys \
                --simplified_bidding --learning_rate 1e-3 --save_every 50000
                
python online/main_train_onbc.py --algo onbc_transformer --num_envs 2 --batch_size=2 --num_steps=10_000_000 \
    --n_rollout_steps 192 --budget_min=400 --budget_max=12000 --target_cpa_min=6 --target_cpa_max=12 \
        --new_action --exp_action --out_prefix=003_ --out_suffix=_transformer \
            --obs_type obs_19_keys_transformer --use_transformer --embed_size 64 --num_heads 4 --num_layers 4 \
                --dim_feedforward 256 --dropout 0.1 --layer_norm_eps 1e-5 --learning_rate 1e-3 --save_every 50000 \
                    --simplified_bidding


python online/main_train_onbc.py --algo onbc_transformer --num_envs 20 --batch_size=20 --num_steps=10_000_000 \
    --n_rollout_steps 192 --budget_min=400 --budget_max=12000 --target_cpa_min=6 --target_cpa_max=12 \
        --new_action --exp_action --out_prefix=004_ --out_suffix=_transformer \
            --obs_type obs_19_keys_transformer --use_transformer --embed_size 64 --num_heads 4 --num_layers 4 \
                --dim_feedforward 256 --dropout 0.1 --layer_norm_eps 1e-5 --learning_rate 1e-4 --save_every 50000 \
                    --simplified_bidding

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 005_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified_resume_002 \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys \
                --simplified_bidding --learning_rate 1e-5 --save_every 50000 \
                    --load_path output/training/ongoing/002_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
                        --checkpoint_num 3150000
                        
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 006_ \
    --budget_min 1e-6 --budget_max 24000 --target_cpa_min 1e-6 --target_cpa_max 24 \
        --new_action --exp_action --out_suffix=_wide_ranges_simplified_resume_002 \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys \
                --simplified_bidding --learning_rate 1e-3 --save_every 50000 \
                    --load_path output/training/ongoing/002_onbc_seed_0_dense_base_ranges_29_obs_exp_single_action_full_bc_simplified \
                        --checkpoint_num 3150000
                        
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 007_ \
    --budget_min 1e-6 --budget_max 24000 --target_cpa_min 1e-6 --target_cpa_max 24 \
        --new_action --exp_action --out_suffix=_wide_ranges_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys \
                --simplified_bidding --learning_rate 1e-3 --save_every 50000

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 008_ \
    --budget_min 1e-6 --budget_max 24000 --target_cpa_min 1e-6 --target_cpa_max 24 \
        --new_action --exp_action --out_suffix=_wide_ranges_simplified_3_layers --num_layers 3\
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys \
                --simplified_bidding --learning_rate 1e-3 --save_every 50000

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 009_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_small_pvals_auction_noise_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys --auction_noise 0.1 --pvalues_rescale_min 0.01 --pvalues_rescale_max 0.2 \
                --simplified_bidding --learning_rate 1e-3 --save_every 50000
                
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 010_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_all_pvals_auction_noise_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys --auction_noise 0.1 --pvalues_rescale_min 0.01 --pvalues_rescale_max 1 \
                --simplified_bidding --learning_rate 1e-3 --save_every 50000

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 011_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_small_pvals_auction_noise_stoch_exposure_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys --auction_noise 0.1 --pvalues_rescale_min 0.01 --pvalues_rescale_max 0.2 \
                --exposure_prob_min 0.5 --exposure_prob_max 1 --simplified_bidding --learning_rate 1e-3 --save_every 50000
                
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 012_ \
    --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
        --new_action --exp_action --out_suffix=_all_pvals_auction_noise_stoch_exposure_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys --auction_noise 0.1 --pvalues_rescale_min 0.01 --pvalues_rescale_max 1 \
                --exposure_prob_min 0.5 --exposure_prob_max 1 --simplified_bidding --learning_rate 1e-3 --save_every 50000
                    
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 013_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_simplified \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys \
                --simplified_bidding --learning_rate 1e-3 --save_every 50000
                
                
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 015_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_simplified_resume_013 \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys \
                --simplified_bidding --learning_rate 1e-3 --save_every 10000 \
                    --load_path output/training/ongoing/013_onbc_seed_0_new_data_simplified --checkpoint 3650000
                    
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 016_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_simplified_small_lr_resume_013 \
            --dense_weight 1 --sparse_weight 0 --obs_type obs_29_keys \
                --simplified_bidding --learning_rate 1e-5 --save_every 10000 \
                    --load_path output/training/ongoing/013_onbc_seed_0_new_data_simplified --checkpoint 3650000
                
python online/main_train_onbc.py --algo onbc_transformer --num_envs 20 --batch_size=20 --num_steps=10_000_000 \
    --n_rollout_steps 192 --budget_min=1000 --budget_max=6000 --target_cpa_min=50 --target_cpa_max=150 \
        --new_action --exp_action --out_prefix=014_ --out_suffix=_transformer_new_data \
            --obs_type obs_19_keys_transformer --use_transformer --embed_size 64 --num_heads 4 --num_layers 4 \
                --dim_feedforward 256 --dropout 0.1 --layer_norm_eps 1e-5 --learning_rate 1e-4 --save_every 50000 \
                    --simplified_bidding
                    
python online/main_train_onbc.py --algo onbc --num_envs 20 --batch_size=512 --num_steps=10_000_000 \
    --budget_min=1000 --budget_max=6000 --target_cpa_min=50 --target_cpa_max=150 \
        --new_action --exp_action --out_prefix=017_ --out_suffix=_stoch_exposure_simplified_new_data \
            --obs_type obs_29_keys --exposure_prob_min 0.5 --exposure_prob_max 1 \
                --learning_rate 1e-3 --save_every 50000 --simplified_bidding
                
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 019_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic_auction_simplified_oracle \
            --obs_type obs_29_keys --learning_rate 1e-3 --save_every 50000 --simplified_oracle
            
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 018_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic \
            --obs_type obs_29_keys --learning_rate 1e-3 --save_every 50000
            
python online/main_train_onbc.py --algo onbc_transformer --num_envs 20 --batch_size=20 --num_steps=10_000_000 \
    --n_rollout_steps 192 --budget_min=1000 --budget_max=6000 --target_cpa_min=50 --target_cpa_max=150 \
        --new_action --exp_action --out_prefix=020_ --out_suffix=_transformer_new_data_realistic \
            --obs_type obs_19_keys_transformer --use_transformer --embed_size 64 --num_heads 4 --num_layers 4 \
                --dim_feedforward 256 --dropout 0.1 --layer_norm_eps 1e-5 --learning_rate 1e-4 --save_every 50000
                
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 021_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic_resume_018 \
            --obs_type obs_29_keys --learning_rate 1e-5 --save_every 10000 \
                --load_path output/training/ongoing/018_onbc_seed_0_new_data_realistic --checkpoint_num 2000000
                
python online/main_train_onbc.py --algo onbc_transformer --num_envs 20 --batch_size=20 --num_steps=10_000_000 \
    --n_rollout_steps 192 --budget_min=1000 --budget_max=6000 --target_cpa_min=50 --target_cpa_max=150 \
        --new_action --exp_action --out_prefix=022_ --out_suffix=_transformer_new_data_realistic_resume_020 \
            --obs_type obs_19_keys_transformer --use_transformer --embed_size 64 --num_heads 4 --num_layers 4 \
                --dim_feedforward 256 --dropout 0.1 --layer_norm_eps 1e-5 --learning_rate 1e-5 --save_every 10000 \
                    --load_path output/training/ongoing/020_onbc_seed_0_transformer_new_data_realistic --checkpoint_num 4000000
                    
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 023_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic \
            --obs_type obs_60_keys --learning_rate 1e-3 --save_every 50000 --num_layers 3
            
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 026_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic_60_obs_resume_023 \
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 10000 --num_layers 3 \
                --load_path output/training/ongoing/023_onbc_seed_0_new_data_realistic --checkpoint_num 3100000
                
python online/main_train_onbc.py --algo onbc_transformer --num_envs 20 --batch_size=20 --num_steps=10_000_000 \
    --n_rollout_steps 192 --budget_min=1000 --budget_max=6000 --target_cpa_min=50 --target_cpa_max=150 \
        --new_action --exp_action --out_prefix=029_ --out_suffix=_transformer_new_data_realistic \
            --obs_type obs_35_keys_transformer --use_transformer --embed_size 64 --num_heads 4 --num_layers 4 \
                --dim_feedforward 256 --dropout 0.1 --layer_norm_eps 1e-5 --learning_rate 1e-3 --save_every 50000

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 031_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic_no_self_id \
            --obs_type obs_60_keys --learning_rate 1e-3 --save_every 50000 --num_layers 3

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 032_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic_no_self_id_resume_031 \
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 10000 --num_layers 3 \
                --load_path output/training/ongoing/031_onbc_seed_0_new_data_realistic_no_self_id --checkpoint_num 2300000
                
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 033_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --new_action --exp_action --out_suffix=_new_data_realistic_no_self_id_fix_cpa_resume_031 \
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 10000 --num_layers 3 \
                --load_path output/training/ongoing/031_onbc_seed_0_new_data_realistic_no_self_id --checkpoint_num 2300000

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 034_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --new_action --exp_action --out_suffix=_new_data_realistic_no_self_id_fix_cpa_resume_031 \
            --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 --num_layers 3
            
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 035_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --new_action --exp_action --out_suffix=_no_self_id_flex_two_slopes_oracle \
            --flex_oracle --two_slopes_action --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 --num_layers 3
            
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 036_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150\
        --new_action --exp_action --out_suffix=_flex_two_slopes_oracle \
            --flex_oracle --two_slopes_action --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 --num_layers 3

python online/main_train_onbc.py --algo onbc_transformer --num_envs 20 --batch_size=20 --num_steps=10_000_000 \
    --n_rollout_steps 192 --budget_min=1000 --budget_max=6000 --target_cpa_min=50 --target_cpa_max=150 \
        --new_action --exp_action --out_prefix=044_ --out_suffix=_transformer_35_obs_two_slopes \
            --obs_type obs_35_keys_transformer --use_transformer --embed_size 64 --num_heads 4 --num_layers 4 \
                --dim_feedforward 256 --dropout 0.1 --layer_norm_eps 1e-5 --learning_rate 1e-3 --save_every 20000 \
                    --flex_oracle --two_slopes_action
            
python online/main_train_onbc.py --num_envs 20 --num_steps 1_000_000_000 --out_prefix 046_ \
    --n_rollout_steps 16192 --batch_size 2048 --out_suffix=_detailed_bid \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150\
        --new_action --exp_action \
            --detailed_bid --obs_type obs_60_keys --learning_rate 1e-3 --save_every 500000 --num_layers 3
            
python online/main_train_onbc.py --num_envs 1 --num_steps 1_000_000_000 --out_prefix 051_ \
    --n_rollout_steps 128 --batch_size 32 --detailed_bid --batch_state \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150\
        --new_action --exp_action --out_suffix=_detailed_bid_batch_state \
            --obs_type obs_60_keys --learning_rate 1e-3 --save_every 500000 --num_layers 3

python online/main_train_onbc.py --num_envs 20 --num_steps 1_000_000_000 --out_prefix 055_ \
    --n_rollout_steps 128 --batch_size 256 --detailed_bid --batch_state \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150\
        --new_action --exp_action --out_suffix=_detailed_bid_batch_state_20_envs_1000_subs \
            --obs_type obs_60_keys --learning_rate 1e-3 --save_every 50000 --num_layers 3
            
            
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 037_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --new_action --exp_action --out_suffix=_no_self_id_flex_two_slopes_oracle_145_keys \
            --flex_oracle --two_slopes_action --obs_type obs_145_keys --learning_rate 1e-3 --save_every 10000 --num_layers 3
            
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 038_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150\
        --new_action --exp_action --out_suffix=_flex_two_slopes_oracle_145_keys \
            --flex_oracle --two_slopes_action --obs_type obs_145_keys --learning_rate 1e-3 --save_every 10000 --num_layers 3
            
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 039_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150\
        --new_action --exp_action --out_suffix=_flex_two_slopes_oracle_145_keys_4_layers \
            --flex_oracle --two_slopes_action --obs_type obs_145_keys --learning_rate 1e-3 --save_every 10000 --num_layers 4

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 040_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic_60_obs_seed_1 --seed 1\
            --obs_type obs_60_keys --learning_rate 1e-3 --save_every 20000 --num_layers 3
            
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 040_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic_60_obs_seed_1 --seed 1\
            --obs_type obs_60_keys --learning_rate 1e-3 --save_every 20000 --num_layers 3

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 041_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic_145_obs_seed_1 --seed 1\
            --obs_type obs_145_keys --learning_rate 1e-3 --save_every 20000 --num_layers 3
            
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 043_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic_145_obs_4_layers_seed_1 --seed 1\
            --obs_type obs_145_keys --learning_rate 1e-3 --save_every 20000 --num_layers 4
            
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 046_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic_60_obs_resume_040 --seed 1\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 10000 --num_layers 3 \
                --load_path output/training/ongoing/040_onbc_seed_1_new_data_realistic_60_obs_seed_1 --checkpoint_num 1640000

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 047_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150\
        --new_action --exp_action --out_suffix=_flex_two_slopes_oracle_60 \
            --flex_oracle --two_slopes_action --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 --num_layers 3

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 048_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150\
        --new_action --exp_action --out_suffix=_flex_two_slopes_oracle_60_resume_047 \
            --flex_oracle --two_slopes_action --obs_type obs_60_keys --learning_rate 1e-5 --save_every 10000 --num_layers 3 \
                --load_path output/training/ongoing/047_onbc_seed_0_flex_two_slopes_oracle_60 --checkpoint_num 6000000

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 050_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic_60_obs_fix_oracle --seed 0\
            --obs_type obs_60_keys --learning_rate 1e-3 --save_every 20000 --num_layers 3
            
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 051_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150\
        --new_action --exp_action --out_suffix=_flex_two_slopes_oracle_60_obs_fix_oracle \
            --flex_oracle --two_slopes_action --obs_type obs_60_keys --learning_rate 1e-3 --save_every 20000 --num_layers 3

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 052_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150\
        --new_action --exp_action --out_suffix=_flex_two_slopes_oracle_60_obs_fix_oracle_resume_051 \
            --flex_oracle --two_slopes_action --obs_type obs_60_keys --learning_rate 1e-5 --save_every 20000 --num_layers 3 \
                --load_path output/training/ongoing/051_onbc_seed_0_flex_two_slopes_oracle_60_obs_fix_oracle

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 053_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic_60_obs_resume_050 --seed 0\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 10000 --num_layers 3 \
                --load_path output/training/ongoing/050_onbc_seed_0_new_data_realistic_60_obs_fix_oracle

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 054_ \
    --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 \
        --new_action --exp_action --out_suffix=_new_data_realistic_60_obs_resume_050_with_bid_noise --seed 1\
            --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 --num_layers 3 \
                --auction_noise_min 0.0 --auction_noise_max 0.2 \
                    --load_path output/training/ongoing/050_onbc_seed_0_new_data_realistic_60_obs_fix_oracle
                    
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 056_ \
    --budget_min 1000 --budget_max 3000 --target_cpa_min 40 --target_cpa_max 90 \
        --new_action --exp_action --out_suffix=_specialize_050_1000_3000_40_90 --seed 1 \
            --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 --num_layers 3 \
                --load_path output/training/ongoing/050_onbc_seed_0_new_data_realistic_60_obs_fix_oracle
                
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 057_ \
    --budget_min 2000 --budget_max 4000 --target_cpa_min 40 --target_cpa_max 90 \
        --new_action --exp_action --out_suffix=_specialize_050_2000_4000_40_90 --seed 1 \
            --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 --num_layers 3 \
                --load_path output/training/ongoing/050_onbc_seed_0_new_data_realistic_60_obs_fix_oracle

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 058_ \
    --budget_min 3000 --budget_max 5000 --target_cpa_min 40 --target_cpa_max 90 \
        --new_action --exp_action --out_suffix=_specialize_050_3000_5000_40_90 --seed 1 \
            --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 --num_layers 3 \
                --load_path output/training/ongoing/050_onbc_seed_0_new_data_realistic_60_obs_fix_oracle
"""
