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
from definitions import ROOT_DIR, OBS_CONFIG_PATH, ACT_CONFIG_PATH
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
    default="oil",
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
    default="paper",
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
    "--learning_rate",
    type=float,
    default=1e-3,
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
    "--flex_oracle_cost_weight",
    type=float,
    default=0.5,
    help="Cost weight for the flexible oracle",
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
parser.add_argument(
    "--advertiser_categories",
    type=int,
    nargs="+",
    default=None,
    help="Advertiser categories where to sample from, if None all are used",
)
parser.add_argument(
    "--advertiser_id",
    type=int,
    default=None,
    help="Advertiser id to use, if None all are used",
)
parser.add_argument(
    "--data_folder_name",
    type=str,
    default="online_rl_data_final_with_ad_idx",
    help="Data folder name",
)
args = parser.parse_args()

run_name = f"{args.out_prefix}oil_seed_{args.seed}{args.out_suffix}"
TENSORBOARD_LOG = os.path.join(ROOT_DIR, "output", "training", "ongoing", run_name)

# Reward structure and task parameters:
with open(OBS_CONFIG_PATH / f"{args.obs_type}.json", "r") as f:
    obs_keys = json.load(f)
with open(ACT_CONFIG_PATH / f"{args.act_type}.json", "r") as f:
    act_keys = json.load(f)

config_list = []
for period in range(7, 7 + args.num_envs):  # one period per env
    pvalues_df_path = (
        ROOT_DIR
        / "data"
        / "traffic"
        / args.data_folder_name
        / f"period-{period}_pvalues.parquet"
    )
    bids_df_path = (
        ROOT_DIR
        / "data"
        / "traffic"
        / args.data_folder_name
        / f"period-{period}_bids.parquet"
    )
    rwd_weights = {
        "dense": 0,
        "sparse": 1,
    }

    config_list.append(
        {
            "env_name": args.env_name,
            "pvalues_df_path": pvalues_df_path,
            "bids_df_path": bids_df_path,
            "budget_range": (args.budget_min, args.budget_max),
            "target_cpa_range": (args.target_cpa_min, args.target_cpa_max),
            "rwd_weights": rwd_weights,
            "obs_keys": obs_keys,
            "act_keys": act_keys,
            "stochastic_exposure": args.stochastic_exposure,
            "auction_noise": (
                args.auction_noise_min,
                args.auction_noise_max,
            ),
            "simplified_oracle": args.simplified_oracle,
            "exclude_self_bids": args.exclude_self_bids,
            "flex_oracle": args.flex_oracle or args.detailed_bid,
            "two_slopes_action": args.two_slopes_action,
            "flex_oracle_cost_weight": args.flex_oracle_cost_weight,
            "detailed_bid": args.detailed_bid,
            "batch_state": args.batch_state,
            "advertiser_categories": args.advertiser_categories,
            "seed": args.seed,
            "advertiser_id": args.advertiser_id,
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
    learning_rate=lambda x: x * args.learning_rate,
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
# OIL MLP first set of exp. (expert and medium opponents)
TODO: use lr schedule
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 000_ \
    --seed 0 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_flex_oracle_two_slopes --obs_type obs_60_keys --learning_rate 2e-5 --save_every 10000 \
            --num_layers 3 --flex_oracle --two_slopes_action --data_folder_name online_rl_data_final_expert_bids
TODO: use lr schedule
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 001_ \
    --seed 1 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_flex_oracle_two_slopes --obs_type obs_60_keys --learning_rate 2e-5 --save_every 10000 \
            --num_layers 3 --flex_oracle --two_slopes_action --data_folder_name online_rl_data_final_expert_bids

python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 002_ \
    --seed 2 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_flex_oracle_two_slopes --obs_type obs_60_keys --learning_rate 2e-5 --save_every 10000 \
            --num_layers 3 --flex_oracle --two_slopes_action --data_folder_name online_rl_data_final_expert_bids
Done
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 003_ \
    --seed 0 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_flex_oracle_two_slopes_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --flex_oracle --two_slopes_action --data_folder_name online_rl_data_final_with_ad_idx
Done
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 004_ \
    --seed 1 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_flex_oracle_two_slopes_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --flex_oracle --two_slopes_action --data_folder_name online_rl_data_final_with_ad_idx
Done
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 005_ \
    --seed 2 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_flex_oracle_two_slopes_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --flex_oracle --two_slopes_action --data_folder_name online_rl_data_final_with_ad_idx
Done
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 011_ \
    --seed 0 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_realistic_oracle_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --data_folder_name online_rl_data_final_with_ad_idx
Done
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 012_ \
    --seed 1 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_realistic_oracle_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --data_folder_name online_rl_data_final_with_ad_idx
Done
python online/main_train_oil.py --num_envs 20 --b/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/traffic/raw_traffic_parquetatch_size 512 --num_steps 10_000_000 --out_prefix 013_ \
    --seed 2 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_realistic_oracle_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --data_folder_name online_rl_data_final_with_ad_idx


python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 003_ \
    --seed 0 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_flex_oracle_two_slopes_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --flex_oracle --two_slopes_action --data_folder_name online_rl_data_final_with_ad_idx

python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 004_ \
    --seed 1 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_flex_oracle_two_slopes_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --flex_oracle --two_slopes_action --data_folder_name online_rl_data_final_with_ad_idx

python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 005_ \
    --seed 2 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_flex_oracle_two_slopes_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --flex_oracle --two_slopes_action --data_folder_name online_rl_data_final_with_ad_idx
Done
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 011_ \
    --seed 0 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_realistic_oracle_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --data_folder_name online_rl_data_final_with_ad_idx
Done
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 012_ \
    --seed 1 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_realistic_oracle_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --data_folder_name online_rl_data_final_with_ad_idx
TO CHECK
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 013_ \
    --seed 2 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_final_dataset_realistic_oracle_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --data_folder_name online_rl_data_final_with_ad_idx

Done
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 039_ \
    --seed 0 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_official_dataset_realistic_oracle_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --data_folder_name online_rl_data_with_ad_idx
Done
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 040_ \
    --seed 1 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_official_dataset_realistic_oracle_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --data_folder_name online_rl_data_with_ad_idx
Done
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 041_ \
    --seed 2 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_official_dataset_realistic_oracle_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --num_layers 3 --data_folder_name online_rl_data_with_ad_idx
Done
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 042_ \
    --seed 0 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_official_dataset_flex_oracle_two_slopes_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --flex_oracle --two_slopes_action --num_layers 3 --data_folder_name online_rl_data_with_ad_idx
Done
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 043_ \
    --seed 1 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_official_dataset_flex_oracle_two_slopes_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --flex_oracle --two_slopes_action --num_layers 3 --data_folder_name online_rl_data_with_ad_idx
Done
python online/main_train_oil.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 044_ \
    --seed 2 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_official_dataset_flex_oracle_two_slopes_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --flex_oracle --two_slopes_action --num_layers 3 --data_folder_name online_rl_data_with_ad_idx


"""
