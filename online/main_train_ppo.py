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
from online.callbacks.custom_callbacks import CustomCheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from definitions import ROOT_DIR, OBS_CONFIG_PATH, ACT_CONFIG_PATH
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
    default=[256, 256, 256],
    help="Layer sizes for the policy and value networks",
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
    default=10_000,
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
    "--pg_coef",
    type=float,
    default=1.0,
    help="Policy gradient coefficient",
)
parser.add_argument(
    "--imitation_coef",
    type=float,
    default=0.0,
    help="Imitation coefficient",
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
    "--ent_coef",
    type=float,
    default=3e-06,
    help="Entropy coefficient",
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
    "--oracle_upgrade",
    action="store_true",
    help="Use flexible oracle",
)
parser.add_argument(
    "--two_slopes_action",
    action="store_true",
    help="Use two slopes for the action transformation",
)
parser.add_argument(
    "--single_io_bid",
    action="store_true",
    help="Predict for each IO",
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
    "--data_folder_name",
    type=str,
    default="online_rl_data_sparse",
    help="Data folder name",
)
args = parser.parse_args()

run_name = f"{args.out_prefix}ppo_seed_{args.seed}{args.out_suffix}"
TENSORBOARD_LOG = os.path.join(ROOT_DIR, "output", "training", "ongoing", run_name)

# Reward structure and task parameters:
with open(OBS_CONFIG_PATH / f"{args.obs_type}.json", "r") as f:
    obs_keys = json.load(f)
with open(ACT_CONFIG_PATH / f"{args.act_type}.json", "r") as f:
    act_keys = json.load(f)

config_list = []
for period in range(7, 7 + args.num_envs):  # one perwith_ad_idx
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
            "obs_keys": obs_keys,
            "act_keys": act_keys,
            "stochastic_exposure": args.stochastic_exposure,
            "exclude_self_bids": args.exclude_self_bids,
            "oracle_upgrade": args.oracle_upgrade or args.single_io_bid,
            "two_slopes_action": args.two_slopes_action,
            "single_io_bid": args.single_io_bid,
            "batch_state": args.batch_state,
            "advertiser_categories": args.advertiser_categories,
            "seed": args.seed,
        }
    )

model_config = dict(
    policy="MlpPolicy",
    device=args.device,
    batch_size=args.batch_size,
    n_steps=args.n_rollout_steps,
    learning_rate=lambda x: x * args.learning_rate,
    ent_coef=args.ent_coef,
    vf_coef=0.5,
    pg_coef=args.pg_coef,
    imitation_coef=args.imitation_coef,
    clip_range=0.3,
    gamma=0.99,
    gae_lambda=0.9,
    max_grad_norm=0.7,
    n_epochs=10,
    use_sde=False,
    policy_kwargs=dict(
        log_std_init=args.log_std_init,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=args.net_arch, vf=args.net_arch),
    ),
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

    checkpoint_callback = CustomCheckpointCallback(
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
        algo="bc_ppo",
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
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 006_ \
    --seed 0 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_dense_reward_two_slopes --obs_type obs_60_keys --learning_rate 2e-5 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_sparse

python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 007_ \
    --seed 1 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_dense_reward_two_slopes --obs_type obs_60_keys --learning_rate 2e-5 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_sparse

python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 008_ \
    --seed 2 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_dense_reward_two_slopes --obs_type obs_60_keys --learning_rate 2e-5 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_sparse

python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 009_ \
    --seed 0 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_dense_reward_two_slopes_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_sparse

python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 010_ \
    --seed 1 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_dense_reward_two_slopes_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_sparse

python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 011_ \
    --seed 2 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_dense_reward_two_slopes_medium_bids_lin_lr --obs_type obs_60_keys --learning_rate 1e-3 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_sparse
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 015_ \
    --seed 0 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_sparse
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 016_ \
    --seed 1 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_sparse
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 017_ \
    --seed 2 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_sparse
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 018_ \
    --seed 0 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_dense
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 019_ \
    --seed 1 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_dense
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 020_ \
    --seed 2 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_dense
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 021_ \
    --seed 0 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_dense_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --data_folder_name online_rl_data_sparse
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 022_ \
    --seed 1 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_dense_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --data_folder_name online_rl_data_sparse
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 023_ \
    --seed 2 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_dense_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --data_folder_name online_rl_data_sparse

# Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 024_ \
    --seed 0 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_dense_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --data_folder_name online_rl_data_dense
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 025_ \
    --seed 1 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_dense_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --data_folder_name online_rl_data_dense
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 026_ \
    --seed 2 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_dense_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --data_folder_name online_rl_data_dense



Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 033_ \
    --seed 0 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_sparse_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 1 --dense_weight 0 \
                --data_folder_name online_rl_data_sparse
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 034_ \
    --seed 1 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_sparse_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 1 --dense_weight 0 \
                --data_folder_name online_rl_data_sparse
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 035_ \
    --seed 2 --budget_min 1000 --budget_max 6000 --target_cpa_min 50 --target_cpa_max 150 --exclude_self_bids\
        --out_suffix=_sparse_dataset_sparse_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 1 --dense_weight 0 \
                --data_folder_name online_rl_data_sparse
Done
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 036_ \
    --seed 0 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1
                --data_folder_name online_rl_data_dense

python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 037_ \
    --seed 1 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1
                --data_folder_name online_rl_data_dense
Old
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 038_ \
    --seed 2 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1
                --data_folder_name online_rl_data_dense

PPO dense two slopes dense
cpu 3
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 055_ \
    --seed 0 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_dense
cpu 2
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 056_ \
    --seed 1 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_dense
cpu 2
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 057_ \
    --seed 2 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_dense_reward_two_slopes_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --oracle_upgrade --two_slopes_action --data_folder_name online_rl_data_dense
                
PPO dense dense
cpu 2
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 068_ \
    --seed 0 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_dense_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --data_folder_name online_rl_data_dense
cpu 2
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 069_ \
    --seed 1 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_dense_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --data_folder_name online_rl_data_dense
cpu 3
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 070_ \
    --seed 2 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_dense_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 0 --dense_weight 1 \
                --data_folder_name online_rl_data_dense

PPO sparse dense
cpu 3
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 071_ \
    --seed 0 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_sparse_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 1 --dense_weight 0 \
                --data_folder_name online_rl_data_dense
cpu 1
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 072_ \
    --seed 1 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_sparse_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 1 --dense_weight 0 \
                --data_folder_name online_rl_data_dense
cpu 1
python online/main_train_ppo.py --num_envs 20 --batch_size 512 --num_steps 10_000_000 --out_prefix 073_ \
    --seed 2 --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --exclude_self_bids\
        --out_suffix=_dense_dataset_sparse_reward_medium_bids_smaller_lin_lr --obs_type obs_60_keys --learning_rate 1e-4 --save_every 10000 \
            --net_arch 256 256 256 --imitation_coef 0 --pg_coef 1 --sparse_weight 1 --dense_weight 0 \
                --data_folder_name online_rl_data_dense

"""
