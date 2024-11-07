import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import logging
import numpy as np
import torch
import os
import argparse
import json
import shutil
import imitation
import wandb
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.bc import BC
from online.train.dagger import OracleDaggerTrainer
from stable_baselines3.common.policies import ActorCriticPolicy
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from imitation.algorithms.dagger import LinearBetaSchedule
from envs.helpers import get_model_and_env_path
from helpers import my_safe_to_tensor
from train.bc import BCLossCalculatorWithMSE



imitation.util.util.safe_to_tensor = (
    my_safe_to_tensor  # The standard my_safe_to_tensor is buggy
)
torch.manual_seed(0)
rng = np.random.default_rng(0)

logging.getLogger().setLevel(logging.INFO)


parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument(
    "--project_name",
    type=str,
    default="alibaba",
    help="Project name for wandb",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Seed for random number generator",
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
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments",
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
    "--net_arch",
    type=int,
    nargs="*",
    default=[256, 256],
    help="Layer sizes for the policy and value networks",
)
parser.add_argument(
    "--budget_min",
    type=float,
    default=400,
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
    "--advertiser_id",
    type=int,
    default=None,
    help="Advertiser ID - can be fixed for debugging",
)
parser.add_argument(
    "--deterministic_conversion",
    action="store_true",
    help="Use deterministic conversion equal to pvalue - for debugging",
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
parser.add_argument(
    "--num_buffer_episodes",
    type=int,
    default=100,
    help="Number of episodes to store in the buffer",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=1_000_000,
    help="Number of steps to train the agent",
)
parser.add_argument(
    "--min_rollout_episodes",
    type=int,
    default=2,
    help="Minimum number of episodes to rollout",
)
parser.add_argument(
    "--min_rollout_timesteps",
    type=int,
    default=100,
    help="Minimum number of timesteps to rollout",
)
parser.add_argument(
    "--rollout_reuse_epoch",
    type=int,
    default=1,
    help="Number of epochs to reuse the rollouts",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size for the BC training",
)
parser.add_argument(
    "--save_every",
    type=int,
    default=10,
    help="Save the model every n rounds",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-3,
    help="Learning rate for the BC training",
)
parser.add_argument(
    "--ent_weight",
    type=float,
    default=0.0,
    help="Entropy weight for the BC training",
)
parser.add_argument(
    "--l2_weight",
    type=float,
    default=1e-4,
    help="Weoght decay for the BC training",
)
parser.add_argument(
    "--neglogp_weight",
    type=float,
    default=1.0,
    help="Negative log prob weight for the BC training",
)
parser.add_argument(
    "--mse_weight",
    type=float,
    default=0.0,
    help="MSE weight for the BC training",
)
parser.add_argument(
    "--log_std_init",
    type=float,
    default=-1,
    help="Initial value for the log std of the policy",
)
parser.add_argument(
    "--beta_rampdown_rounds",
    type=int,
    default=50,
    help="Number of rounds to rampdown the beta param of DAgger",
)
# parser.add_argument(
#     "--log_interval",
#     type=int,
#     default=100,
#     help="Interval to log the BC training",
# )
# parser.add_argument(
#     "--log_rollouts_n_episodes",
#     type=int,
#     default=5,
#     help="Number of episodes to log the rollouts",
# )

args = parser.parse_args()

run_name = f"{args.out_prefix}dagger_seed_{args.seed}{args.out_suffix}"
TENSORBOARD_LOG = os.path.join(ROOT_DIR, "output", "training", "dagger", run_name)

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
            "advertiser_id": args.advertiser_id,
            "deterministic_conversion": args.deterministic_conversion,
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
        config=args.__dict__,
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

    model_path, env_path = get_model_and_env_path(
        TENSORBOARD_LOG, args.load_path, args.checkpoint_num
    )
    envs = make_parallel_envs(config_list)
    if env_path is not None:
        envs = VecNormalize.load(env_path, envs)
    else:
        envs = VecNormalize(envs)

    policy = ActorCriticPolicy(
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        lr_schedule=lambda _: 2e-5,
        net_arch=dict(pi=args.net_arch, vf=args.net_arch),
        activation_fn=torch.nn.ReLU,
        log_std_init=args.log_std_init,
    )

    custom_logger = imitation.util.logger.configure(
        folder=TENSORBOARD_LOG,
        format_strs=["stdout", "log", "csv", "wandb", "tensorboard"],
    )

    bc_trainer = BC(
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        rng=rng,
        policy=policy,
        batch_size=args.batch_size,
        minibatch_size=None,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=dict(lr=args.learning_rate),
        device="auto",
        custom_logger=custom_logger,
    )
    bc_trainer.loss_calculator = BCLossCalculatorWithMSE(
        ent_weight=args.ent_weight,
        l2_weight=args.l2_weight,
        neglogp_weight=args.neglogp_weight,
        mse_weight=args.mse_weight,
    )

    dagger_trainer = OracleDaggerTrainer(
        venv=envs,
        scratch_dir=TENSORBOARD_LOG,
        rng=rng,
        max_stored_trajs=args.num_buffer_episodes,
        beta_schedule=LinearBetaSchedule(rampdown_rounds=args.beta_rampdown_rounds),
        bc_trainer=bc_trainer,
        custom_logger=custom_logger,
    )
    dagger_trainer.train(
        total_timesteps=args.num_steps,
        rollout_round_min_episodes=args.min_rollout_episodes,
        rollout_round_min_timesteps=args.min_rollout_timesteps,
        save_every=args.save_every,
        bc_train_kwargs={
            "n_epochs": args.rollout_reuse_epoch,
            "on_epoch_end": None,
            "on_batch_end": None,
            # "log_interval": args.log_interval,
            # "log_rollouts_venv": envs,
            # "log_rollouts_n_episodes": args.log_rollouts_n_episodes,
            "progress_bar": True,
        }
    )

    reward, _ = evaluate_policy(dagger_trainer.policy, envs, 10)
    print("Reward:", reward)
    wandb.finish()

"""
python online/main_train_dagger.py --num_envs 1 --batch_size 32 --min_rollout_episodes 3 --min_rollout_timesteps 100 \
    --num_steps 100_000 --seed 0 --out_prefix "001_" --out_suffix "_test" --num_buffer_episodes 100 \
        --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding

python online/main_train_dagger.py --num_envs 20 --batch_size 256 --min_rollout_episodes 20 --min_rollout_timesteps 500 \
    --num_steps 10_000_000  --seed 0 --out_prefix "009_" --out_suffix "_no_vec_norm" --num_buffer_episodes 2000 --rollout_reuse_epoch 5\
        --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10

python online/main_train_dagger.py --num_envs 20 --batch_size 256 --min_rollout_episodes 20 --min_rollout_timesteps 500 \
    --num_steps 10_000_000  --seed 0 --out_prefix "010_" --out_suffix "_no_vec_norm" --num_buffer_episodes 2000 --rollout_reuse_epoch 1\
        --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 \
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10

python online/main_train_dagger.py --num_envs 20 --batch_size 256 --min_rollout_episodes 20 --min_rollout_timesteps 500 \
    --num_steps 10_000_000  --seed 0 --out_prefix "011_" --out_suffix "_wd_small_lr" --num_buffer_episodes 2000 --rollout_reuse_epoch 1\
        --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10

python online/main_train_dagger.py --num_envs 20 --batch_size 256 --min_rollout_episodes 20 --min_rollout_timesteps 500 \
    --num_steps 10_000_000  --seed 0 --out_prefix "012_" --out_suffix "_wd_small_lr_beta_50" --num_buffer_episodes 2000 --rollout_reuse_epoch 1\
        --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 50
            
python online/main_train_dagger.py --num_envs 20 --batch_size 256 --min_rollout_episodes 20 --min_rollout_timesteps 500 \
    --num_steps 10_000_000  --seed 0 --out_prefix "013_" --out_suffix "_wd_small_lr_beta_0" --num_buffer_episodes 2000 --rollout_reuse_epoch 1\
        --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 0
            
python online/main_train_dagger.py --num_envs 1 --batch_size 256 --min_rollout_episodes 5 --min_rollout_timesteps 50 \
    --num_steps 10_000_000  --seed 0 --out_prefix "015_" --out_suffix "_overfit_1_ep" --num_buffer_episodes 2000 --rollout_reuse_epoch 1\
        --budget_min 8000 --budget_max 8000 --target_cpa_min 8 --target_cpa_max 8 --advertiser_id 0 --deterministic_conversion --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 1
            
python online/main_train_dagger.py --num_envs 1 --batch_size 256 --min_rollout_episodes 5 --min_rollout_timesteps 50 \
    --num_steps 10_000_000  --seed 0 --out_prefix "016_" --out_suffix "_overfit_1_ep" --num_buffer_episodes 100 --rollout_reuse_epoch 1\
        --budget_min 8000 --budget_max 8000 --target_cpa_min 8 --target_cpa_max 8 --advertiser_id 0 --deterministic_conversion --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 1
            
python online/main_train_dagger.py --num_envs 1 --batch_size 256 --min_rollout_episodes 5 --min_rollout_timesteps 50 \
    --num_steps 10_000_000  --seed 0 --out_prefix "017_" --out_suffix "_overfit_1_ep" --num_buffer_episodes 100 --rollout_reuse_epoch 1\
        --budget_min 8000 --budget_max 8000 --target_cpa_min 8 --target_cpa_max 8 --advertiser_id 0 --deterministic_conversion --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 50
            
python online/main_train_dagger.py --num_envs 1 --batch_size 256 --min_rollout_episodes 5 --min_rollout_timesteps 50 \
    --num_steps 10_000_000  --seed 0 --out_prefix "018_" --out_suffix "_overfit_1_ep_mse" --num_buffer_episodes 100 --rollout_reuse_epoch 1\
        --budget_min 8000 --budget_max 8000 --target_cpa_min 8 --target_cpa_max 8 --advertiser_id 0 --deterministic_conversion --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 1 \
                --mse_weight 1.0 --neglogp_weight 0.0 --ent_weight 0.0 --l2_weight 1e-4
                
python online/main_train_dagger.py --num_envs 1 --batch_size 256 --min_rollout_episodes 5 --min_rollout_timesteps 50 \
    --num_steps 10_000_000  --seed 0 --out_prefix "019_" --out_suffix "_overfit_1_ep_mse_beta_50" --num_buffer_episodes 100 --rollout_reuse_epoch 1\
        --budget_min 8000 --budget_max 8000 --target_cpa_min 8 --target_cpa_max 8 --advertiser_id 0 --deterministic_conversion --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 50 \
                --mse_weight 1.0 --neglogp_weight 0.0 --ent_weight 0.0 --l2_weight 1e-4
                
python online/main_train_dagger.py --num_envs 1 --batch_size 256 --min_rollout_episodes 5 --min_rollout_timesteps 50 \
    --num_steps 10_000_000  --seed 0 --out_prefix "020_" --out_suffix "_overfit_1_ep_mse_beta_50" --num_buffer_episodes 100 --rollout_reuse_epoch 1\
        --budget_min 8000 --budget_max 8000 --target_cpa_min 8 --target_cpa_max 8 --advertiser_id 0 --deterministic_conversion --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 50 \
                --mse_weight 1.0 --neglogp_weight 0.0 --ent_weight 0.0 --l2_weight 1e-4
                
python online/main_train_dagger.py --num_envs 1 --batch_size 256 --min_rollout_episodes 5 --min_rollout_timesteps 50 \
    --num_steps 10_000_000  --seed 0 --out_prefix "021_" --out_suffix "_overfit_1_ep_mse" --num_buffer_episodes 100 --rollout_reuse_epoch 1\
        --budget_min 8000 --budget_max 8000 --target_cpa_min 8 --target_cpa_max 8 --advertiser_id 0 --deterministic_conversion --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 1 \
                --mse_weight 1.0 --neglogp_weight 0.0 --ent_weight 0.0 --l2_weight 1e-4

python online/main_train_dagger.py --num_envs 1 --batch_size 256 --min_rollout_episodes 20 --min_rollout_timesteps 50 \
    --num_steps 10_000_000  --seed 0 --out_prefix "022_" --out_suffix "_overfit_1_ep_mse_beta_50" --num_buffer_episodes 1000 --rollout_reuse_epoch 1\
        --budget_min 8000 --budget_max 8000 --target_cpa_min 8 --target_cpa_max 8 --advertiser_id 0 --deterministic_conversion --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 50 \
                --mse_weight 1.0 --neglogp_weight 0.0 --ent_weight 0.0 --l2_weight 1e-4
                
                
python online/main_train_dagger.py --num_envs 1 --batch_size 256 --min_rollout_episodes 20 --min_rollout_timesteps 50 \
    --num_steps 10_000_000  --seed 0 --out_prefix "023_" --out_suffix "_overfit_1_ep_mse_beta_50_large_buffer" --num_buffer_episodes 10000 --rollout_reuse_epoch 1\
        --budget_min 8000 --budget_max 8000 --target_cpa_min 8 --target_cpa_max 8 --advertiser_id 0 --deterministic_conversion --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 50 \
                --mse_weight 1.0 --neglogp_weight 0.0 --ent_weight 0.0 --l2_weight 1e-4
                
python online/main_train_dagger.py --num_envs 1 --batch_size 256 --min_rollout_episodes 20 --min_rollout_timesteps 50 \
    --num_steps 10_000_000  --seed 0 --out_prefix "023_" --out_suffix "_overfit_1_ep_mse_beta_50_large_buffer" --num_buffer_episodes 10000 --rollout_reuse_epoch 1\
        --budget_min 8000 --budget_max 8000 --target_cpa_min 8 --target_cpa_max 8 --advertiser_id 0 --deterministic_conversion --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 50 \
                --mse_weight 1.0 --neglogp_weight 0.0 --ent_weight 0.0 --l2_weight 1e-4
                
python online/main_train_dagger.py --num_envs 1 --batch_size 256 --min_rollout_episodes 20 --min_rollout_timesteps 50 \
    --num_steps 10_000_000  --seed 0 --out_prefix "024_" --out_suffix "_overfit_1_ep_mse_beta_50_large_buffer" --num_buffer_episodes 10000 --rollout_reuse_epoch 1\
        --budget_min 8000 --budget_max 8000 --target_cpa_min 8 --target_cpa_max 8 --advertiser_id 0 --deterministic_conversion --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 50 \
                --mse_weight 1.0 --neglogp_weight 0.0 --ent_weight 0.0 --l2_weight 0
                
python online/main_train_dagger.py --num_envs 1 --batch_size 256 --min_rollout_episodes 20 --min_rollout_timesteps 50 \
    --num_steps 10_000_000  --seed 0 --out_prefix "025_" --out_suffix "_overfit_1_ep_mse_beta_1" --num_buffer_episodes 100 --rollout_reuse_epoch 1\
        --budget_min 8000 --budget_max 8000 --target_cpa_min 8 --target_cpa_max 8 --advertiser_id 0 --deterministic_conversion --learning_rate 1e-4\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 1 \
                --mse_weight 1.0 --neglogp_weight 0.0 --ent_weight 1e-5 --l2_weight 1e-4 --log_std_init -0.5
                
python online/main_train_dagger.py --num_envs 20 --batch_size 256 --min_rollout_episodes 40 --min_rollout_timesteps 50 \
    --num_steps 10_000_000  --seed 0 --out_prefix "026_" --out_suffix "_mse_beta_1" --num_buffer_episodes 100 --rollout_reuse_epoch 1\
        --budget_min 400 --budget_max 12000 --target_cpa_min 6 --target_cpa_max 12 --learning_rate 1e-3\
            --new_action --exp_action --obs_type obs_29_keys --simplified_bidding --save_every 10 --beta_rampdown_rounds 1 \
                --mse_weight 1.0 --neglogp_weight 0.0 --ent_weight 1e-5 --l2_weight 1e-4 --log_std_init -0.5
"""