import os
import shutil
import argparse
import json
import torch.nn as nn
import wandb
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from datetime import datetime
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from metrics.custom_callbacks import EnvDumpCallback, TensorboardCallback
from train.trainer import SingleEnvTrainer
from envs.helpers import get_model_and_env_path


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
    help="Number of training steps once an environment is sampled",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="Batch size for training",
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
    default="arnold",
    help="Project name for wandb",
)
parser.add_argument(
    "--save_every",
    type=int,
    default=500_000,
    help="Save a checkpoint every N number of steps",
)
parser.add_argument(
    "--target_elevation",
    type=float,
    default=0.04,
    help="Target elevation",
)
parser.add_argument(
    "--traj_mul",
    type=int,
    default=1,
    help="Trajectory multiplier",
)
parser.add_argument(
    "--target_character",
    type=str,
    default="random",
    help="Target character for the BionicHand environment",
)
parser.add_argument(
    "--pose",
    type=float,
    default=1,
    help="Rwd pose",
)
parser.add_argument(
    "--force",
    type=float,
    default=0.1,
    help="Rwd force",
)
parser.add_argument(
    "--prev_pentip",
    type=float,
    default=-0.1,
    help="Rwd prev_pentip",
)
parser.add_argument(
    "--pentip_ground",
    type=float,
    default=-0.01,
    help="Rwd pentip_ground",
)
parser.add_argument('--flag_kill', action='store_true', default=False, #False
                    help='Flag to kill if > 20 ground')
parser.add_argument('--penfall_kill', action='store_true', default=False, #False
                    help='Flag to kill if > 20 ground')
args = parser.parse_args()

run_name = f"{args.env_name}_trajmul_{args.traj_mul}_target_elevation_{args.target_elevation}_pose_{args.pose}_force_{args.force}_prev_pentip_{args.prev_pentip}_pentip_ground_{args.pentip_ground}_flagkill_{args.flag_kill}_penfall_kill_{args.penfall_kill}_log_std_init_{args.log_std_init}_ppo_seed_{args.seed}{args.out_suffix}"
TENSORBOARD_LOG = os.path.join(ROOT_DIR, "output", "training", "ongoing", run_name)

rwd_dict = {
        "pose": args.pose,
        "force":args.force,
        "prev_pentip": args.prev_pentip,
        "pentip_ground": args.pentip_ground,
    }

# Reward structure and task parameters:
config = {
    "env_name": args.env_name,
    "target_character": args.target_character,
    "traj_mul": args.traj_mul,
    "reward_weight_dict": rwd_dict,
    "target_elevation":args.target_elevation,
    "flag_kill": args.flag_kill,
    "penfall_kill": args.penfall_kill,
}  # , "seed": args.seed}

model_config = dict(
    policy="MultiInputPolicy",
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
)


# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_config, num_env, start_index=0):
    def make_env(_):
        def _thunk():
            env = EnvironmentFactory.create(**env_config)
            # env.seed(args.seed)
            env = Monitor(env, TENSORBOARD_LOG)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":
    run = wandb.init(
        project=args.project_name,
        name=run_name,
        config=model_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    # ensure tensorboard log directory exists and copy this file to track
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)
    with open(os.path.join(TENSORBOARD_LOG, "args.json"), "w") as file:
        json.dump(args.__dict__, file, indent=4, default=lambda _: "<not serializable>")

    checkpoint_callback = CheckpointCallback(
        save_freq=int(args.save_every/args.num_envs),
        save_path=TENSORBOARD_LOG,
        save_vecnormalize=True,
        verbose=1,
    )

    tensorboard_callback = TensorboardCallback(
        info_keywords=(
            "pose",
            "force",
            "prev_pentip",
            # "solved",
            # "rwd_dense",
            # "done",
        ),
    )

    model_path, env_path = get_model_and_env_path(
        TENSORBOARD_LOG, args.load_path, args.checkpoint_num
    )

    # Create and wrap the training and evaluations environments
    envs = make_parallel_envs(config, args.num_envs)
    if env_path is not None:
        envs = VecNormalize.load(env_path, envs)
    else:
        envs = VecNormalize(envs)

    # Define trainer
    trainer = SingleEnvTrainer(
        algo="ppo",
        envs=envs,
        env_config=config,
        load_model_path=model_path,
        log_dir=TENSORBOARD_LOG,
        model_config=model_config,
        callbacks=[tensorboard_callback, checkpoint_callback],
        timesteps=args.num_steps,
    )

    # Train agent
    trainer.train()
    trainer.save()
