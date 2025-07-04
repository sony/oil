import json
import os
from dataclasses import dataclass, field
from typing import List
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from definitions import ALGO_CLASS_DICT


@dataclass
class SingleEnvTrainer:
    algo: str
    envs: VecNormalize
    load_model_path: str
    log_dir: str
    model_config: dict = field(default_factory=dict)
    callbacks: List[BaseCallback] = field(default_factory=list)
    timesteps: int = 10_000_000

    def __post_init__(self):
        self.dump_configs(path=self.log_dir)
        self.agent = self._init_agent()

    def dump_configs(self, path: str) -> None:
        with open(os.path.join(path, "model_config.json"), "w", encoding="utf8") as f:
            json.dump(
                self.model_config, f, indent=4, default=lambda _: "<not serializable>"
            )

    def _init_agent(self):
        algo_class = self.get_algo_class()
        if self.load_model_path is not None:
            agent = algo_class.load(
                self.load_model_path,
                env=self.envs,
                tensorboard_log=self.log_dir,
                custom_objects=self.model_config,
            )
        else:
            print("\nNo model path provided. Initializing new model.\n")
            agent = algo_class(
                env=self.envs,
                verbose=2,
                tensorboard_log=self.log_dir,
                **self.model_config,
            )
        return agent

    def train(self) -> None:
        self.agent.learn(
            total_timesteps=self.timesteps,
            callback=self.callbacks,
            reset_num_timesteps=False,
        )

    def save(self) -> None:
        self.agent.save(os.path.join(self.log_dir, "sparse_model.pkl"))
        self.envs.save(os.path.join(self.log_dir, "sparse_env.pkl"))

    def get_algo_class(self):
        if self.algo in ALGO_CLASS_DICT:
            return ALGO_CLASS_DICT[self.algo]
        else:
            raise ValueError("Unknown algorithm ", self.algo)


if __name__ == "__main__":
    print("This is a module. Run main.py to train the agent.")
