import pathlib
import sys
import wandb

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import logging
from offline.lp.onlineLp import OnlineLp
from definitions import ROOT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def train_model():
    """Create the reference dataset to deploy a linear LP agent"""
    seed = 2
    dataset = "dense"  # "dense", "sparse"

    save_path = ROOT_DIR / "output" / "offline" / f"online_lp_{dataset}_seed_{seed}"
    if dataset == "dense":
        data_path = ROOT_DIR / "data" / "traffic" / f"raw_traffic_parquet"
    elif dataset == "sparse":
        data_path = ROOT_DIR / "data" / "traffic" / f"raw_traffic_sparse_parquet"
    else:
        raise ValueError("Invalid dataset name")
    onlineLp = OnlineLp(data_path, dataset=dataset, seed=seed)
    onlineLp.train(save_path)


def run_onlineLp():
    """
    Run onlinelp model training and evaluation.
    """
    train_model()


if __name__ == "__main__":
    run_onlineLp()
