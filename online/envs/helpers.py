import glob
import os
import numpy as np
from definitions import ROOT_DIR, MODEL_PATTERN


def get_number(filename):
    return int(filename.split("_steps.zip")[0].split("_")[-1])


def get_last_checkpoint(path):
    model_list = sorted(
        glob.glob(os.path.join(path, MODEL_PATTERN)),
        key=get_number,
    )
    checkpoints_list = [get_number(el) for el in model_list]
    if len(checkpoints_list) > 0:
        return max(checkpoints_list)
    else:
        return None


def get_model_and_env_path(tensorboard_log, load_path, checkpoint_num):
    """This function is used to robustly recover the checkpoint when the training is interrupted.
    When tensorboard_log already exists, the function looks for the latest checkpoint in such
    folder. This is done so that an automatic restart of the training resumes from the last
    available checkpoint, ignoring load_path and checkpoint_num. If tensorboard_log does not
    exist or has no checkpoint, the training starts either from the specified checkpoint_num, or,
    if no checkpoint_num is specified, from the last checkpoint of load_path. If load_path has
    no checkpoint, the training starts from scratch

    :param tensorboard_log: path of the Tensorboard log directory
    :param load_path: path of the directory of the experiment we want to resume
    :param checkpoint_num: number of the checkpoint to load
    :return: model_path and env_path
    """
    if os.path.isdir(tensorboard_log):
        # The folder already exists, then we resume the training if there are already checkpoints
        c_n = get_last_checkpoint(tensorboard_log)
        if c_n is None:
            print(
                f"No previous checkpoint found at {tensorboard_log}."
            )
        else:
            load_path = tensorboard_log
            checkpoint_num = c_n
            if load_path is not None:
                print(
                    f"A checkpoint was found at {tensorboard_log}, so we are resuming from there."
                    f"Ignoring any checkpoint at {load_path}."
                )

    if load_path is not None:
        if checkpoint_num is None:
            checkpoint_num = get_last_checkpoint(load_path)
        if checkpoint_num is None:
            print(
                f"No checkpoints at the given path {load_path}, starting a new training"
            )
            model_path = None
            env_path = None
        else:
            model_path = os.path.join(
                ROOT_DIR, load_path, f"rl_model_{checkpoint_num}_steps.zip"
            )
            env_path = os.path.join(
                load_path, f"rl_model_vecnormalize_{checkpoint_num}_steps.pkl"
            )
            print(
                f"Resuming training from checkpoint {checkpoint_num} of {load_path}")
    else:
        model_path = None
        env_path = None
    return model_path, env_path

def safe_mean(arr):
    if len(arr) == 0:
        return 0
    return np.mean(arr)