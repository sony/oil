import os
import json
import shutil
import argparse
from definitions import ROOT_DIR


def copy_experiments(json_file, output_folder):

    output_folder.mkdir(parents=True, exist_ok=True)

    # Read the JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract the checkpoints and write them to new files
    for _, exp_list in data.items():
        for exp in exp_list:
            exp_name = exp["experiment"]

            exp_path = ROOT_DIR / "output" / "training" / "ongoing" / exp_name
            dest_folder = output_folder / exp_name

            if exp_path.exists():
                dest_folder.mkdir(parents=True, exist_ok=True)
                for item in exp_path.rglob("*"):
                    relative_path = item.relative_to(exp_path)
                    dest_item = dest_folder / relative_path

                    if item.is_dir():
                        # Create directories in the destination
                        dest_item.mkdir(parents=True, exist_ok=True)
                    elif item.is_file():
                        # Handle files
                        if (
                            item.name == "rl_model_10000000_steps.zip"
                            or item.name == "rl_model_vecnormalize_10000000_steps.pkl"
                            or not item.name.startswith("rl_model_")
                        ):
                            shutil.copy2(item, dest_item)

def copy_folder_contents(src_path, dest_path):

    # Ensure the destination folder exists
    dest_path.mkdir(parents=True, exist_ok=True)

    # Copy each item in the source folder
    for item in src_path.iterdir():
        dest_item = dest_path / item.name
        if item.is_dir():
            # Recursively copy subdirectories
            shutil.copytree(item, dest_item, dirs_exist_ok=True)  # Use dirs_exist_ok=True for Python 3.8+
        else:
            # Copy files
            shutil.copy2(item, dest_item)

if __name__ == "__main__":

    default_json_path = ROOT_DIR / "data" / "results"
    offline_exp_path = ROOT_DIR / "output" / "offline"
    output_folder = ROOT_DIR / "pretrained"

    # Extract the checkpoints
    for json_file in ["eval_dense.json", "eval_sparse.json"]:
        json_file = default_json_path / json_file
        copy_experiments(json_file, output_folder)
        copy_folder_contents(offline_exp_path, output_folder) 

        
