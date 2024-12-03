import os
import json
import shutil
import argparse
from definitions import ROOT_DIR


def extract_checkpoints(json_file, output_folder):

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


if __name__ == "__main__":

    default_json_path = ROOT_DIR / "data" / "results"

    output_folder = ROOT_DIR / "saved_model" / "paper"

    # Extract the checkpoints
    for json_file in ["eval_official.json", "eval_final.json"]:
        json_file = default_json_path / json_file
        extract_checkpoints(json_file, output_folder)
