import os
import json
import shutil
import argparse


def extract_checkpoints(json_file, output_folder):
    # If the outut folder exists delete it
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Recreate the output folder
    os.makedirs(output_folder)

    # Read the JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract the checkpoints and write them to new files
    for entry in data:
        advertiser_path = entry.get("path", None)
        checkpoint = entry.get("checkpoint", None)

        if checkpoint is not None:
            sorce_checkpoint_path = "/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/saved_model/ONBC"

            full_checkpoint_path = os.path.join(sorce_checkpoint_path, advertiser_path)

            checkpoint_zip = f"rl_model_{checkpoint}_steps.zip"
            vecnormalize_pkl = f"rl_model_vecnormalize_{checkpoint}_steps.pkl"

            # full paths to the source files
            source_zip_path = os.path.join(full_checkpoint_path, checkpoint_zip)
            source_pkl_path = os.path.join(full_checkpoint_path, vecnormalize_pkl)
            model_config_path = os.path.join(full_checkpoint_path, "model_config.json")
            env_config_path = os.path.join(full_checkpoint_path, "env_config.json")

            # Create a folder inside the output folder named after the advertiser path
            destination_advertiser_folder = os.path.join(output_folder, advertiser_path)
            os.makedirs(destination_advertiser_folder, exist_ok=True)

            # Check if the files exist and copy them
            if os.path.exists(source_zip_path):
                shutil.copy2(source_zip_path, destination_advertiser_folder)
                print(f"Copied {source_zip_path} to {destination_advertiser_folder}")
            else:
                raise ValueError(f"File {source_zip_path} does not exist")

            if os.path.exists(source_pkl_path):
                shutil.copy2(source_pkl_path, destination_advertiser_folder)
                print(f"Copied {source_pkl_path} to {destination_advertiser_folder}")
            else:
                raise ValueError(f"File {source_pkl_path} does not exist")

            shutil.copy2(model_config_path, destination_advertiser_folder)
            print(f"Copied {model_config_path} to {destination_advertiser_folder}")

            shutil.copy2(env_config_path, destination_advertiser_folder)
            print(f"Copied {env_config_path} to {destination_advertiser_folder}")
        else:
            print("No checkpoint found in the entry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy checkpoint files from ensemble.json"
    )
    parser.add_argument(
        "--file_name", type=str, default="ensemble_36", help="Path to the JSON file"
    )
    default_json_path = (
        "/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/ensembles"
    )
    # Parse the arguments
    args = parser.parse_args()
    json_file = default_json_path + "/" + args.file_name + ".json"
    output_folder = f"/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/saved_model/{args.file_name}"

    # Extract the checkpoints
    extract_checkpoints(json_file, output_folder)
