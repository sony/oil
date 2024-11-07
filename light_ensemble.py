import pathlib
import json
import shutil
import argparse


def copy_checkpoint_recursive(advertiser_path, checkpoint, out_path):
    source_checkpoint_path = pathlib.Path(
        "/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/saved_model/ONBC"
    )
    alternate_source_checkpoint_path = pathlib.Path(
        "/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/output/training/ongoing"
    )
    full_checkpoint_path = source_checkpoint_path / advertiser_path
    if not full_checkpoint_path.exists():
        full_checkpoint_path = alternate_source_checkpoint_path / advertiser_path

    if checkpoint is None:
        # Get the highest checkpoint number from the load path folder
        checkpoint_files = list(
            full_checkpoint_path.glob("rl_model_*.zip")
        )
        checkpoint_numbers = [
            int(str(file).split("_")[-2]) for file in checkpoint_files
        ]
        checkpoint = max(checkpoint_numbers)
        
    checkpoint_zip = f"rl_model_{checkpoint}_steps.zip"
    vecnormalize_pkl = f"rl_model_vecnormalize_{checkpoint}_steps.pkl"

    # full paths to the source files
    source_zip_path = full_checkpoint_path / checkpoint_zip
    source_pkl_path = full_checkpoint_path / vecnormalize_pkl
    model_config_path = full_checkpoint_path / "model_config.json"
    env_config_path = full_checkpoint_path / "env_config.json"
    args_path = full_checkpoint_path / "args.json"

    # Create a folder inside the output folder named after the advertiser path
    destination_advertiser_folder = out_path / advertiser_path
    destination_advertiser_folder.mkdir(parents=True, exist_ok=True)

    # Check if the files exist and copy them
    if source_zip_path.exists():
        shutil.copy2(source_zip_path, destination_advertiser_folder)
        print(f"Copied {source_zip_path} to {destination_advertiser_folder}")
    else:
        raise ValueError(f"File {source_zip_path} does not exist")

    if source_pkl_path.exists():
        shutil.copy2(source_pkl_path, destination_advertiser_folder)
        print(f"Copied {source_pkl_path} to {destination_advertiser_folder}")
    else:
        raise ValueError(f"File {source_pkl_path} does not exist")

    shutil.copy2(model_config_path, destination_advertiser_folder)
    print(f"Copied {model_config_path} to {destination_advertiser_folder}")

    shutil.copy2(env_config_path, destination_advertiser_folder)
    print(f"Copied {env_config_path} to {destination_advertiser_folder}")

    shutil.copy2(args_path, destination_advertiser_folder)
    print(f"Copied {args_path} to {destination_advertiser_folder}")

    with open(args_path, "r") as f:
        args = json.load(f)
        if args.get("load_path") is not None:
            checkpoint_num = args.get("checkpoint_num")
            exp_name = args["load_path"].split("/")[-1]
            copy_checkpoint_recursive(exp_name, checkpoint_num, out_path)
    print("No checkpoint found in the entry:", advertiser_path)


def extract_checkpoints(json_file, out_path):
    # If the outut folder exists delete it
    if out_path.exists():
        shutil.rmtree(out_path)

    # Recreate the output folder
    out_path.mkdir(parents=True, exist_ok=True)

    # Read the JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract the checkpoints and write them to new files
    for entry in data:
        advertiser_path = entry.get("path", None)
        checkpoint = entry.get("checkpoint", None)
        copy_checkpoint_recursive(advertiser_path, checkpoint, out_path)


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
    out_path = pathlib.Path(f"/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/saved_model/{args.file_name}")

    # Extract the checkpoints
    extract_checkpoints(json_file, out_path)
