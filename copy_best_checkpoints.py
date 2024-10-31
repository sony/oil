import pandas as pd
import subprocess
import os
import argparse
import json


def extract_exp_name(file_path):
    # Extracts the experiment name from the file path by isolating the folder name just before the results CSV
    # Path looks like: output/testing/<exp_name>/results_xxx.csv
    exp_name = os.path.basename(os.path.dirname(file_path))
    return exp_name


def run_script(file_path, n):
    # Load the file with pandas
    df = pd.read_csv(file_path)

    # Sort the dataframe by the 'score' column
    df_sorted = df.sort_values(by="score", ascending=False).reset_index(drop=True)

    # Extract the experiment name from the path
    exp_name = extract_exp_name(file_path)
    print(exp_name)

    # Run the script 'copy_experiment.sh' n times
    checkpoint_list = []
    for i in range(min(n, len(df_sorted))):
        checkpoint = df_sorted.iloc[i]["checkpoint"]
        subprocess.run(["bash", "copy_experiment.sh", exp_name, str(int(checkpoint))])
        checkpoint_list.append(checkpoint)

    target_folder = f"saved_model/ONBC/{exp_name}"

    # Save a list of the sorted best checkpoints in the target folder
    with open(f"{target_folder}/best_checkpoints.json", "w") as f:
        json.dump(checkpoint_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run copy_experiment.sh for n times with specified file."
    )
    parser.add_argument("results_path", type=str, help="Path to the results CSV file")
    args = parser.parse_args()

    # Run the function
    run_script(args.results_path, 10)
