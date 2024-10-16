#!/bin/bash

# Variables
exp_name="$1"           # Experiment name
checkpoint="$2"         # Checkpoint number to copy
source_folder="output/training/ongoing/${exp_name}"
target_folder="saved_model/ONBC/${exp_name}"

# Create the target directory if it does not exist
mkdir -p "$target_folder"

# Copy all files that are not checkpoint files
find "$source_folder" -type f ! -name "rl_model_*_steps.zip" ! -name "rl_model_vecnormalize_*_steps.pkl" -exec cp {} "$target_folder" \;

# Copy the specified checkpoint files
cp "${source_folder}/rl_model_${checkpoint}_steps.zip" "${target_folder}/"
cp "${source_folder}/rl_model_vecnormalize_${checkpoint}_steps.pkl" "${target_folder}/"

echo "Files for checkpoint ${checkpoint} copied to ${target_folder}"
