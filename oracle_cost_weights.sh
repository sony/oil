#!/bin/bash

# Define the range for flex_oracle_cost_weight
start=0
end=1
num_values=21

# Compute step size
step=$(echo "scale=10; ($end - $start) / ($num_values - 1)" | bc)

# Output file
output_file="result.txt"

# Remove the existing output file if it exists
rm -f $output_file

# Loop through the 21 values
for i in $(seq 0 $(($num_values - 1)))
do
    # Compute the current value of flex_oracle_cost_weight
    flex_weight=$(echo "scale=10; $start + $i * $step" | bc)

    # Run the python command with the current flex_oracle_cost_weight value
    echo "Running with flex_oracle_cost_weight=$flex_weight"
    echo "Running with flex_oracle_cost_weight=$flex_weight" >> $output_file
    python online/main_eval.py --algo onbc --experiment_path=output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
        --num_episodes=100 --no_save_df --deterministic --checkpoint 4600000 \
        --eval_config_path=/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/env_configs/eval_config_realistic.json \
        --compute_flex_topline --two_slopes_action --flex_oracle_cost_weight $flex_weight >> $output_file 2>&1
done
