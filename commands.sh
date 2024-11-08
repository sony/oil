scp -r -i "Downloads/alberto.pem" ./Downloads/online_rl_data.zip ubuntu@52.207.170.60:/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data
scp -r -i "Downloads/alberto.pem" ./Downloads/raw_traffic_parquet.zip ubuntu@52.207.170.60:/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data 
scp -r -i "Downloads/alberto.pem" ubuntu@3.232.120.127:/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/data/online_rl_data.zip ./Downloads

scp -r -i "Downloads/alberto.pem" ubuntu@52.207.170.60:/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/output/training/ongoing/050_onbc_seed_0_new_data_realistic_60_obs_fix_oracle ./Downloads
    scp -r -i "Downloads/alberto.pem" ./Downloads/050_onbc_seed_0_new_data_realistic_60_obs_fix_oracle ubuntu@3.232.120.127:/home/ubuntu/Dev/NeurIPS_Auto_Bidding_General_Track_Baseline/output/training/ongoing