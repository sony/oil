python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 420_ \
    --budget_min 2900 --budget_max 2900 --target_cpa_min 100 --target_cpa_max 100 --advertiser_id 0 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_0_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 421_ \
    --budget_min 4300 --budget_max 4300 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 1 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_1_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 422_ \
    --budget_min 3000 --budget_max 3000 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 2 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_2_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 423_ \
    --budget_min 2400 --budget_max 2400 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 3 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_3_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 424_ \
    --budget_min 4800 --budget_max 4800 --target_cpa_min 60 --target_cpa_max 60 --advertiser_id 4 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_4_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 425_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 130 --target_cpa_max 130 --advertiser_id 5 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_5_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids


python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 426_ \
    --budget_min 2050 --budget_max 2050 --target_cpa_min 120 --target_cpa_max 120 --advertiser_id 6 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_6_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure  --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 400_ \
    --budget_min 3500 --budget_max 3500 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 7 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_7_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 427_ \
    --budget_min 4600 --budget_max 4600 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 8 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_8_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 428_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 130 --target_cpa_max 130 --advertiser_id 9 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_9_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure  --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 429_ \
    --budget_min 2800 --budget_max 2800 --target_cpa_min 100 --target_cpa_max 100 --advertiser_id 10 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_10_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 430_ \
    --budget_min 2350 --budget_max 2350 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 11 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_11_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 431_ \
    --budget_min 2050 --budget_max 2050 --target_cpa_min 120 --target_cpa_max 120 --advertiser_id 12 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_12_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure  --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 401_ \
    --budget_min 2900 --budget_max 2900 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 13 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_13_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 432_ \
    --budget_min 4750 --budget_max 4750 --target_cpa_min 60 --target_cpa_max 60 --advertiser_id 14 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_14_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 402_ \
    --budget_min 3400 --budget_max 3400 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 15 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_15_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 403_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 130 --target_cpa_max 130 --advertiser_id 16 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_16_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 404_ \
    --budget_min 3500 --budget_max 3500 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 17 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_17_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 405_ \
    --budget_min 2200 --budget_max 2200 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 18 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_18_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 433_ \
    --budget_min 2700 --budget_max 2700 --target_cpa_min 100 --target_cpa_max 100 --advertiser_id 19 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_19_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 306_ \
    --budget_min 3100 --budget_max 3100 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 20 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_20_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 406_ \
    --budget_min 2100 --budget_max 2100 --target_cpa_min 120 --target_cpa_max 120 --advertiser_id 21 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_21_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids 

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 434_ \
    --budget_min 4850 --budget_max 4850 --target_cpa_min 60 --target_cpa_max 60 --advertiser_id 22 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_22_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 407_ \
    --budget_min 4100 --budget_max 4100 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 23 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_23_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 408_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 120 --target_cpa_max 120 --advertiser_id 24 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_24_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids 

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 435_ \
    --budget_min 4800 --budget_max 4800 --target_cpa_min 60 --target_cpa_max 60 --advertiser_id 25 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_25_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 409_ \
    --budget_min 3050 --budget_max 3050 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 26 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_26_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 436_ \
    --budget_min 4250 --budget_max 4250 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 27 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_27_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 410_ \
    --budget_min 2850 --budget_max 2850 --target_cpa_min 100 --target_cpa_max 100 --advertiser_id 28 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_28_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 437_ \
    --budget_min 2250 --budget_max 2250 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 29 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_29_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 438_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 130 --target_cpa_max 130 --advertiser_id 30 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_30_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids 

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 411_ \
    --budget_min 3900 --budget_max 3900 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 31 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_31_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 412_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 120 --target_cpa_max 120 --advertiser_id 32 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_32_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids        

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 413_ \
    --budget_min 2350 --budget_max 2350 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 33 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_33_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids 

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 414_ \
    --budget_min 4450 --budget_max 4450 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 34 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_34_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 439_ \
    --budget_min 3550 --budget_max 3550 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 35 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_35_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 125_ \
    --budget_min 2700 --budget_max 2700 --target_cpa_min 100 --target_cpa_max 100 --advertiser_id 36 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_36_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids 

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 126_ \
    --budget_min 2100 --budget_max 2100 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 37 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_37_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 127_ \
    --budget_min 4650 --budget_max 4650 --target_cpa_min 60 --target_cpa_max 60 --advertiser_id 38 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_38_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids  

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 128_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 130 --target_cpa_max 130 --advertiser_id 39 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_39_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids 

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 415_ \
    --budget_min 3400 --budget_max 3400 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 40 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_40_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 130_ \
    --budget_min 2650 --budget_max 2650 --target_cpa_min 100 --target_cpa_max 100 --advertiser_id 41 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_41_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids     

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 416_ \
    --budget_min 2300 --budget_max 2300 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 42 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_42_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 417_ \
    --budget_min 4100 --budget_max 4100 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 43 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_43_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 418_ \
    --budget_min 4800 --budget_max 4800 --target_cpa_min 60 --target_cpa_max 60 --advertiser_id 44 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_44_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 134_ \
    --budget_min 4450 --budget_max 4450 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 45 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_45_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids 

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 419_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 130 --target_cpa_max 130 --advertiser_id 46 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_46_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 136_ \
    --budget_min 2050 --budget_max 2050 --target_cpa_min 120 --target_cpa_max 120 --advertiser_id 47 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_47_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids                      








python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 500_ \
    --budget_min 4300 --budget_max 4300 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 1 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_1_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 21660000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 501_ \
    --budget_min 3000 --budget_max 3000 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 2 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_2_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 10850000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 502_ \
    --budget_min 3500 --budget_max 3500 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 7 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_7_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
                    --checkpoint_num 4600000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 503_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 130 --target_cpa_max 130 --advertiser_id 9 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_9_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 10850000 --stochastic_exposure  --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 504_ \
    --budget_min 2350 --budget_max 2350 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 11 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_11_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 20390000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 505_ \
    --budget_min 2050 --budget_max 2050 --target_cpa_min 120 --target_cpa_max 120 --advertiser_id 12 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_12_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 21660000 --stochastic_exposure  --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 506_ \
    --budget_min 3400 --budget_max 3400 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 15 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_15_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 3270000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 507_ \
    --budget_min 2100 --budget_max 2100 --target_cpa_min 120 --target_cpa_max 120 --advertiser_id 21 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_21_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 16868520 --stochastic_exposure --exclude_self_bids 
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 508_ \
    --budget_min 4100 --budget_max 4100 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 23 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_23_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 15938892 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 509_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 120 --target_cpa_max 120 --advertiser_id 24 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_24_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 16868520 --stochastic_exposure --exclude_self_bids
            
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 510_ \
    --budget_min 3050 --budget_max 3050 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 26 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_26_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 21660000 --stochastic_exposure  --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 511_ \
    --budget_min 2850 --budget_max 2850 --target_cpa_min 100 --target_cpa_max 100 --advertiser_id 28 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_28_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 14439492 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 512_ \
    --budget_min 3900 --budget_max 3900 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 31 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_31_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 3270000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 513_ \
    --budget_min 2350 --budget_max 2350 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 33 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_33_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 21660000 --stochastic_exposure  --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 514_ \
    --budget_min 4450 --budget_max 4450 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 34 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_34_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 15938892 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 515_ \
    --budget_min 3400 --budget_max 3400 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 40 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_40_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/084_onbc_seed_42_resume_053_expert_competitors \
                    --checkpoint_num 14290000 --stochastic_exposure --exclude_self_bids    

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 516_ \
    --budget_min 2300 --budget_max 2300 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 42 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_42_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 3270000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 517_ \
    --budget_min 4100 --budget_max 4100 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 43 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_43_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 17078436 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 518_ \
    --budget_min 4800 --budget_max 4800 --target_cpa_min 60 --target_cpa_max 60 --advertiser_id 44 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_44_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 20390000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 519_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 130 --target_cpa_max 130 --advertiser_id 46 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_46_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
                    --checkpoint_num 4600000 --stochastic_exposure --exclude_self_bids








python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 600_ \
    --budget_min 2100 --budget_max 2100 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 37 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_37_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 15938892 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 601_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 130 --target_cpa_max 130 --advertiser_id 39 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_39_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 14439492 --stochastic_exposure --exclude_self_bids 

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 602_ \
    --budget_min 3400 --budget_max 3400 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 40 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_40_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/084_onbc_seed_42_resume_053_expert_competitors \
                    --checkpoint_num 14290000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 603_ \
    --budget_min 2650 --budget_max 2650 --target_cpa_min 100 --target_cpa_max 100 --advertiser_id 41 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_41_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/084_onbc_seed_42_resume_053_expert_competitors \
                    --checkpoint_num 14290000 --stochastic_exposure --exclude_self_bids     

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 604_ \
    --budget_min 4450 --budget_max 4450 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 45 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_45_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 15938892 --stochastic_exposure --exclude_self_bids 
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 605_ \
    --budget_min 2050 --budget_max 2050 --target_cpa_min 120 --target_cpa_max 120 --advertiser_id 47 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_47_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/084_onbc_seed_42_resume_053_expert_competitors \
                    --checkpoint_num 14290000 --stochastic_exposure --exclude_self_bids                      

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 606_ \
    --budget_min 2400 --budget_max 2400 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 3 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_3_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 21370000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 607_ \
    --budget_min 4800 --budget_max 4800 --target_cpa_min 60 --target_cpa_max 60 --advertiser_id 4 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_4_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/075_onbc_seed_0_new_data_realistic_60_obs_resume_055 \
                    --checkpoint_num 16810000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 608_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 130 --target_cpa_max 130 --advertiser_id 5 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_5_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 3270000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 609_ \
    --budget_min 4600 --budget_max 4600 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 8 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_8_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 14439492 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 610_ \
    --budget_min 2800 --budget_max 2800 --target_cpa_min 100 --target_cpa_max 100 --advertiser_id 10 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_10_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
                    --checkpoint_num 4600000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 611_ \
    --budget_min 4750 --budget_max 4750 --target_cpa_min 60 --target_cpa_max 60 --advertiser_id 14 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_14_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 15938892 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 612_ \
    --budget_min 3500 --budget_max 3500 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 17 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_17_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 15938892 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 613_ \
    --budget_min 2200 --budget_max 2200 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 18 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_18_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 21370000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 614_ \
    --budget_min 2700 --budget_max 2700 --target_cpa_min 100 --target_cpa_max 100 --advertiser_id 19 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_19_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13990000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 615_ \
    --budget_min 3100 --budget_max 3100 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 20 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_20_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 20390000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 616_ \
    --budget_min 4850 --budget_max 4850 --target_cpa_min 60 --target_cpa_max 60 --advertiser_id 22 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_22_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13990000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 617_ \
    --budget_min 4800 --budget_max 4800 --target_cpa_min 60 --target_cpa_max 60 --advertiser_id 25 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_25_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 15938892 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 618_ \
    --budget_min 4250 --budget_max 4250 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 27 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_27_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 14439492 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 619_ \
    --budget_min 2250 --budget_max 2250 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 29 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_29_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/075_onbc_seed_0_new_data_realistic_60_obs_resume_055 \
                    --checkpoint_num 17185000 --stochastic_exposure --exclude_self_bids



python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 700_ \
    --budget_min 4300 --budget_max 4300 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 1 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_1_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 21660000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 701_ \
    --budget_min 3000 --budget_max 3000 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 2 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_2_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 10850000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 702_ \
    --budget_min 3500 --budget_max 3500 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 7 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_7_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
                    --checkpoint_num 4600000 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 703_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 130 --target_cpa_max 130 --advertiser_id 9 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_9_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 10850000 --stochastic_exposure  --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 704_ \
    --budget_min 2350 --budget_max 2350 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 11 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_11_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 20390000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 705_ \
    --budget_min 2050 --budget_max 2050 --target_cpa_min 120 --target_cpa_max 120 --advertiser_id 12 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_12_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 21660000 --stochastic_exposure  --exclude_self_bids

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 706_ \
    --budget_min 3400 --budget_max 3400 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 15 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_15_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 3270000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 707_ \
    --budget_min 2100 --budget_max 2100 --target_cpa_min 120 --target_cpa_max 120 --advertiser_id 21 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_21_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 16868520 --stochastic_exposure --exclude_self_bids 
                    
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 708_ \
    --budget_min 4100 --budget_max 4100 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 23 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_23_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 15938892 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 709_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 120 --target_cpa_max 120 --advertiser_id 24 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_24_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 16868520 --stochastic_exposure --exclude_self_bids
            
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 710_ \
    --budget_min 3050 --budget_max 3050 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 26 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_26_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 21660000 --stochastic_exposure  --exclude_self_bids

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 711_ \
    --budget_min 2850 --budget_max 2850 --target_cpa_min 100 --target_cpa_max 100 --advertiser_id 28 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_28_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 14439492 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 712_ \
    --budget_min 3900 --budget_max 3900 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 31 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_31_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 3270000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 713_ \
    --budget_min 2350 --budget_max 2350 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 33 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_33_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 21660000 --stochastic_exposure  --exclude_self_bids

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 714_ \
    --budget_min 4450 --budget_max 4450 --target_cpa_min 70 --target_cpa_max 70 --advertiser_id 34 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_34_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 15938892 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 715_ \
    --budget_min 3400 --budget_max 3400 --target_cpa_min 90 --target_cpa_max 90 --advertiser_id 40 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_40_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/084_onbc_seed_42_resume_053_expert_competitors \
                    --checkpoint_num 14290000 --stochastic_exposure --exclude_self_bids    

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 716_ \
    --budget_min 2300 --budget_max 2300 --target_cpa_min 110 --target_cpa_max 110 --advertiser_id 42 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_42_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 3270000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 717_ \
    --budget_min 4100 --budget_max 4100 --target_cpa_min 80 --target_cpa_max 80 --advertiser_id 43 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_43_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/068_onbc_seed_42_new_data_realistic_60_obs_resume_053_with_period_27 \
                    --checkpoint_num 17078436 --stochastic_exposure --exclude_self_bids
                    
python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 718_ \
    --budget_min 4800 --budget_max 4800 --target_cpa_min 60 --target_cpa_max 60 --advertiser_id 44 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_44_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/080_onbc_seed_42_resume_076_stoch_exp \
                    --checkpoint_num 20390000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 20 --batch_size 512 --num_steps 20_000_000 --out_prefix 719_ \
    --budget_min 2000 --budget_max 2000 --target_cpa_min 130 --target_cpa_max 130 --advertiser_id 46 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_46_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/026_onbc_seed_0_new_data_realistic_60_obs_resume_023 \
                    --checkpoint_num 4600000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 720_ \
    --budget_min 2900 --budget_max 2900 --target_cpa_min 100 --target_cpa_max 100 --advertiser_id 0 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_0_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-5 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13990000 --stochastic_exposure --exclude_self_bids

python online/main_train_onbc.py --num_envs 21 --batch_size 512 --num_steps 20_000_000 --out_prefix 721_ \
    --budget_min 2650 --budget_max 2650 --target_cpa_min 100 --target_cpa_max 100 --advertiser_id 41 \
        --new_action --exp_action --out_suffix=_resume_053_advertiser_41_exclude_self --seed 42\
            --obs_type obs_60_keys --learning_rate 1e-6 --save_every 5000 --num_layers 3 \
                --load_path output/training/ongoing/053_onbc_seed_0_new_data_realistic_60_obs_resume_050 \
                    --checkpoint_num 13170000 --stochastic_exposure --exclude_self_bids  