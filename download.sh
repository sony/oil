wget https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_7-8.zip -P ./data/raw_traffic_final/
wget https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_9-10.zip -P ./data/raw_traffic_final/
wget https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_11-12.zip -P ./data/raw_traffic_final/
wget https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_13.zip -P ./data/raw_traffic_final/
wget https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_14-15.zip -P ./data/raw_traffic_final/
wget https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_16-17.zip -P ./data/raw_traffic_final/
wget https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_18-19.zip -P ./data/raw_traffic_final/
wget https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_20-21.zip -P ./data/raw_traffic_final/
wget https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_22-23.zip -P ./data/raw_traffic_final/
wget https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_24-25.zip -P ./data/raw_traffic_final/
wget https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_general_track_final_data_period_26-27.zip -P ./data/raw_traffic_final/

for file in ./data/raw_traffic_final/*.zip; do unzip "$file" -d "${file%.zip}"; done
