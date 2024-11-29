import os
import pandas as pd


class OnlineLp:
    """
    OnlineLp model
    """

    def __init__(self, dataPath, dataset, seed=0):
        self.dataPath = dataPath
        self.dataset = dataset
        self.seed=seed

    def train(self, save_path):
        
        save_path.mkdir(parents=True, exist_ok=True)
        file_names = [f"period-{i}.parquet" for i in range(7, 27)]
        print(file_names)
        df_list = []
        for name in file_names:
            file_path = self.dataPath / name
            print("Start: ", file_path)
            df = pd.read_parquet(file_path)
            episode_df = self.onlinelp_for_specific_episode(df)
            df_list.append(episode_df)
            print("End: ", file_path)
        full_df = pd.concat(df_list).reset_index(drop=True)
        full_df.to_csv(f"{save_path}/data.csv")

    def onlinelp_for_specific_episode(self, df):
        df_filter = df[(df["pValue"] > 0) & (df["leastWinningCost"] > 0.0001)]
        grouped_df = df_filter.groupby("advertiserCategoryIndex")
        num_tick = 48
        if self.dataset == "final":
            max_budget = 6000
        elif self.dataset == "official":
            max_budget = 12000
        else:
            raise ValueError("Invalid dataset name")
        interval = 10
        result_dfs = []

        for category, group in grouped_df:
            print("category:", category)
            sampled_group = group.sample(frac=1 / 150, random_state=self.seed)
            sampled_group["realCPA"] = sampled_group["leastWinningCost"] / (
                sampled_group["pValue"] + 0.0001
            )
            for timestep in range(num_tick):
                timestep_filtered = sampled_group[
                    sampled_group["timeStepIndex"] >= timestep
                ]
                timestep_filtered = timestep_filtered.sort_values(
                    by="realCPA"
                ).reset_index(drop=True)
                column_list = [
                    "deliveryPeriodIndex",
                    "advertiserCategoryIndex",
                    "realCPA",
                    "cum_cost",
                ]
                timestep_filtered["cum_cost"] = timestep_filtered[
                    "leastWinningCost"
                ].cumsum()
                timestep_filtered = timestep_filtered[column_list]
                timestep_filtered["timeStepIndex"] = timestep
                filtered_df = timestep_filtered[
                    timestep_filtered["cum_cost"] < max_budget
                ]
                last_selected = 0
                result = []

                for index, row in filtered_df.iterrows():
                    if row["cum_cost"] - last_selected >= interval:
                        result.append(row)
                        last_selected = row["cum_cost"]
                    elif index == len(filtered_df) - 1:
                        result.append(row)
                        last_selected = row["cum_cost"]

                final_df = pd.DataFrame(result)
                result_dfs.append(final_df)

        final_result_df = pd.concat(result_dfs).reset_index(drop=True)
        return final_result_df
