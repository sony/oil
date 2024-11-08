# Append root directory to path
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
import pandas as pd
import numpy as np
from definitions import ROOT_DIR


def reorder_list_of_lists(lst, positions):
    array_of_lists = np.array(lst)
    reordered_array = np.take_along_axis(array_of_lists, positions, axis=1)
    # We need to convert to nested lists because parquet does not support 2d arrays
    reordered_lists = reordered_array.tolist()
    return reordered_lists


def generate_pvalue_df(data):
    pvalues_df = data.groupby(
        [
            "deliveryPeriodIndex",
            "timeStepIndex",
            "advertiserNumber",
            "advertiserCategoryIndex",
        ]
    ).agg(
        {
            "pValue": lambda x: x.tolist(),
            "pValueSigma": lambda x: x.tolist(),
        }
    )
    pvalues_df.reset_index(inplace=True)
    return pvalues_df


def generate_bids_df(data):
    bids_df = (
        data[data["xi"] == 1]
        .groupby(["deliveryPeriodIndex", "timeStepIndex", "pvIndex"])
        .agg(
            {
                "bid": lambda x: x.tolist(),
                "isExposed": lambda x: x.tolist(),
                "cost": lambda x: x.tolist(),
                "advertiserNumber": lambda x: x.tolist(),
                "conversionAction": lambda x: x.tolist(),
            }
        )
    )
    bids_df.reset_index(inplace=True)
    bids_df = bids_df.groupby(["deliveryPeriodIndex", "timeStepIndex"]).agg(
        {
            "bid": lambda x: x.tolist(),
            "isExposed": lambda x: x.tolist(),
            "cost": lambda x: x.tolist(),
            "advertiserNumber": lambda x: x.tolist(),
            "conversionAction": lambda x: x.tolist(),
        }
    )
    bids_df.reset_index(inplace=True)

    # Sort bid, isExposed, cost, advertiserNumber according to bid
    bids_df["positions"] = bids_df.apply(lambda x: np.argsort(x.bid), axis=1)
    bids_df["bid"] = bids_df.apply(
        lambda x: reorder_list_of_lists(x.bid, x.positions), axis=1
    )
    bids_df["isExposed"] = bids_df.apply(
        lambda x: reorder_list_of_lists(x.isExposed, x.positions), axis=1
    )
    bids_df["cost"] = bids_df.apply(
        lambda x: reorder_list_of_lists(x.cost, x.positions), axis=1
    )
    bids_df["advertiserNumber"] = bids_df.apply(
        lambda x: reorder_list_of_lists(x.advertiserNumber, x.positions), axis=1
    )
    bids_df["conversionAction"] = bids_df.apply(
        lambda x: reorder_list_of_lists(x.conversionAction, x.positions), axis=1
    )
    bids_df.drop(columns=["positions"], inplace=True)
    return bids_df


def generate_online_rl_data(traffic_data_paths, out_dir, use_precomputed=False):
    """
    Generate online rl data for training.
    """

    if use_precomputed:
        for data_type in ["bids"]:
            print(f"Generating {data_type} data")
            data_list = []
            for traffic_data_path in traffic_data_paths:
                file_name = traffic_data_path.stem
                file_path = out_dir / f"{file_name}_{data_type}.parquet"
                print("Reading data from", file_path)
                data = pd.read_parquet(file_path)
                data_list.append(data)
            data = pd.concat(data_list)
            data.to_parquet(out_dir / f"{data_type}.parquet")
    else:
        pvalues_df_list = []
        bids_df_list = []

        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        for traffic_data_path in traffic_data_paths:
            print("Reading data from", traffic_data_path)
            data = pd.read_parquet(traffic_data_path)
            file_name = traffic_data_path.stem

            print("Generating online rl data for", file_name)
            pvalues_df = generate_pvalue_df(data)
            pvalues_file_name = f"{file_name}_pvalues.parquet"
            pvalues_df_list.append(pvalues_df)
            pvalues_df.to_parquet(out_dir / pvalues_file_name)

            bids_df = generate_bids_df(data)
            bids_file_name = f"{file_name}_bids.parquet"
            bids_df_list.append(bids_df)
            bids_df.to_parquet(out_dir / bids_file_name)


if __name__ == "__main__":
    periods = list(range(7, 28))
    data_dir = ROOT_DIR / "data" / "raw_traffic_final_parquet"
    out_dir = ROOT_DIR / "data" / "online_rl_data_final_with_ad_idx"

    traffic_data_paths = [data_dir / f"period-{period}.parquet" for period in periods]
    use_precomputed = False
    generate_online_rl_data(
        traffic_data_paths, out_dir, use_precomputed=use_precomputed
    )
