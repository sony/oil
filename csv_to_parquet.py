import pathlib
import pandas as pd
from definitions import ROOT_DIR


# Convert all traffic data to parquet
first_period = 7
last_period = 27
for dataset in ["dense", "sparse"]:
    for period in range(first_period, last_period + 1):
        data_path = ROOT_DIR / "data" / "traffic" / f"raw_traffic_{dataset}" / f"period-{period}.csv"
        print(f"Loading {data_path}")
        df = pd.read_csv(data_path, dtype="float32")
        pathlib.Path(ROOT_DIR / "data" / "traffic" / f"raw_traffic_{dataset}_parquet").mkdir(
            parents=True, exist_ok=True
        )
        out_path = (
            ROOT_DIR / "data" / "traffic" / f"raw_traffic_{dataset}_parquet" / f"period-{period}.parquet"
        )
        print(f"Saving to {out_path}")
        df.to_parquet(out_path)
