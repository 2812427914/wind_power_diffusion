import argparse
import os
from typing import Tuple

import pandas as pd

from .utils import detect_time_column, ensure_datetime, find_column, compute_time_features, encode_time_features


def load_and_clean(
    csv_path: str,
    id_col: str = None,
    time_col: str = None,
    wind_col: str = None,
    power_col: str = None,
    out_csv_path: str = None,
    time_encode: str = "raw",
) -> Tuple[pd.DataFrame, str, str, str, str]:
    """
    Load raw CSV and apply cleaning:
    1) Drop rows with power == 0
    2) Drop rows with wind speed == 0
    3) Optional time feature encoding: 'raw' (hour/dayofweek) or 'sin-cos'
    """
    df = pd.read_csv(csv_path)
    # 强制使用示例数据的列名
    id_col = "TurbID"
    time_col = "Datetime"
    wind_col = "Wspd"
    power_col = "Patv"

    # ensure datetime and sort
    df = ensure_datetime(df, time_col)
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    # cleaning filters
    before = len(df)
    df = df[(df[power_col] != 0) & (df[power_col].notna())]
    df = df[(df[wind_col] != 0) & (df[wind_col].notna())]
    # abnormal ratio: wind <= power/50 -> inconsistent (too high power for given wind)
    # df = df[~(df[wind_col] <= (df[power_col] / 50.0))]

    # 时间特征编码
    df = encode_time_features(df, time_col, mode=time_encode)
    after = len(df)

    if out_csv_path:
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        df.to_csv(out_csv_path, index=False)

    print(f"Cleaned data: {before} -> {after} rows. Saved: {out_csv_path}")
    return df, id_col, time_col, wind_col, power_col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to raw CSV (hourly).")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--wind-col", default=None)
    parser.add_argument("--power-col", default=None)
    parser.add_argument("--time-encode", choices=["raw", "sin-cos"], default="raw", help="Time feature encoding mode.")
    parser.add_argument("--out", default="v2/results/cleaned.csv", help="Output cleaned CSV path.")
    args = parser.parse_args()

    load_and_clean(
        csv_path=args.data,
        id_col=args.id_col,
        time_col=args.time_col,
        wind_col=args.wind_col,
        power_col=args.power_col,
        out_csv_path=args.out,
        time_encode=args.time_encode,
    )


if __name__ == "__main__":
    main()