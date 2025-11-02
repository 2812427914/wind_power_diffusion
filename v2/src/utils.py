import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Required column not found. Candidates: {candidates}, available: {list(df.columns)}")
    return None


def detect_time_column(df: pd.DataFrame) -> str:
    candidates = [
        "time", "timestamp", "tmstamp", "datetime", "date", "ds"
    ]
    col = find_column(df, candidates, required=False)
    if col is not None:
        return col
    # fallback: first datetime-like column
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    raise ValueError("No datetime-like column found.")


def ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    return df


def compute_time_features(df: pd.DataFrame, time_col: str, add_if_missing: bool = True) -> pd.DataFrame:
    df = df.copy()
    if add_if_missing or "hour" not in df.columns:
        df["hour"] = df[time_col].dt.hour
    if add_if_missing or "dayofweek" not in df.columns:
        df["dayofweek"] = df[time_col].dt.dayofweek
    return df


def encode_time_features(df: pd.DataFrame, time_col: str, mode: str = "raw") -> pd.DataFrame:
    df = df.copy()
    # Ensure the time_col is datetime
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["hour"] = df[time_col].dt.hour
    df["dayofweek"] = df[time_col].dt.dayofweek
    if mode.lower() == "sin-cos":
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    return df

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))))


def continuous_segments(times: pd.Series, freq: str = "1h") -> List[Tuple[int, int]]:
    """
    Given a sorted datetime series (per turbine), return list of (start_idx, end_idx_exclusive)
    for maximal continuous segments with step == freq.
    """
    if len(times) == 0:
        return []
    diffs = times.diff().fillna(pd.Timedelta(0))
    # 统一用小写 h
    step = pd.to_timedelta(freq.lower())
    segs = []
    start = 0
    for i in range(1, len(times)):
        if diffs.iloc[i] != step:
            segs.append((start, i))
            start = i
    segs.append((start, len(times)))
    return segs


def sliding_window_indices(seg_len: int, hist: int, pred: int, stride: int) -> List[int]:
    starts = []
    end_limit = seg_len - (hist + pred) + 1
    i = 0
    while i < end_limit:
        starts.append(i)
        i += stride
    return starts


def save_splits(path: str, splits, groups: List[dict]) -> None:
    """
    Save splits to JSON using turbine ID for robustness:
    Each entry: {"turb_id": <id>, "seg_start": int, "win_start": int}
    """
    def serialize(indices: List[Tuple[int, int, int]]):
        result = []
        for g_idx, s, w in indices:
            turb_id = groups[g_idx]["id"]
            result.append({
                "turb_id": str(turb_id),
                "seg_start": int(s),
                "win_start": int(w),
            })
        return result

    obj = {
        "train": serialize(splits.train),
        "val": serialize(splits.val),
        "test": serialize(splits.test),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_json(obj, path)


def load_splits(path: str, groups: List[dict]):
    """
    Load splits from JSON and map turb_id back to current group indices.
    Returns an object with attributes: train, val, test (lists of tuples).
    """
    data = load_json(path)
    id2gidx = {str(g["id"]): i for i, g in enumerate(groups)}

    def deserialize(arr):
        out = []
        for item in arr:
            tid = str(item["turb_id"])
            if tid in id2gidx:
                out.append((id2gidx[tid], int(item["seg_start"]), int(item["win_start"])))
        return out

    class _SplitsObj:
        def __init__(self, train, val, test):
            self.train = train
            self.val = val
            self.test = test

    return _SplitsObj(
        train=deserialize(data.get("train", [])),
        val=deserialize(data.get("val", [])),
        test=deserialize(data.get("test", [])),
    )