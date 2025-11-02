from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from .utils import continuous_segments, sliding_window_indices


@dataclass
class SplitIndices:
    train: List[Tuple[int, int, int]]  # (group_idx, seg_start, win_start_in_seg)
    val: List[Tuple[int, int, int]]
    test: List[Tuple[int, int, int]]


class WindSeqIndexer:
    def __init__(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        feature_cols: List[str],
        target_col: str,
        hist_len: int,
        pred_len: int,
        stride: int = 1,
        freq: str = "1H",
    ):
        self.df = df.copy()
        self.id_col = id_col
        self.time_col = time_col
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.stride = stride
        self.freq = freq

        self.groups = []
        self.group_offsets = []  # offsets in concatenated arrays
        self._prepare_groups()

    def _prepare_groups(self):
        self.id2idx = {}
        next_idx = 0
        for gid, gdf in self.df.groupby(self.id_col):
            if gid not in self.id2idx:
                self.id2idx[gid] = next_idx
                next_idx += 1
            gdf = gdf.sort_values(self.time_col).reset_index(drop=True)
            times = gdf[self.time_col]
            segs = continuous_segments(times, self.freq)
            self.groups.append(
                {
                    "id": gid,
                    "id_idx": self.id2idx[gid],
                    "df": gdf,
                    "segs": segs,
                }
            )
        self.n_turbines = len(self.id2idx)

    def split_811(self, shuffle: bool = False, seed: Optional[int] = None) -> SplitIndices:
        """
        Build all continuous windows per turbine first, then split by sample counts (8:1:1).
        Optionally shuffle windows before splitting to randomize samples.
        Each window lies fully within a continuous segment.
        """
        train, val, test = [], [], []
        rng = np.random.default_rng(seed) if shuffle else None
        for g_idx, g in enumerate(self.groups):
            # Collect all windows for this turbine
            windows = []
            for (s, e) in g["segs"]:
                seg_len = e - s
                starts = sliding_window_indices(seg_len, self.hist_len, self.pred_len, self.stride)
                for w in starts:
                    windows.append((g_idx, s, w))

            n = len(windows)
            if n == 0:
                continue

            # Optional shuffle before 8:1:1 split
            if shuffle:
                rng.shuffle(windows)

            # 8:1:1 by sample count
            t_end = max(1, int(n * 0.8))
            v_end = max(t_end, int(n * 0.9))
            train.extend(windows[:t_end])
            val.extend(windows[t_end:v_end])
            test.extend(windows[v_end:])

        return SplitIndices(train=train, val=val, test=test)


class WindSeqDataset(Dataset):
    def __init__(
        self,
        groups: List[Dict],
        indices: List[Tuple[int, int, int]],
        feature_cols: List[str],
        target_col: str,
        hist_len: int,
        pred_len: int,
        exo_cols: List[str],
        fit_scalers: bool = False,
        feature_scaler: Optional[StandardScaler] = None,
        target_scaler: Optional[StandardScaler] = None,
        exo_scaler: Optional[StandardScaler] = None,
    ):
        self.groups = groups
        self.indices = indices
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.exo_cols = exo_cols

        self.feature_scaler = feature_scaler or StandardScaler()
        self.target_scaler = target_scaler or StandardScaler()
        self.exo_scaler = exo_scaler or StandardScaler()
        self._fitted = False

        if fit_scalers:
            self._fit_scalers()
        # 预计算每个 group 的标准化视图，避免 __getitem__ 每次 transform
        self._prepare_scaled_views()

    def _fit_scalers(self):
        Xs = []
        ys = []
        exos = []
        for g_idx, seg_start, win_start in self.indices:
            g = self.groups[g_idx]
            df = g["df"]
            s = seg_start + win_start
            he = s + self.hist_len
            pe = he + self.pred_len
            X = df.iloc[s:he][self.feature_cols].values.astype(np.float32)
            y = df.iloc[he:pe][self.target_col].values.astype(np.float32).reshape(-1, 1)
            X_future = df.iloc[he:pe][self.exo_cols].values.astype(np.float32)
            Xs.append(X)
            ys.append(y)
            exos.append(X_future)
        if Xs:
            Xc = np.concatenate(Xs, axis=0)
            yc = np.concatenate(ys, axis=0)
            exoc = np.concatenate(exos, axis=0)
            self.feature_scaler.fit(Xc)
            self.target_scaler.fit(yc)
            self.exo_scaler.fit(exoc)
            self._fitted = True

    def _prepare_scaled_views(self):
        """
        预计算并缓存每个 group 的标准化特征、目标和未来 exo，避免在 __getitem__ 里重复调用 transform。
        """
        for g in self.groups:
            df = g["df"]
            feats = df[self.feature_cols].values.astype(np.float32)
            exos = df[self.exo_cols].values.astype(np.float32)
            tgt = df[[self.target_col]].values.astype(np.float32)
            g["_feat_scaled"] = self.feature_scaler.transform(feats)
            g["_exo_scaled"] = self.exo_scaler.transform(exos)
            g["_tgt_scaled"] = self.target_scaler.transform(tgt)
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        g_idx, seg_start, win_start = self.indices[idx]
        g = self.groups[g_idx]
        df = g["df"]
        s = seg_start + win_start
        he = s + self.hist_len
        pe = he + self.pred_len

        # 使用预计算的标准化视图，避免每个样本重复 transform
        X_hist = g["_feat_scaled"][s:he]                                   # [hist, feat]
        y = df.iloc[he:pe][self.target_col].values.astype(np.float32).reshape(-1, 1)  # 原始单位
        X_future_scaled = g["_exo_scaled"][he:pe]                          # [pred, exo_dim]
        y_scaled = g["_tgt_scaled"][he:pe]                                 # [pred, 1]
        y0_scaled = g["_tgt_scaled"][he - 1].reshape(-1)                   # [1]

        sample = {
            "X": torch.from_numpy(X_hist),                 # [hist, feat]
            "X_future": torch.from_numpy(X_future_scaled), # [pred, exo_dim]
            "y_scaled": torch.from_numpy(y_scaled),        # [pred, 1]
            "y": torch.from_numpy(y),                      # [pred, 1]
            "y0_scaled": torch.from_numpy(y0_scaled),      # [1]
            "turb_id": g["id"],
            "turb_idx": torch.tensor(g["id_idx"], dtype=torch.long),
            "start_idx": s,
        }
        return sample