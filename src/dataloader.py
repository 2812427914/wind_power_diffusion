import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/wtbdata_hourly.csv')

def load_data(path=DATA_PATH, max_samples=None, chunksize=None):
    # 保留原有实现，兼容旧代码
    if chunksize is not None:
        dfs = []
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunk = chunk.dropna(how='all')
            chunk = chunk.fillna(method='ffill').fillna(method='bfill')
            dfs.append(chunk)
            if max_samples is not None and sum(len(df) for df in dfs) >= max_samples:
                break
        df = pd.concat(dfs, ignore_index=True)
        if max_samples is not None:
            df = df.head(max_samples)
    else:
        df = pd.read_csv(path)
        df = df.dropna(how='all')
        df = df.fillna(method='ffill').fillna(method='bfill')
        if max_samples is not None:
            df = df.head(max_samples)
    return df

class WindPowerCSVDataset(Dataset):
    """
    懒加载风电数据集，支持单点和序列切片。
    支持 feature_cols、target_col 指定，支持 seq_mode。
    支持多风机独立切片。
    """
    def __init__(self, csv_path, feature_cols=None, target_col='Patv', transform=None, hist_len=1, seq_len=1, seq_mode=False, wind_id_col='TurbID'):
        self.csv_path = csv_path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.transform = transform
        self.hist_len = hist_len
        self.seq_len = seq_len
        self.seq_mode = seq_mode
        self.wind_id_col = wind_id_col
        # 预读取 header，获取列名
        with open(csv_path, 'r') as f:
            self.header = f.readline().strip().split(',')
        # 默认特征列
        if self.feature_cols is None:
            self.feature_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv']
        # 列索引
        self.feature_idx = [self.header.index(col) for col in self.feature_cols]
        self.target_idx = self.header.index(self.target_col)
        self.wind_id_idx = self.header.index(self.wind_id_col)
        # 预处理所有风机的起止行号
        self.slice_index = []  # [(wind_id, start_line, end_line)]
        wind_id_to_lines = {}
        with open(csv_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                row = line.strip().split(',')
                wind_id = row[self.wind_id_idx]
                if wind_id not in wind_id_to_lines:
                    wind_id_to_lines[wind_id] = []
                wind_id_to_lines[wind_id].append(i)
        # 生成所有风机的切片索引
        self.seq_indices = []  # [(wind_id, start_line_idx)]
        for wind_id, lines in wind_id_to_lines.items():
            n = len(lines)
            for start in range(0, n - self.hist_len - self.seq_len + 1, 24):
                self.seq_indices.append((wind_id, lines[start]))
        self.n_samples = len(self.seq_indices) if self.seq_mode else sum(len(lines) for lines in wind_id_to_lines.values())
        self.wind_id_to_lines = wind_id_to_lines

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if not self.seq_mode:
            # 单点模式
            # 只支持全量遍历
            all_lines = []
            with open(self.csv_path, 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    all_lines.append(line)
            row = all_lines[idx].strip().split(',')
            x = np.array([float(row[j]) if row[j] != '' else 0.0 for j in self.feature_idx], dtype=np.float32)
            y = float(row[self.target_idx]) if row[self.target_idx] != '' else 0.0
            if self.transform:
                x, y = self.transform(x, y)
            return x, y
        else:
            # 序列模式：每台风机独立切片
            wind_id, start_line_idx = self.seq_indices[idx]
            # 找到该风机的所有行号
            lines = self.wind_id_to_lines[wind_id]
            # 找到start_line_idx在lines中的位置
            start_pos = lines.index(start_line_idx)
            # 取hist_len+seq_len行
            x_seq = []
            y_seq = []
            with open(self.csv_path, 'r') as f:
                for i, line in enumerate(f):
                    if i in lines[start_pos : start_pos + self.hist_len]:
                        row = line.strip().split(',')
                        x_seq.append([float(row[j]) if row[j] != '' else 0.0 for j in self.feature_idx])
                    if i in lines[start_pos + self.hist_len : start_pos + self.hist_len + self.seq_len]:
                        row = line.strip().split(',')
                        y_seq.append(float(row[self.target_idx]) if row[self.target_idx] != '' else 0.0)
                    if i > lines[start_pos + self.hist_len + self.seq_len - 1]:
                        break
            x_seq = np.array(x_seq, dtype=np.float32)  # (hist_len, feature_dim)
            y_seq = np.array(y_seq, dtype=np.float32)  # (seq_len,)
            if self.transform:
                x_seq, y_seq = self.transform(x_seq, y_seq)
            return x_seq, y_seq

def preprocess_features(df,
                        feature_cols=None,
                        target_col='Patv',
                        seq_len=24,
                        hist_len=24,
                        seq_mode=False,
                        turb_id_col='TurbID'):
    """
    seq_mode: False（兼容原有），X shape (N, feature_dim); True，X shape (N, hist_len, feature_dim)
    turb_id_col: 风机ID字段，支持分风机独立切片
    """
    if feature_cols is None:
        feature_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv']

    # ---------- 序列模式（seq_vae / seq_diffusion / seq_ar_diffusion） ----------
    if seq_mode:
        X = []
        y = []
        for i in range(0, len(df) - hist_len - seq_len + 1, 24):
            X.append(df[feature_cols].iloc[i:i+hist_len].values)  # shape (hist_len, feature_dim)
            y.append(df[target_col].iloc[i+hist_len:i+hist_len+seq_len].values)  # shape (seq_len,)
        X = np.array(X)  # (N, hist_len, feature_dim)
        y = np.array(y)  # (N, seq_len)
        # 分别归一化
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_2d = X.reshape(-1, X.shape[-1])
        X_scaled_2d = scaler_x.fit_transform(X_2d)
        X_scaled = X_scaled_2d.reshape(X.shape)
        y_flat = y.reshape(-1, 1)
        y_scaled_flat = scaler_y.fit_transform(y_flat)
        y_scaled = y_scaled_flat.reshape(y.shape)
        return X_scaled, y_scaled, scaler_x, scaler_y

    # ---------- 单点模式 ----------
    X, y = [], []
    # 分风机独立切片（推荐）
    if turb_id_col in df.columns:
        for turb_id, grp in df.groupby(turb_id_col):
            grp = grp.sort_values('Datetime') if 'Datetime' in grp else grp
            n = len(grp)
            for i in range(0, n - seq_len + 1, 1):  # 步长1，可改成24
                X.append(grp[feature_cols].iloc[i].values)
                y.append(grp[target_col].iloc[i:i+seq_len].values)
    # 旧逻辑（全局混合）
    else:
        for i in range(0, len(df) - seq_len + 1, 24):
            X.append(df[feature_cols].iloc[i].values)
            y.append(df[target_col].iloc[i:i+seq_len].values)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)  # shape (N, 24)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_flat = y.reshape(-1, 1)
    y_scaled_flat = scaler_y.fit_transform(y_flat)
    y_scaled = y_scaled_flat.reshape(y.shape)
    return X_scaled, y_scaled, scaler_x, scaler_y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    df = load_data()
    X, y, scaler_x, scaler_y = preprocess_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Train X shape:", X_train.shape)
    print("Train y shape:", y_train.shape)