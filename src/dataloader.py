import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/wtbdata_hourly.csv')

def load_data(path=DATA_PATH, max_samples=None):
    df = pd.read_csv(path)
    # 基本清洗：去除全空行
    df = df.dropna(how='all')
    # 填充缺失值（可根据业务调整）
    df = df.fillna(method='ffill').fillna(method='bfill')
    if max_samples is not None:
        df = df.head(max_samples)
    return df

def preprocess_features(df, feature_cols=None, target_col='Patv', seq_len=24, hist_len=24, seq_mode=False):
    """
    seq_mode: False（兼容原有），X shape (N, feature_dim); True，X shape (N, hist_len, feature_dim)
    """
    if feature_cols is None:
        feature_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv']
    X = []
    y = []
    if not seq_mode:
        for i in range(len(df) - seq_len + 1):
            X.append(df[feature_cols].iloc[i].values)
            y.append(df[target_col].iloc[i:i+seq_len].values)
        X = np.array(X)
        y = np.array(y)  # shape (N, 24)
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_x.fit_transform(X)
        y_flat = y.reshape(-1, 1)
        y_scaled_flat = scaler_y.fit_transform(y_flat)
        y_scaled = y_scaled_flat.reshape(y.shape)
        return X_scaled, y_scaled, scaler_x, scaler_y
    else:
        # 序列输入：X为历史hist_len小时的特征序列，y为未来seq_len小时的功率
        for i in range(len(df) - hist_len - seq_len + 1):
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

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    df = load_data()
    X, y, scaler_x, scaler_y = preprocess_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Train X shape:", X_train.shape)
    print("Train y shape:", y_train.shape)