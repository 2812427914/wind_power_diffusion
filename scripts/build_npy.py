import pandas as pd
import numpy as np
from pathlib import Path

DATA_CSV = Path("/Users/bytedance/Downloads/gits/wind_power_diffusion/data/wtbdata_hourly.csv").resolve()
OUT_DIR  = DATA_CSV.parent

def main():
    df = pd.read_csv(DATA_CSV)
    feat_cols = ['Wspd','Wdir','Etmp','Itmp','Ndir','Pab1','Pab2','Pab3','Prtv']
    hist, pred = 24, 24

    X_seq_list = []
    y_seq_list = []

    # 分风机独立切片
    if 'TurbID' not in df.columns:
        raise ValueError("数据缺少 TurbID 字段，无法分风机切片！")

    for turb_id, group in df.groupby('TurbID'):
        X = group[feat_cols].astype(np.float32).values
        y = group['Patv'].astype(np.float32).values
        # 滑窗切片，每台风机每天一个样本
        if len(X) >= hist + pred:
            X_seq = np.lib.stride_tricks.sliding_window_view(X, (hist, len(feat_cols)))[::24, 0]
            y_seq = np.lib.stride_tricks.sliding_window_view(y, pred)[hist::24]
            X_seq_list.append(X_seq)
            y_seq_list.append(y_seq)

    X_seq_all = np.concatenate(X_seq_list, axis=0)
    y_seq_all = np.concatenate(y_seq_list, axis=0)

    np.save(OUT_DIR/'X_seq.npy', X_seq_all)
    np.save(OUT_DIR/'y_seq.npy', y_seq_all)
    print(f"分风机切片完成，总样本数: {X_seq_all.shape[0]}，X_seq shape: {X_seq_all.shape}，y_seq shape: {y_seq_all.shape}")

if __name__ == '__main__':
    main()