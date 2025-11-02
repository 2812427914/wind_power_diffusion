import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_sequences(df, feature_cols, target_col, hist_len=24, seq_len=24, stride=24):
    """
    分风机按时间生成滑窗序列（不做划分）。
    返回：X_all, y_all, seq_turb_ids, seq_positions（序列起点索引用于后续按时间划分）
    """
    X_list, Y_list, T_list, P_list = [], [], [], []
    if 'TurbID' not in df.columns:
        raise ValueError("数据缺少 TurbID 字段，无法分风机切片！")
    # 保证按风机内时间排序（上游已排序）
    for turb_id, group in df.groupby('TurbID'):
        group = group.copy()
        data = group[feature_cols + [target_col]].values
        N = len(data)
        for i in range(0, N - hist_len - seq_len + 1, stride):
            x_seq = data[i:i+hist_len, :-1]  # (hist_len, feature_dim)
            y_seq = data[i+hist_len:i+hist_len+seq_len, -1]  # (seq_len,)
            X_list.append(x_seq)
            Y_list.append(y_seq)
            T_list.append(turb_id)
            P_list.append(i)  # 窗口起点（按时间递增）
    if len(X_list) == 0:
        feature_dim = len(feature_cols)
        return (np.empty((0, hist_len, feature_dim)),
                np.empty((0, seq_len)),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=int))
    X_all = np.stack(X_list, axis=0)
    y_all = np.stack(Y_list, axis=0)
    seq_turb_ids = np.array(T_list, dtype=int)
    seq_positions = np.array(P_list, dtype=int)
    return X_all, y_all, seq_turb_ids, seq_positions

if __name__ == "__main__":
    # 假设脚本在 scripts/ 目录，数据在 data/ 目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base_dir, 'data', 'wtbdata_hourly.csv')
    output_data_dir = os.path.join(base_dir, 'data')

    # 1. 加载原始数据
    df = pd.read_csv(raw_data_path)
    # 按时间排序，确保序列连续有序，并构造时间特征
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df[df['Datetime'].notnull()].sort_values(['TurbID', 'Datetime'])
        # 时间特征：小时与星期几
        df['Hour'] = df['Datetime'].dt.hour
        df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    elif 'Date' in df.columns and 'Tmstamp' in df.columns:
        # 尝试通过 Date + Tmstamp 构造完整时间戳
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Tmstamp'].astype(str), errors='coerce')
        df = df[df['Datetime'].notnull()].sort_values(['TurbID', 'Datetime'])
        df['Hour'] = df['Datetime'].dt.hour
        df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    elif 'Day' in df.columns and 'Tmstamp' in df.columns:
        df = df.sort_values(['TurbID', 'Day', 'Tmstamp'])
        # 解析 "HH:MM" 字符串或数值到小时
        hours_str = pd.to_datetime(df['Tmstamp'].astype(str), format='%H:%M', errors='coerce').dt.hour
        hours_num = pd.to_numeric(df['Tmstamp'], errors='coerce') % 24
        df['Hour'] = hours_str.fillna(hours_num).fillna(0).astype(int)
        # 近似星期几：按 Day 对 7 取模
        dow_numeric = pd.to_numeric(df['Day'], errors='coerce')
        df['DayOfWeek'] = (dow_numeric.fillna(0) % 7).astype(int)
    else:
        # 无法识别时间字段，至少保持按 TurbID 排序
        df = df.sort_values(['TurbID'])

    # 构建特征列：基础特征 + 时间特征（若可用）
    target_col = 'Patv'
    hist_len = 72
    seq_len = 24
    val_ratio = 0.1
    stride = 1
    # 基础特征
    feature_cols = ['Wspd', 'TurbID']
    # 时间特征（如果可用）
    if 'Hour' in df.columns:
        feature_cols.append('Hour')
    if 'DayOfWeek' in df.columns:
        feature_cols.append('DayOfWeek')
    # 其他可用气象特征（若存在则加入）
    for col in ['Wdir', 'Etmp', 'Prs']:
        if col in df.columns:
            feature_cols.append(col)

    # 2. 生成序列数据（不划分）
    print("Creating sequences...")
    X_all, y_all, seq_turb_ids, seq_positions = create_sequences(df, feature_cols, target_col, hist_len, seq_len, stride=stride)
    print(f"Original sequences: X_all={X_all.shape}, y_all={y_all.shape}")

    # 3. 清理 NaN/inf 和异常（统一在全部序列上清洗，然后再划分）
    def clean_all(X, y, max_bad_steps=2):
        if X.size == 0 or y.size == 0:
            return X, y, np.zeros((0,), dtype=bool)
        mask = (~np.isnan(X).any(axis=(1, 2)) & ~np.isinf(X).any(axis=(1, 2)) &
                ~np.isnan(y).any(axis=1) & ~np.isinf(y).any(axis=1))
        # 进一步过滤物理异常值：Patv < 0 或 Wspd < 0（允许每条序列最多 2 个异常点）
        mask_patv = (y < 0).sum(axis=1) <= max_bad_steps
        # Wspd 假定为 feature_cols[0]
        mask_wspd = (X[:, :, 0] < 0).sum(axis=1) <= max_bad_steps
        # 新增：去掉所有含零功率或负功率的序列
        mask_no_zero = np.all(y > 0, axis=1)
        final_mask = mask & mask_patv & mask_wspd & mask_no_zero
        return X[final_mask], y[final_mask], final_mask

    X_all, y_all, valid_mask = clean_all(X_all, y_all)
    seq_turb_ids = seq_turb_ids[valid_mask] if valid_mask.size else seq_turb_ids
    seq_positions = seq_positions[valid_mask] if valid_mask.size else seq_positions
    print(f"Shapes after cleaning: X_all={X_all.shape}, y_all={y_all.shape}")

    # 4. 基于清洗后的样本，按每台风机时间顺序划分训练/验证（最后 val_ratio 作为验证）
    def split_per_turbine_after_clean(X, y, turb_ids, positions, val_ratio=0.1):
        tr_idx_all, va_idx_all = [], []
        unique_ids = np.unique(turb_ids)
        for tid in unique_ids:
            idx = np.where(turb_ids == tid)[0]
            if idx.size == 0:
                continue
            # 按窗口起点（时间）排序，保持时间一致
            order = np.argsort(positions[idx], kind='mergesort')
            idx_ordered = idx[order]
            M = idx_ordered.size
            if M == 0:
                continue
            split_idx = max(1, int(np.floor(M * (1.0 - val_ratio))))
            tr_idx_all.append(idx_ordered[:split_idx])
            va_idx_all.append(idx_ordered[split_idx:])
        if len(tr_idx_all) == 0 and len(va_idx_all) == 0:
            feature_dim = X.shape[-1] if X.ndim == 3 else len(feature_cols)
            return (np.empty((0, hist_len, feature_dim)),
                    np.empty((0, seq_len)),
                    np.empty((0, hist_len, feature_dim)),
                    np.empty((0, seq_len)))
        tr_idx_all = np.concatenate(tr_idx_all, axis=0) if len(tr_idx_all) else np.array([], dtype=int)
        va_idx_all = np.concatenate(va_idx_all, axis=0) if len(va_idx_all) else np.array([], dtype=int)
        return X[tr_idx_all], y[tr_idx_all], X[va_idx_all], y[va_idx_all]

    X_train, y_train, X_val, y_val = split_per_turbine_after_clean(X_all, y_all, seq_turb_ids, seq_positions, val_ratio=val_ratio)
    print(f"Post-split shapes: X_train={X_train.shape}, y_train={y_train.shape} | X_val={X_val.shape}, y_val={y_val.shape}")

    # 5. 仅用训练集拟合 Scaler；X 按特征维度统一 MinMax；y 扁平后统一 MinMax 再 reshape
    scaler_x = MinMaxScaler()
    if X_train.size > 0:
        X_train_scaled = scaler_x.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    else:
        # 空训练集保护
        scaler_x.fit(np.zeros((1, len(feature_cols))))
        X_train_scaled = X_train
    if X_val.size > 0:
        X_val_scaled = scaler_x.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    else:
        X_val_scaled = X_val

    scaler_y = MinMaxScaler()
    if y_train.size > 0:
        y_train_flat = y_train.reshape(-1, 1)
        scaler_y.fit(y_train_flat)
        y_train_scaled = scaler_y.transform(y_train_flat).reshape(y_train.shape)
    else:
        scaler_y.fit(np.zeros((1, 1)))
        y_train_scaled = y_train
    if y_val.size > 0:
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    else:
        y_val_scaled = y_val

    # 5. 保存结果（分别保存 train/val；同时保存合并版以兼容旧流程）
    np.save(os.path.join(output_data_dir, 'X_seq_train.npy'), X_train_scaled)
    np.save(os.path.join(output_data_dir, 'y_seq_train.npy'), y_train_scaled)
    np.save(os.path.join(output_data_dir, 'X_seq_val.npy'), X_val_scaled)
    np.save(os.path.join(output_data_dir, 'y_seq_val.npy'), y_val_scaled)

    # 合并版（注意：使用训练段拟合的 scaler 统一变换，避免数据泄漏）
    X_all_scaled = np.concatenate([X_train_scaled, X_val_scaled], axis=0) if X_train_scaled.size and X_val_scaled.size else (X_train_scaled if X_val_scaled.size == 0 else X_val_scaled)
    y_all_scaled = np.concatenate([y_train_scaled, y_val_scaled], axis=0) if y_train_scaled.size and y_val_scaled.size else (y_train_scaled if y_val_scaled.size == 0 else y_val_scaled)
    np.save(os.path.join(output_data_dir, 'X_seq.npy'), X_all_scaled)
    np.save(os.path.join(output_data_dir, 'y_seq.npy'), y_all_scaled)

    joblib.dump(scaler_x, os.path.join(output_data_dir, 'scaler_x.joblib'))
    joblib.dump(scaler_y, os.path.join(output_data_dir, 'scaler_y.joblib'))

    print(f"Successfully preprocessed and saved data to {output_data_dir}")
    print(f"  - X_seq_train.npy: {X_train_scaled.shape}")
    print(f"  - y_seq_train.npy: {y_train_scaled.shape}")
    print(f"  - X_seq_val.npy:   {X_val_scaled.shape}")
    print(f"  - y_seq_val.npy:   {y_val_scaled.shape}")
    print(f"  - X_seq.npy (merged, transformed by train scalers): {X_all_scaled.shape}")
    print(f"  - y_seq.npy (merged, transformed by train scalers): {y_all_scaled.shape}")
    print(f"  - scaler_x.joblib (fit on train)")
    print(f"  - scaler_y.joblib (fit on train)")

    # ========== 可视化：随机选择 1 台风机，按时间绘制 Patv，并区分训练/验证目标点 ==========
    try:
        rng = np.random.RandomState(42)
        # 选择有足够样本的风机
        candidate_ids = []
        for turb_id, group in df.groupby('TurbID'):
            N = len(group)
            if N >= (hist_len + seq_len + 1):
                # 该风机至少能产生 1 个样本窗口
                candidate_ids.append(turb_id)
        if len(candidate_ids) == 0:
            print("[Viz] 未找到可视化的风机（样本不足），跳过绘图。")
        else:
            pick_id = int(rng.choice(candidate_ids))
            g = df[df['TurbID'] == pick_id].copy()
            # 确保排序
            if 'Datetime' in g.columns:
                g = g[g['Datetime'].notnull()].sort_values(['TurbID', 'Datetime'])
                x_axis = g['Datetime'].values
            elif 'Day' in g.columns and 'Tmstamp' in g.columns:
                g = g.sort_values(['TurbID', 'Day', 'Tmstamp'])
                # 无法还原完整时间戳时，用序号作为横轴
                x_axis = np.arange(len(g))
            else:
                g = g.sort_values(g.columns.tolist())  # 保底
                x_axis = np.arange(len(g))
            y_series = g[target_col].values

            # 计算该风机的训练/验证目标点掩码
            N = len(g)
            train_mask = np.zeros(N, dtype=bool)
            val_mask = np.zeros(N, dtype=bool)
            # 生成窗口起点序列
            starts = list(range(0, max(0, N - hist_len - seq_len + 1), stride))
            M = len(starts)
            if M > 0:
                split_idx = max(1, int(np.floor(M * (1.0 - val_ratio))))
                for idx, i in enumerate(starts):
                    target_idx = np.arange(i + hist_len, i + hist_len + seq_len)
                    target_idx = target_idx[target_idx < N]  # 安全截断
                    if idx < split_idx:
                        train_mask[target_idx] = True
                    else:
                        val_mask[target_idx] = True

            # 绘图
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.scatter(x_axis, y_series, s=6, color='#9e9e9e', alpha=0.5, label='All Patv (raw)')
            ax.scatter(np.array(x_axis)[train_mask], y_series[train_mask], s=12, color='#2ca02c', alpha=0.9, label='Train targets')
            ax.scatter(np.array(x_axis)[val_mask], y_series[val_mask], s=12, color='#d62728', alpha=0.9, label='Val targets')
            ax.set_title(f'Turbine {pick_id} Patv with Train/Val Target Points')
            ax.set_xlabel('Time')
            ax.set_ylabel('Patv')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            if 'Datetime' in g.columns:
                fig.autofmt_xdate()
            out_png = os.path.join(output_data_dir, f'preview_turb_{pick_id}.png')
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"[Viz] 预览图已保存: {out_png}")
    except Exception as e:
        print(f"[Viz] 绘图失败: {e}")