import os
import numpy as np
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')

# 选择数据文件（可选 train/val 或合并版）
y_seq_path = os.path.join(data_dir, 'y_seq.npy')
scaler_y_path = os.path.join(data_dir, 'scaler_y.joblib')

y_scaled = np.load(y_seq_path)
scaler_y = joblib.load(scaler_y_path)

# 反归一化
y = scaler_y.inverse_transform(y_scaled)

# 统计“含零功率序列”的比例（每条序列只要有一个时间步为 0 就算作含零）
total_seq = y.shape[0]
zero_seq_count = np.sum(np.any(y == 0, axis=1))
zero_seq_ratio = zero_seq_count / total_seq

# 统计“含负功率序列”的比例（每条序列只要有一个时间步为负就算作含负）
neg_seq_count = np.sum(np.any(y < 0, axis=1))
neg_seq_ratio = neg_seq_count / total_seq

print(f"总序列数: {total_seq}")
print(f"含零功率的序列数: {zero_seq_count}")
print(f"含零功率的序列比例: {zero_seq_ratio:.6f}")
print(f"含负功率的序列数: {neg_seq_count}")
print(f"含负功率的序列比例: {neg_seq_ratio:.6f}")