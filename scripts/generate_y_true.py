import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
from dataloader import load_data, preprocess_features, split_data

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    # 加载数据
    df = load_data()
    X, y_scaled, _, scaler_y = preprocess_features(df)
    _, X_test, _, y_test_scaled = split_data(X, y_scaled)
    # 反归一化
    y_true = scaler_y.inverse_transform(y_test_scaled).flatten()
    # 保存
    np.save(os.path.join(RESULTS_DIR, 'y_true.npy'), y_true)
    print(f"Saved y_true.npy to {RESULTS_DIR}, shape={y_true.shape}")

if __name__ == "__main__":
    main()