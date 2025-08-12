import numpy as np
import os
import csv

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MODEL_FILES = {
    'diffusion': 'y_samples_diffusion.npy',
    'gan': 'y_samples_gan.npy',
    'vae': 'y_samples_vae.npy'
}

N_HOURS = 24  # 只导出前24小时

def export_model_scenarios(model_name, npy_file):
    npy_path = os.path.join(RESULTS_DIR, npy_file)
    if not os.path.exists(npy_path):
        print(f"File not found: {npy_path}")
        return

    # 加载数据
    arr = np.load(npy_path)
    # arr shape: (num_samples, n_scenarios, T) or (num_samples, n_scenarios, 24)
    if arr.ndim == 2:
        # (num_samples, n_scenarios) -> (num_samples, n_scenarios, 1)
        arr = arr[..., np.newaxis]
    num_samples, n_scenarios = arr.shape[0], arr.shape[1]
    T = arr.shape[2] if arr.ndim == 3 else 1

    # 只取前24小时
    arr = arr[..., :N_HOURS] if T >= N_HOURS else arr

    # 展平为 (num_samples * n_scenarios, N_HOURS)
    arr_flat = arr.reshape(-1, arr.shape[-1])
    n_total = arr_flat.shape[0]
    prob = 1.0 / n_total

    # 写入CSV
    csv_path = os.path.join(RESULTS_DIR, f"scenarios_{model_name}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['prob'] + [f"hour_{i+1}" for i in range(N_HOURS)]
        writer.writerow(header)
        for row in arr_flat:
            writer.writerow([prob] + list(row))
    print(f"Exported {n_total} scenarios to {csv_path}")

def main():
    for model, npy_file in MODEL_FILES.items():
        export_model_scenarios(model, npy_file)

if __name__ == "__main__":
    main()