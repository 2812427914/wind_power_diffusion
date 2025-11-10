import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--y-true", default="v2/results/seq2seq_y_true.npy")
    p.add_argument("--y-pred-mean", default="v2/results/seq2seq_y_pred_mean.npy")
    p.add_argument("--y-samples-all", default="v2/results/seq2seq_y_samples_all.npy")
    p.add_argument("--turb-ids", default="v2/results/seq2seq_turb_ids.npy", help="Path to turbine IDs array")
    p.add_argument("--out", default="v2/results/plots/seq2seq_scenarios.png")
    p.add_argument("--index", type=int, default=-1, help="Index of sample to plot (if >=0). If <0, use --turb-id.")
    p.add_argument("--turb-id", type=str, default=None, help="Turbine ID to plot (required if index < 0)")
    return p.parse_args()


def main():
    args = parse_args()
    print("Run args:", vars(args))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    y_true = np.load(args.y_true)            # [N, T]
    y_pred_mean = np.load(args.y_pred_mean)  # [N, T]
    y_samples_all = np.load(args.y_samples_all)  # [N, S, T]
    turb_ids = np.load(args.turb_ids)        # [N]

    # 选择要绘制的样本
    if args.index >= 0:
        # 使用索引直接选择
        idx = min(max(0, args.index), y_true.shape[0] - 1)
        turb_id = turb_ids[idx]
        print(f"Using index {idx}, Turbine ID: {turb_id}")
    elif args.turb_id is not None:
        # 根据风机ID查找
        target_indices = np.where(turb_ids == args.turb_id)[0]
        if len(target_indices) == 0:
            print(f"Error: Turbine ID {args.turb_id} not found in predictions.")
            print(f"Available turbine IDs: {np.unique(turb_ids)}")
            return
        idx = target_indices[0]  # 使用第一个匹配的样本
        print(f"Found {len(target_indices)} samples for Turbine ID {args.turb_id}, using index {idx}")
    else:
        print("Error: Must specify either --index (>=0) or --turb-id")
        return

    T = y_true.shape[1]

    plt.figure(figsize=(12, 7))
    
    # 绘制100个场景预测（spaghetti plot）
    if y_samples_all.size > 0 and y_samples_all.shape[0] > idx:
        samples_arr = y_samples_all[idx]  # [S, T]
        if samples_arr.shape[1] == T:
            # 绘制所有场景，限制为100条以避免图形过于密集
            n_scenarios = min(100, samples_arr.shape[0])
            for s in range(n_scenarios):
                plt.plot(range(T), samples_arr[s], color="lightsteelblue", alpha=0.35, linewidth=1)
        else:
            print(f"[WARN] Samples file has horizon {samples_arr.shape[1]}, but y_true has {T}. Skipping spaghetti plot.")
    else:
        print(f"[WARN] y_samples_all size={y_samples_all.size}, shape={y_samples_all.shape}, idx={idx}")

    # 绘制真实值和预测均值
    plt.plot(range(T), y_true[idx], label="True", color="black", linewidth=2.5, marker='o', markersize=4)
    plt.plot(range(T), y_pred_mean[idx], label="Pred mean", color="tab:blue", linewidth=2.5, marker='s', markersize=4)

    plt.xlabel("Hour of Day", fontsize=12)
    plt.ylabel("Power (kW)", fontsize=12)
    plt.title(f"24h Power Forecast with 100 Scenarios\nTurbine ID: {turb_ids[idx]}", fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.xticks(range(0, 24, 2), [f"{h}:00" for h in range(0, 24, 2)], rotation=45)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print("Saved plot to:", args.out)


if __name__ == "__main__":
    main()