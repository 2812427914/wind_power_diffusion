import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--y-true", default="v2/results/seq2seq_y_true.npy")
    p.add_argument("--y-pred-mean", default="v2/results/seq2seq_y_pred_mean.npy")
    p.add_argument("--y-samples-first", default="v2/results/seq2seq_y_samples_first.npy")
    p.add_argument("--out", default="v2/results/plots/seq2seq_scenarios.png")
    p.add_argument("--index", type=int, default=0, help="Index of sample to plot.")
    return p.parse_args()


def main():
    args = parse_args()
    print("Run args:", vars(args))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    y_true = np.load(args.y_true)            # [N, T]
    y_pred_mean = np.load(args.y_pred_mean)  # [N, T]
    y_samples_first = np.load(args.y_samples_first)  # [S, T] for the first example

    idx = min(max(0, args.index), y_true.shape[0] - 1)
    T = y_true.shape[1]

    plt.figure(figsize=(10, 6))
    # spaghetti scenarios: if a samples file is present, plot the saved scenarios.
    # The samples file produced by predict.py contains scenarios for a selected test sample.
    if y_samples_first.size > 0:
        # Ensure dimension compatibility: samples array should be [S, T]
        if y_samples_first.ndim == 1:
            samples_arr = y_samples_first.reshape(1, -1)
        else:
            samples_arr = y_samples_first
        if samples_arr.shape[1] == T:
            for s in range(min(100, samples_arr.shape[0])):
                plt.plot(range(T), samples_arr[s], color="lightsteelblue", alpha=0.35, linewidth=1)
        else:
            print(f"[WARN] Samples file has horizon {samples_arr.shape[1]}, but y_true has {T}. Skipping spaghetti plot.")

    plt.plot(range(T), y_true[idx], label="True", color="black", linewidth=2)
    plt.plot(range(T), y_pred_mean[idx], label="Pred mean", color="tab:blue", linewidth=2)

    plt.xlabel("Horizon (hours)")
    plt.ylabel("Power (kW)")
    plt.title("24h Power Forecast with Scenario Samples (Seq2Seq + MC Dropout)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("Saved plot to:", args.out)


if __name__ == "__main__":
    main()