import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_scenarios(time, y_samples, y_true=None, title="Wind Power Scenarios", save_path=None):
    plt.figure(figsize=(16, 6))
    # 只画第0条样本的所有场景
    n_samples_to_plot = min(10, y_samples.shape[1])
    for i in range(n_samples_to_plot):
        plt.plot(time, y_samples[0, i], color='deepskyblue', alpha=0.7, label='_nolegend_')
    # 画真实值
    if y_true is not None:
        plt.plot(time, y_true[0], color='red', label='True', linewidth=2)
    # 画均值预测
    y_pred = np.mean(y_samples[0], axis=0)
    plt.plot(time, y_pred, color='blue', linestyle='--', label='Mean Prediction', linewidth=2)

    plt.xlabel('Hour')
    plt.ylabel('Power (kW)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved scenario plot to {save_path}")
    plt.close()

def plot_metrics(metrics_dict, save_path=None):
    df = pd.DataFrame(metrics_dict).T  # Transpose to have models as index
    df.plot(kind='bar', figsize=(10, 7), rot=0)
    plt.title('Model Evaluation Metrics Comparison')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved metrics plot to {save_path}")
    plt.close()

def main():
    # Plot scenarios for all available models
    try:
        y_true = np.load(os.path.join(RESULTS_DIR, 'y_true.npy'))
    except FileNotFoundError:
        print("y_true.npy not found in results/. Please run prediction script first.")
        return
        
    # 只画第0条样本的24小时
    time_ax = np.arange(24)
    found_samples = False
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith('y_samples_') and f.endswith('.npy'):
            found_samples = True
            model_name = f.replace('y_samples_', '').replace('.npy', '')
            y_samples = np.load(os.path.join(RESULTS_DIR, f))
            plot_scenarios(time_ax, y_samples, y_true,
                           title=f"{model_name.upper()} Model Scenarios (Sample 0, 24 Hours)",
                           save_path=os.path.join(RESULTS_DIR, f'{model_name}_scenarios.png'))

    if not found_samples:
        print("No y_samples_*.npy files found in results/. Please run prediction script first.")

    # Plot metrics comparison
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        plot_metrics(metrics, save_path=os.path.join(RESULTS_DIR, 'metrics_comparison.png'))
    else:
        print("metrics.json not found in results/. Run evaluation script first.")

if __name__ == "__main__":
    main()
    print("\nVisualization script finished. Plots are saved in the 'results' directory.")