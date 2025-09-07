import argparse
import numpy as np
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from properscoring import crps_ensemble

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')

def quantile_score(y_true, y_samples, quantile=0.5):
    """
    Compute quantile score for univariate data.
    y_true: shape (N,)
    y_samples: shape (n_samples, N)
    """
    y_true = np.asarray(y_true)
    y_samples = np.asarray(y_samples)
    q_pred = np.quantile(y_samples, quantile, axis=0)
    q = quantile
    return 2 * np.mean((y_true - q_pred) * ((y_true < q_pred) - q))

def energy_score(y, y_samples):
    """
    Simple energy score implementation for univariate data.
    y: shape (N,)
    y_samples: shape (n_samples, N)
    """
    y = np.asarray(y)
    y_samples = np.asarray(y_samples)
    n_samples = y_samples.shape[0]
    # First term: mean L2 distance between y and each sample
    term1 = np.mean(np.linalg.norm(y_samples - y, axis=1))
    # Second term: mean L2 distance between all pairs of samples
    diffs = y_samples[:, None, :] - y_samples[None, :, :]
    term2 = 0.5 * np.mean(np.linalg.norm(diffs, axis=2))
    return term1 - term2

def evaluate_metrics(y_true, y_pred, y_samples=None):
    # y_true, y_pred: shape (N, 24)
    # y_samples: shape (N, n_samples, 24) for probabilistic metrics
    results = {}
    # 逐小时计算 MAE、RMSE，然后取均值
    results['MAE'] = mean_absolute_error(y_true, y_pred)
    results['RMSE'] = mean_squared_error(y_true, y_pred, squared=False)
    if y_samples is not None:
        # 对每个时间步分别计算 CRPS/ES/QS，然后对 24 小时取均值
        N, n_samples, T = y_samples.shape
        crps_list = []
        es_list = []
        qs_list = []
        for t in range(T):
            yt = y_true[:, t]
            yp_samples = y_samples[:, :, t]
            crps_list.append(np.mean(crps_ensemble(yt, yp_samples)))
            es_list.append(energy_score(yt, yp_samples.T))
            qs_list.append(quantile_score(yt, yp_samples.T, quantile=0.5))
        results['CRPS'] = float(np.mean(crps_list))
        results['ES'] = float(np.mean(es_list))
        results['QS'] = float(np.mean(qs_list))
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model's predictions.")
    parser.add_argument('--model', type=str, required=True, choices=['diffusion', 'gan', 'vae', 'weibull_diffusion', 'seq_vae', 'seq_diffusion', 'seq_ar_diffusion'],
                        help='The model to evaluate.')
    args = parser.parse_args()
    model_name = args.model

    try:
        y_true = np.load(os.path.join(RESULTS_DIR, 'y_true.npy'))
        y_pred = np.load(os.path.join(RESULTS_DIR, f'y_pred_{model_name}.npy'))
        y_samples = np.load(os.path.join(RESULTS_DIR, f'y_samples_{model_name}.npy'))
    except FileNotFoundError as e:
        print(f"Error loading prediction files for model '{model_name}': {e}")
        print("Please run the prediction script first using 'bash scripts/run_predict.sh'.")
        return

    metrics = evaluate_metrics(y_true, y_pred, y_samples)
    print(f"Metrics for {model_name.upper()} model:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.5f}")

    # Save metrics to a JSON file
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    all_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            all_metrics = json.load(f)

    all_metrics[model_name.upper()] = metrics
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nUpdated metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()