import warnings
warnings.filterwarnings("ignore")
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import numpy as np
import torch
torch.set_num_threads(1)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '../checkpoints')
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

def get_default_device():
    """Pick GPU if available, else CPU. Priority: CUDA > CPU (MPS disabled for cumprod)."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Warning: torch.cumprod not supported on MPS, switching to CPU.")
        return "cpu"
    return "cpu"

def predict_seq_ar_diffusion_lazy(args):
    from torch.utils.data import DataLoader, Subset
    from dataloader import WindPowerCSVDataset
    from seq_ar_diffusion import SeqCondEncoder as ARCondEncoder, SeqARDiffusionModel, SeqARGaussianDiffusion

    device = args.device

    # 构建数据集（与 train.py 懒加载路径保持一致）
    csv_path = os.path.join(DATA_DIR, 'wtbdata_hourly.csv')
    dataset_full = WindPowerCSVDataset(csv_path, hist_len=args.hist_len, seq_len=args.seq_len, seq_mode=True)

    if args.max_samples is not None:
        n = min(args.max_samples, len(dataset_full))
        dataset = Subset(dataset_full, range(n))
        feature_dim = len(dataset_full.feature_cols)
    else:
        dataset = dataset_full
        feature_dim = len(dataset_full.feature_cols)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 构建模型与扩散
    cond_encoder = ARCondEncoder(feature_dim=feature_dim, hist_len=args.hist_len).to(device)
    model = SeqARDiffusionModel(cond_dim=cond_encoder.hidden_dim, y_dim=1).to(device)
    diffusion = SeqARGaussianDiffusion(model, cond_encoder, device=device, seq_len=args.seq_len)

    # 加载权重（优先 best）
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'seq_ar_diffusion_best.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, 'seq_ar_diffusion_last.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到 SeqAR-Diffusion 的 checkpoint: {CHECKPOINT_DIR}/seq_ar_diffusion_best.pth 或 seq_ar_diffusion_last.pth")

    state = torch.load(ckpt_path, map_location=device)
    cond_encoder.load_state_dict(state['cond_encoder'])
    model.load_state_dict(state['model'])
    cond_encoder.eval()
    model.eval()

    # 逐批预测并采样
    all_y_true = []
    all_y_pred_mean = []
    all_y_samples = []

    with torch.no_grad():
        for batch in loader:
            # 期望 Dataset 返回 (x_hist, y_future)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x_hist, y_true = batch
            else:
                raise RuntimeError("Unexpected dataset output; expected (x_hist, y_future) tuple.")
            x_hist = x_hist.float().to(device)
            y_true_np = y_true.float().cpu().numpy()  # (B, seq_len)

            samples_list = []
            for _ in range(args.n_samples):
                y_sample = diffusion.sample(x_hist, seq_len=args.seq_len)  # (B, seq_len)
                samples_list.append(y_sample.unsqueeze(1).cpu().numpy())   # (B,1,seq_len)
            y_samples_batch = np.concatenate(samples_list, axis=1)         # (B, n_samples, seq_len)
            y_pred_mean_batch = y_samples_batch.mean(axis=1)               # (B, seq_len)

            all_y_true.append(y_true_np)
            all_y_pred_mean.append(y_pred_mean_batch)
            all_y_samples.append(y_samples_batch)

    y_true_arr = np.concatenate(all_y_true, axis=0)                # (N, seq_len)
    y_pred_arr = np.concatenate(all_y_pred_mean, axis=0)           # (N, seq_len)
    y_samples_arr = np.concatenate(all_y_samples, axis=0)          # (N, n_samples, seq_len)

    # 保存到 results（评估脚本会自动读取这些文件）
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_name = args.model
    np.save(os.path.join(RESULTS_DIR, f'y_true_{model_name}.npy'), y_true_arr)
    np.save(os.path.join(RESULTS_DIR, f'y_pred_{model_name}.npy'), y_pred_arr)
    np.save(os.path.join(RESULTS_DIR, f'y_samples_{model_name}.npy'), y_samples_arr)
    print(f"Saved predictions to {RESULTS_DIR}: y_true_{model_name}.npy, y_pred_{model_name}.npy, y_samples_{model_name}.npy")

def main():
    parser = argparse.ArgumentParser(description="Lazy dataset prediction for wind power models.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['seq_ar_diffusion'],
                        help='Model to run prediction for (currently supports seq_ar_diffusion).')
    parser.add_argument('--n_samples', type=int, default=200, help='Number of diffusion samples per instance.')
    parser.add_argument('--device', type=str, default=get_default_device(), help="Device to run prediction on (e.g., 'cpu', 'cuda').")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for prediction.')
    parser.add_argument('--use_lazy_dataset', action='store_true', help='Use lazy CSV dataset for prediction.')
    parser.add_argument('--max_samples', type=int, default=None, help='Only use the first N samples from CSV (lazy mode).')
    parser.add_argument('--hist_len', type=int, default=24, help='History window length for sequence models.')
    parser.add_argument('--seq_len', type=int, default=24, help='Forecast horizon for sequence models.')
    args = parser.parse_args()

    if not args.use_lazy_dataset:
        raise RuntimeError("predict_lazy.py 仅支持 --use_lazy_dataset 模式。请添加该参数或使用原始 predict.py。")

    if args.model == 'seq_ar_diffusion':
        predict_seq_ar_diffusion_lazy(args)
    else:
        raise NotImplementedError(f"Model {args.model} not supported in predict_lazy.py.")

if __name__ == "__main__":
    main()