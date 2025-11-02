import warnings
warnings.filterwarnings("ignore")
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import numpy as np
import torch
torch.set_num_threads(1)
from tqdm import tqdm
import json

from dataloader import load_data, preprocess_features, split_data
from diffusion import SimpleDiffusionModel, GaussianDiffusion
from baseline_gan import GANGenerator
from baseline_vae import VAE
from diffusion_weibull import SimpleDiffusionModelWeibull, WeibullDiffusion
from seq_vae import SeqVAE
from seq_diffusion import SeqCondEncoder, SeqDiffusionModel, SeqGaussianDiffusion
from seq_ar_diffusion import SeqCondEncoder as ARCondEncoder, SeqARDiffusionModel, SeqARGaussianDiffusion
from seq_lstm import SeqLSTMModel

def predict_seq_ar_diffusion(X_test, y_dim, device='cpu', n_samples=100, batch_size=32,
                             hist_len=24, seq_len=24, feature_dim=9, hidden_dim=256,
                             timesteps=300, schedule='cosine', dropout=0.1, t_embed_dim=32):
    print("Generating predictions with SeqAR-Diffusion model...")
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'seq_ar_diffusion_best.pth'), map_location=device)
    meta = checkpoint.get('meta', {})
    feature_dim = int(meta.get('feature_dim', feature_dim))
    hist_len = int(meta.get('hist_len', hist_len))
    seq_len = int(meta.get('seq_len', seq_len))
    hidden_dim = int(meta.get('hidden_dim', hidden_dim))
    dropout = float(meta.get('dropout', dropout))
    t_embed_dim = int(meta.get('t_embed_dim', t_embed_dim))
    timesteps = int(meta.get('timesteps', timesteps))
    schedule = meta.get('schedule', schedule)

    cond_encoder = ARCondEncoder(feature_dim=feature_dim, hist_len=hist_len).to(device)
    model = SeqARDiffusionModel(
        cond_dim=cond_encoder.hidden_dim + 1,
        y_dim=y_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        t_embed_dim=t_embed_dim
    ).to(device)
    cond_encoder.load_state_dict(checkpoint['cond_encoder'])
    model.load_state_dict(checkpoint['model'])
    cond_encoder.eval()
    model.eval()
    diffusion = SeqARGaussianDiffusion(
        model, cond_encoder, timesteps=timesteps, device=device,
        seq_len=seq_len, schedule=schedule
    )
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    n_test = X_test.shape[0]
    y_samples_list = []
    with torch.no_grad():
        for i in tqdm(range(0, n_test, batch_size), desc="SeqAR-Diffusion Sampling"):
            X_batch = X_test_tensor[i:i+batch_size].to(device)
            batch_samples = []
            for _ in range(n_samples):
                samples = diffusion.sample(X_batch, seq_len=seq_len, deterministic=False)
                batch_samples.append(samples.cpu().numpy())
            batch_samples = np.stack(batch_samples, axis=1)  # (batch, n_samples, seq_len)
            y_samples_list.append(batch_samples)
    y_samples_all = np.concatenate(y_samples_list, axis=0)  # (N, n_samples, seq_len)
    y_samples_all = np.clip(y_samples_all, 0.0, 1.0)
    return y_samples_all

# Directories
BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(BASE_DIR, '../checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, '../results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def predict_diffusion(X_test, y_dim, device='cpu', n_samples=100, batch_size=32):
    print("Generating predictions with Diffusion model...")
    cond_dim = X_test.shape[1]
    input_dim = cond_dim + y_dim + 1  # +1 for timestep
    model = SimpleDiffusionModel(input_dim=input_dim, output_dim=y_dim).to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'diffusion_best.pth'), map_location=device))
    model.eval()
    diffusion = GaussianDiffusion(model, device=device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    n_test = X_test.shape[0]

    y_samples_list = []
    with torch.no_grad():
        for i in tqdm(range(0, n_test, batch_size), desc="Diffusion Sampling"):
            X_batch = X_test_tensor[i:i+batch_size].to(device)
            
            # Expand X_batch to generate n_samples for each input
            X_batch_expanded = X_batch.repeat_interleave(n_samples, dim=0)
            
            samples = diffusion.sample(X_batch_expanded, y_shape=(X_batch_expanded.shape[0], y_dim))
            
            # Reshape samples: from (batch_size * n_samples, y_dim) to (batch_size, n_samples, y_dim)
            samples_reshaped = samples.view(X_batch.shape[0], n_samples, y_dim)
            
            y_samples_list.append(samples_reshaped.cpu().numpy())

    # Concatenate results from all batches
    y_samples_all = np.concatenate(y_samples_list, axis=0)
    return y_samples_all  # shape (N, n_samples, y_dim)

def predict_weibull_diffusion(X_test, y_dim, device='cpu', n_samples=100, batch_size=32, weibull_shape=2.0):
    print("Generating predictions with Weibull Diffusion model...")
    cond_dim = X_test.shape[1]
    input_dim = cond_dim + y_dim + 1  # +1 for timestep
    model = SimpleDiffusionModelWeibull(input_dim=input_dim, output_dim=y_dim).to(device)
    try:
        state = torch.load(os.path.join(CHECKPOINT_DIR, 'diffusion_weibull_best.pth'), map_location=device)
    except FileNotFoundError:
        state = torch.load(os.path.join(CHECKPOINT_DIR, 'diffusion_weibull.pth'), map_location=device)
    model.load_state_dict(state)
    model.eval()
    diffusion = WeibullDiffusion(model, device=device, weibull_shape=weibull_shape)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    n_test = X_test.shape[0]

    y_samples_list = []
    with torch.no_grad():
        for i in tqdm(range(0, n_test, batch_size), desc="WeibullDiff Sampling"):
            X_batch = X_test_tensor[i:i+batch_size].to(device)
            X_batch_expanded = X_batch.repeat_interleave(n_samples, dim=0)
            samples = diffusion.sample(X_batch_expanded, y_shape=(X_batch_expanded.shape[0], y_dim))
            samples_reshaped = samples.view(X_batch.shape[0], n_samples, y_dim)
            y_samples_list.append(samples_reshaped.cpu().numpy())

    y_samples_all = np.concatenate(y_samples_list, axis=0)
    return y_samples_all  # shape (N, n_samples, y_dim)

def predict_gan(X_test, y_dim, device='cpu', n_samples=100, latent_dim=32, batch_size=128):
    print("Generating predictions with GAN model...")
    cond_dim = X_test.shape[1]
    generator = GANGenerator(cond_dim=cond_dim, latent_dim=latent_dim, output_dim=y_dim).to(device)
    generator.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'gan_generator_best.pth'), map_location=device))
    generator.eval()
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    n_test = X_test.shape[0]

    y_samples_list = []
    with torch.no_grad():
        for i in tqdm(range(0, n_test, batch_size), desc="GAN Sampling"):
            X_batch = X_test_tensor[i:i+batch_size].to(device)
            X_batch_expanded = X_batch.repeat_interleave(n_samples, dim=0)

            z = torch.randn(X_batch_expanded.shape[0], latent_dim).to(device)
            samples = generator(z, X_batch_expanded)

            samples_reshaped = samples.view(X_batch.shape[0], n_samples, y_dim)
            y_samples_list.append(samples_reshaped.cpu().numpy())

    y_samples_all = np.concatenate(y_samples_list, axis=0)
    return y_samples_all  # shape (N, n_samples, y_dim)

def predict_vae(X_test, y_dim, device='cpu', n_samples=100, latent_dim=8, batch_size=128):
    print("Generating predictions with VAE model...")
    cond_dim = X_test.shape[1]
    vae = VAE(cond_dim=cond_dim, y_dim=y_dim, latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'vae_best.pth'), map_location=device))
    vae.eval()
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    n_test = X_test.shape[0]

    y_samples_list = []
    with torch.no_grad():
        for i in tqdm(range(0, n_test, batch_size), desc="VAE Sampling"):
            X_batch = X_test_tensor[i:i+batch_size].to(device)
            X_batch_expanded = X_batch.repeat_interleave(n_samples, dim=0)
            
            z = torch.randn(X_batch_expanded.shape[0], latent_dim).to(device)
            samples = vae.decode(z, X_batch_expanded)

            samples_reshaped = samples.view(X_batch.shape[0], n_samples, y_dim)
            y_samples_list.append(samples_reshaped.cpu().numpy())

    y_samples_all = np.concatenate(y_samples_list, axis=0)
    return y_samples_all  # shape (N, n_samples, y_dim)

def predict_seq_vae(X_test, y_dim, device='cpu', n_samples=100, batch_size=32, hist_len=24, seq_len=24, feature_dim=9):
    print("Generating predictions with SeqVAE model...")
    model = SeqVAE(feature_dim=feature_dim, hist_len=hist_len, y_dim=1, seq_len=seq_len).to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'seq_vae_best.pth'), map_location=device))
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    n_test = X_test.shape[0]
    y_samples_list = []
    with torch.no_grad():
        for i in tqdm(range(0, n_test, batch_size), desc="SeqVAE Sampling"):
            X_batch = X_test_tensor[i:i+batch_size].to(device)
            batch_samples = []
            for _ in range(n_samples):
                y_pred = model.decode(model.reparameterize(
                    *model.encode(X_batch)
                ))  # (batch, seq_len)
                batch_samples.append(y_pred.cpu().numpy())
            batch_samples = np.stack(batch_samples, axis=1)  # (batch, n_samples, seq_len)
            y_samples_list.append(batch_samples)
    y_samples_all = np.concatenate(y_samples_list, axis=0)  # (N, n_samples, seq_len)
    # 关键：先裁剪到归一化区间，再反归一化
    y_samples_all = np.clip(y_samples_all, 0.0, 1.0)
    return y_samples_all

def predict_seq_diffusion(X_test, y_dim, device='cpu', n_samples=100, batch_size=32, hist_len=24, seq_len=24, feature_dim=9):
    print("Generating predictions with SeqDiffusion model...")
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'seq_diffusion_best.pth'), map_location=device)
    cond_encoder = SeqCondEncoder(feature_dim=feature_dim, hist_len=hist_len).to(device)
    model = SeqDiffusionModel(cond_dim=cond_encoder.hidden_dim, y_dim=y_dim).to(device)
    cond_encoder.load_state_dict(checkpoint['cond_encoder'])
    model.load_state_dict(checkpoint['model'])
    cond_encoder.eval()
    model.eval()
    diffusion = SeqGaussianDiffusion(model, cond_encoder, device=device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    n_test = X_test.shape[0]
    y_samples_list = []
    with torch.no_grad():
        for i in tqdm(range(0, n_test, batch_size), desc="SeqDiffusion Sampling"):
            X_batch = X_test_tensor[i:i+batch_size].to(device)
            batch_samples = []
            for _ in range(n_samples):
                samples = diffusion.sample(X_batch, y_shape=(X_batch.shape[0], seq_len))
                batch_samples.append(samples.cpu().numpy())
            batch_samples = np.stack(batch_samples, axis=1)  # (batch, n_samples, seq_len)
            y_samples_list.append(batch_samples)
    y_samples_all = np.concatenate(y_samples_list, axis=0)  # (N, n_samples, seq_len)
    # 关键：先裁剪到归一化区间，再反归一化
    y_samples_all = np.clip(y_samples_all, 0.0, 1.0)
    return y_samples_all

def predict_seq_lstm(X_test, y_dim, device='cpu', batch_size=32, hist_len=24, seq_len=24, feature_dim=1, hidden_dim=128, dropout=0.1):
    print("Generating predictions with SeqLSTM model...")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'seq_lstm_best.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    meta = checkpoint.get('meta', None)
    if meta is not None:
        feature_dim = int(meta.get('feature_dim', feature_dim))
        hist_len = int(meta.get('hist_len', hist_len))
        seq_len = int(meta.get('seq_len', seq_len))
        hidden_dim = int(meta.get('hidden_dim', hidden_dim))
        dropout = float(meta.get('dropout', dropout))
    model = SeqLSTMModel(feature_dim=feature_dim, hist_len=hist_len, seq_len=seq_len, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    n_test = X_test.shape[0]
    y_pred_list = []
    with torch.no_grad():
        for i in tqdm(range(0, n_test, batch_size), desc="SeqLSTM Sampling"):
            X_batch = X_test_tensor[i:i+batch_size].to(device, non_blocking=True)
            y_pred = model(X_batch).cpu().numpy()
            y_pred_list.append(y_pred)
    y_pred_all = np.concatenate(y_pred_list, axis=0)  # (N, seq_len)
    y_pred_all = np.clip(y_pred_all, 0.0, 1.0)
    return y_pred_all

def get_default_device():
    """Pick GPU if available, else CPU. Priority: CUDA > CPU (MPS disabled for cumprod)."""
    if torch.cuda.is_available():
        return "cuda"
    # Check for Apple Silicon (M1/M2)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS不支持torch.cumprod，强制用CPU
        print("Warning: torch.cumprod not supported on MPS, switching to CPU.")
        return "cpu"
    return "cpu"

def main():
    parser = argparse.ArgumentParser(description="Generate predictions from a trained model.")
    parser.add_argument('--model', type=str, required=True, choices=['diffusion', 'gan', 'vae', 'weibull_diffusion', 'seq_vae', 'seq_diffusion', 'seq_ar_diffusion', 'seq_lstm'])
    parser.add_argument('--n_samples', type=int, default=100, help="Number of samples to generate per data point.")
    parser.add_argument('--device', type=str, default=get_default_device(),
                        help="Device to run prediction on (e.g., 'cpu', 'cuda', 'mps'). "
                             "Automatically detects available hardware.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for prediction.")
    parser.add_argument('--dry_run_fit_train', action='store_true', help="Use all data as both train and test for dry-run overfit check.")
    parser.add_argument('--subset_frac', type=float, default=0.1, help="Fraction (0-1] of test set to predict (default 0.1 -> 10%).")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden dimension for LSTM/MLP (default: 128)")
    parser.add_argument('--timesteps', type=int, default=300, help="Number of diffusion timesteps for sampling.")
    parser.add_argument('--schedule', type=str, default='cosine', choices=['linear', 'cosine'], help="Noise schedule type for sampling.")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    dryrun = getattr(args, "dry_run_fit_train", False)

    # 统一从预处理好的 .npy 文件和 scaler 加载数据
    import joblib
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    X = np.load(os.path.join(DATA_DIR, 'X_seq.npy'))
    y_scaled = np.load(os.path.join(DATA_DIR, 'y_seq.npy'))
    scaler_y = joblib.load(os.path.join(DATA_DIR, 'scaler_y.joblib'))

    # 清理数据
    mask = (~np.isnan(X).any(axis=(1, 2)) & ~np.isinf(X).any(axis=(1, 2)) &
            ~np.isnan(y_scaled).any(axis=1) & ~np.isinf(y_scaled).any(axis=1))
    X, y_scaled = X[mask], y_scaled[mask]

    # 使用与训练时完全相同的划分获取测试集
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test_scaled = train_test_split(X, y_scaled, test_size=0.1, random_state=42)

    # 如果请求只预测部分测试集，则从测试集中采样固定随机子集（可复现）
    subset_frac = float(getattr(args, "subset_frac", 1.0))
    n_test_original = X_test.shape[0]
    if 0.0 < subset_frac < 1.0 and n_test_original > 0:
        n_keep = max(1, int(np.ceil(n_test_original * subset_frac)))
        rng = np.random.RandomState(42)
        idx = rng.choice(n_test_original, size=n_keep, replace=False)
        X_test = X_test[idx]
        y_test_scaled = y_test_scaled[idx]
    print(f"[Clean] 预测集样本数: {X_test.shape[0]} (subset_frac={subset_frac}, original={n_test_original})")

    if args.model == 'seq_ar_diffusion':
        feature_dim = X_test.shape[2]
        y_dim = 1
        y_samples_scaled = predict_seq_ar_diffusion(
            X_test, y_dim, device=args.device, n_samples=args.n_samples, batch_size=args.batch_size,
            hist_len=X_test.shape[1], seq_len=y_test_scaled.shape[1], feature_dim=feature_dim,
            hidden_dim=args.hidden_dim, timesteps=args.timesteps, schedule=args.schedule
        )
        print('y_samples_scaled range:', y_samples_scaled.min(), y_samples_scaled.max())
        N, n_samples, seq_len = y_samples_scaled.shape
        y_samples_scaled = np.clip(y_samples_scaled, 0.0, 1.0)
        y_samples = scaler_y.inverse_transform(y_samples_scaled.reshape(-1, seq_len)).reshape(N, n_samples, seq_len)
        print('y_samples range [W]:', y_samples.min(), y_samples.max())
        y_pred = np.mean(y_samples, axis=1)
        y_true = scaler_y.inverse_transform(y_test_scaled)
    elif args.model == 'seq_lstm':
        feature_dim = X_test.shape[2]
        y_dim = y_test_scaled.shape[1]
        y_pred_scaled = predict_seq_lstm(
            X_test, y_dim, device=args.device, batch_size=args.batch_size,
            hist_len=X_test.shape[1], seq_len=y_test_scaled.shape[1], feature_dim=feature_dim,
            hidden_dim=args.hidden_dim, dropout=args.dropout
        )
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_test_scaled)
        y_samples = None
    else:
        raise NotImplementedError(f"Prediction for model {args.model} is not adapted to the new data pipeline yet.")
        
#         # 在 predict.py 末尾、main() 里加 3 行调试打印
#         y_samples_scaled = predict_seq_ar_diffusion(...)   # 已有
#         print('y_samples_scaled range:', y_samples_scaled.min(), y_samples_scaled.max())
#         y_samples = scaler_y.inverse_transform(...)
#         print('y_samples range [W]:', y_samples.min(), y_samples.max())

    # 评估指标（反归一化后裁剪为非负，避免负功率导致失真）
    y_true_eval = np.maximum(y_true, 0.0)
    y_pred_eval = np.maximum(y_pred, 0.0)
    yt = y_true_eval.reshape(-1)
    yp = y_pred_eval.reshape(-1)

    # 基础指标
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))

    # 稳健 MAPE：对极小真值使用容量阈值裁剪（5% 容量），避免无意义爆炸
    try:
        # MinMaxScaler 持有 data_max_（shape: (1,)）
        y_cap = float(getattr(scaler_y, 'data_max_', [np.max(y_true_eval)])[0])
    except Exception:
        y_cap = float(np.max(y_true_eval))
    den = np.maximum(np.abs(yt), max(1e-8, 0.05 * y_cap))
    mape = float(np.mean(np.abs(yp - yt) / den))

    # sMAPE（对称 MAPE），对近零更稳健
    smape = float(np.mean(2.0 * np.abs(yp - yt) / (np.abs(yp) + np.abs(yt) + 1e-8)))

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape}

    # Save results
    suffix = "_dryrun" if dryrun else ""
    np.save(os.path.join(RESULTS_DIR, f'y_pred_{args.model}{suffix}.npy'), y_pred)
    if y_samples is not None:
        np.save(os.path.join(RESULTS_DIR, f'y_samples_{args.model}{suffix}.npy'), y_samples)
    np.save(os.path.join(RESULTS_DIR, f'y_true_{args.model}{suffix}.npy'), y_true)
    # 保存评估指标
    metrics_path = os.path.join(RESULTS_DIR, f'metrics_{args.model}{suffix}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # 只在首次保存通用 y_true.npy，避免被覆盖
    y_true_path = os.path.join(RESULTS_DIR, 'y_true.npy')
    if not os.path.exists(y_true_path):
        np.save(y_true_path, y_true)
    
    print(f"[Metrics] {args.model}{suffix} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f} | sMAPE: {smape:.4f}")
    print(f"Saved {args.model} predictions to {RESULTS_DIR} (dryrun={dryrun})")
    

if __name__ == "__main__":
    main()