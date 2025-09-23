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

from dataloader import load_data, preprocess_features, split_data
from diffusion import SimpleDiffusionModel, GaussianDiffusion
from baseline_gan import GANGenerator
from baseline_vae import VAE
from diffusion_weibull import SimpleDiffusionModelWeibull, WeibullDiffusion
from seq_vae import SeqVAE
from seq_diffusion import SeqCondEncoder, SeqDiffusionModel, SeqGaussianDiffusion
from seq_ar_diffusion import SeqCondEncoder as ARCondEncoder, SeqARDiffusionModel, SeqARGaussianDiffusion

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
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'diffusion_weibull_best.pth'), map_location=device))
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

def predict_seq_ar_diffusion(X_test, y_dim, device='cpu', n_samples=100, batch_size=32, hist_len=24, seq_len=24, feature_dim=9, hidden_dim=128):
    print("Generating predictions with SeqAR-Diffusion model...")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'seq_ar_diffusion_best.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 自动推断架构参数
    cond_state = checkpoint['cond_encoder']
    model_state = checkpoint['model']

    # LSTM: weight_ih_l0 shape is (4 * hidden_dim, input_size)
    cond_encoder_hidden_dim = cond_state['lstm.weight_ih_l0'].shape[0] // 4

    # MLP: 第一层 Linear 的 out_features = model hidden_dim
    model_hidden_dim = model_state['net.0.weight'].shape[0]

    # MLP 层数（隐藏层数 = Linear 层数 - 1）
    n_linear_layers = len([k for k in model_state.keys() if k.endswith('.weight')])
    model_n_layers = max(1, n_linear_layers - 1)

    # Dropout 默认 0.1
    model_dropout = 0.1

    # 构建模型
    cond_encoder = ARCondEncoder(feature_dim=feature_dim, hist_len=hist_len, hidden_dim=cond_encoder_hidden_dim).to(device)
    model = SeqARDiffusionModel(
        cond_dim=cond_encoder.hidden_dim,
        y_dim=y_dim,
        hidden_dim=model_hidden_dim,
        n_layers=model_n_layers,
        dropout=model_dropout
    ).to(device)

    try:
        cond_encoder.load_state_dict(checkpoint['cond_encoder'])
        model.load_state_dict(checkpoint['model'])
    except RuntimeError as e:
        print("Failed to load state_dict even after inferring hidden dims from checkpoint.")
        raise e

    cond_encoder.eval()
    model.eval()
    diffusion = SeqARGaussianDiffusion(model, cond_encoder, device=device, seq_len=seq_len)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    n_test = X_test.shape[0]
    y_samples_list = []
    with torch.no_grad():
        for i in tqdm(range(0, n_test, batch_size), desc="SeqAR-Diffusion Sampling"):
            X_batch = X_test_tensor[i:i+batch_size].to(device)
            batch_samples = []
            for _ in range(n_samples):
                samples = diffusion.sample(X_batch, seq_len=seq_len)  # (batch, seq_len)
                batch_samples.append(samples.cpu().numpy())
            batch_samples = np.stack(batch_samples, axis=1)  # (batch, n_samples, seq_len)
            y_samples_list.append(batch_samples)
    y_samples_all = np.concatenate(y_samples_list, axis=0)  # (N, n_samples, seq_len)
    # 关键：先裁剪到归一化区间，再反归一化
    y_samples_all = np.clip(y_samples_all, 0.0, 1.0)
    return y_samples_all

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
    parser.add_argument('--model', type=str, required=True, choices=['diffusion', 'gan', 'vae', 'weibull_diffusion', 'seq_vae', 'seq_diffusion', 'seq_ar_diffusion'])
    parser.add_argument('--n_samples', type=int, default=100, help="Number of samples to generate per data point.")
    parser.add_argument('--device', type=str, default=get_default_device(),
                        help="Device to run prediction on (e.g., 'cpu', 'cuda', 'mps'). "
                             "Automatically detects available hardware.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for prediction.")
    parser.add_argument('--dry_run_fit_train', action='store_true', help="Use all data as both train and test for dry-run overfit check.")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden dimension for LSTM/MLP (default: 128)")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    # Load and prepare data
    # 只用前 10000 条数据预测（可根据内存情况调整）
    df = load_data(max_samples=10000)
    dryrun = getattr(args, "dry_run_fit_train", False)
    if args.model == 'seq_vae':
        X, y_scaled, _, scaler_y = preprocess_features(df, seq_len=24, hist_len=24, seq_mode=True)
        if dryrun:
            X_test, y_test_scaled = X, y_scaled
        else:
            _, X_test, _, y_test_scaled = split_data(X, y_scaled)
        feature_dim = X.shape[-1]
        y_dim = 1
        y_samples_scaled = predict_seq_vae(X_test, y_dim, device=args.device, n_samples=args.n_samples, batch_size=args.batch_size, hist_len=24, seq_len=24, feature_dim=feature_dim)
        N, n_samples, seq_len = y_samples_scaled.shape
        y_samples_scaled = np.clip(y_samples_scaled, 0.0, 1.0)
        y_samples = scaler_y.inverse_transform(y_samples_scaled.reshape(-1, seq_len)).reshape(N, n_samples, seq_len)
        y_pred = np.mean(y_samples, axis=1)
        y_true = scaler_y.inverse_transform(y_test_scaled)
    elif args.model == 'seq_diffusion':
        X, y_scaled, _, scaler_y = preprocess_features(df, seq_len=24, hist_len=24, seq_mode=True)
        if dryrun:
            X_test, y_test_scaled = X, y_scaled
        else:
            _, X_test, _, y_test_scaled = split_data(X, y_scaled)
        feature_dim = X.shape[-1]
        y_dim = 24
        y_samples_scaled = predict_seq_diffusion(X_test, y_dim, device=args.device, n_samples=args.n_samples, batch_size=args.batch_size, hist_len=24, seq_len=24, feature_dim=feature_dim)
        N, n_samples, seq_len = y_samples_scaled.shape
        y_samples_scaled = np.clip(y_samples_scaled, 0.0, 1.0)
        y_samples = scaler_y.inverse_transform(y_samples_scaled.reshape(-1, seq_len)).reshape(N, n_samples, seq_len)
        y_pred = np.mean(y_samples, axis=1)
        y_true = scaler_y.inverse_transform(y_test_scaled)
    elif args.model == 'seq_ar_diffusion':
        X, y_scaled, _, scaler_y = preprocess_features(df, seq_len=24, hist_len=24, seq_mode=True)
        if dryrun:
            X_test, y_test_scaled = X, y_scaled
        else:
            _, X_test, _, y_test_scaled = split_data(X, y_scaled)
        feature_dim = X.shape[-1]
        y_dim = 1
        y_samples_scaled = predict_seq_ar_diffusion(
            X_test, y_dim, device=args.device, n_samples=args.n_samples, batch_size=args.batch_size,
            hist_len=24, seq_len=24, feature_dim=feature_dim, hidden_dim=args.hidden_dim
        )
        N, n_samples, seq_len = y_samples_scaled.shape
        y_samples_scaled = np.clip(y_samples_scaled, 0.0, 1.0)
        y_samples = scaler_y.inverse_transform(y_samples_scaled.reshape(-1, seq_len)).reshape(N, n_samples, seq_len)
        y_pred = np.mean(y_samples, axis=1)
        y_true = scaler_y.inverse_transform(y_test_scaled)
    else:
        X, y_scaled, _, scaler_y = preprocess_features(df, seq_len=24)
        if dryrun:
            X_test, y_test_scaled = X, y_scaled
        else:
            _, X_test, _, y_test_scaled = split_data(X, y_scaled)
        y_dim = y_test_scaled.shape[1]
        if args.model == 'diffusion':
            y_samples_scaled = predict_diffusion(X_test, y_dim, device=args.device, n_samples=args.n_samples, batch_size=args.batch_size)
        elif args.model == 'weibull_diffusion':
            y_samples_scaled = predict_weibull_diffusion(X_test, y_dim, device=args.device, n_samples=args.n_samples, batch_size=args.batch_size)
        elif args.model == 'gan':
            y_samples_scaled = predict_gan(X_test, y_dim, device=args.device, n_samples=args.n_samples, batch_size=args.batch_size)
        elif args.model == 'vae':
            y_samples_scaled = predict_vae(X_test, y_dim, device=args.device, n_samples=args.n_samples, batch_size=args.batch_size)
        else:
            raise ValueError("Invalid model choice")
        N, n_samples, seq_len = y_samples_scaled.shape
        y_samples_scaled = np.clip(y_samples_scaled, 0.0, 1.0)
        y_samples = scaler_y.inverse_transform(y_samples_scaled.reshape(-1, seq_len)).reshape(N, n_samples, seq_len)
        y_pred = np.mean(y_samples, axis=1)
        y_true = scaler_y.inverse_transform(y_test_scaled)

    # Save results
    suffix = "_dryrun" if dryrun else ""
    np.save(os.path.join(RESULTS_DIR, f'y_pred_{args.model}{suffix}.npy'), y_pred)
    np.save(os.path.join(RESULTS_DIR, f'y_samples_{args.model}{suffix}.npy'), y_samples)
    np.save(os.path.join(RESULTS_DIR, f'y_true_{args.model}{suffix}.npy'), y_true)
    # 只在首次保存通用 y_true.npy，避免被覆盖
    y_true_path = os.path.join(RESULTS_DIR, 'y_true.npy')
    if not os.path.exists(y_true_path):
        np.save(y_true_path, y_true)
    
    print(f"Saved {args.model} predictions to {RESULTS_DIR} (dryrun={dryrun})")

if __name__ == "__main__":
    main()