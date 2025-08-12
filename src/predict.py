import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from dataloader import load_data, preprocess_features, split_data
from diffusion import SimpleDiffusionModel, GaussianDiffusion
from baseline_gan import GANGenerator
from baseline_vae import VAE
from diffusion_weibull import SimpleDiffusionModelWeibull, WeibullDiffusion

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
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'diffusion.pth'), map_location=device))
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
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'diffusion_weibull.pth'), map_location=device))
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
    generator.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'gan_generator.pth'), map_location=device))
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
    vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'vae.pth'), map_location=device))
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

def get_default_device():
    """Pick GPU if available, else CPU. Priority: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    # Check for Apple Silicon (M1/M2)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    parser = argparse.ArgumentParser(description="Generate predictions from a trained model.")
    parser.add_argument('--model', type=str, required=True, choices=['diffusion', 'gan', 'vae', 'weibull_diffusion'])
    parser.add_argument('--n_samples', type=int, default=100, help="Number of samples to generate per data point.")
    parser.add_argument('--device', type=str, default=get_default_device(),
                        help="Device to run prediction on (e.g., 'cpu', 'cuda', 'mps'). "
                             "Automatically detects available hardware.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for prediction.")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    # Load and prepare data
    # 只用前 10000 条数据预测（可根据内存情况调整）
    df = load_data(max_samples=10000)
    X, y_scaled, _, scaler_y = preprocess_features(df, seq_len=24)
    _, X_test, _, y_test_scaled = split_data(X, y_scaled)
    y_dim = y_test_scaled.shape[1]

    # Generate predictions based on model choice
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

    # Inverse transform to original scale
    # y_samples_scaled: (N, n_samples, 24)
    N, n_samples, seq_len = y_samples_scaled.shape
    y_samples = scaler_y.inverse_transform(y_samples_scaled.reshape(-1, seq_len)).reshape(N, n_samples, seq_len)
    y_pred = np.mean(y_samples, axis=1)  # (N, 24)
    y_true = scaler_y.inverse_transform(y_test_scaled)  # (N, 24)

    # Save results
    np.save(os.path.join(RESULTS_DIR, f'y_pred_{args.model}.npy'), y_pred)
    np.save(os.path.join(RESULTS_DIR, f'y_samples_{args.model}.npy'), y_samples)
    np.save(os.path.join(RESULTS_DIR, 'y_true.npy'), y_true)
    
    print(f"Saved {args.model} predictions to {RESULTS_DIR}")

if __name__ == "__main__":
    main()