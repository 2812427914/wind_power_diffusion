import argparse
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
from dataloader import load_data, preprocess_features, split_data
from diffusion import SimpleDiffusionModel, GaussianDiffusion
from baseline_gan import GANGenerator, GANDiscriminator
from baseline_vae import VAE
from diffusion_weibull import SimpleDiffusionModelWeibull, WeibullDiffusion

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '../checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_diffusion(X_train, y_train, epochs=10, batch_size=128, device='cpu'):
    cond_dim = X_train.shape[1]
    y_dim = y_train.shape[1]
    input_dim = cond_dim + y_dim + 1  # +1 for timestep
    model = SimpleDiffusionModel(input_dim=input_dim, output_dim=y_dim).to(device)
    diffusion = GaussianDiffusion(model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = diffusion.train_loss(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"[Diffusion] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'diffusion.pth'))
    print(f"Saved Diffusion model to {os.path.join(CHECKPOINT_DIR, 'diffusion.pth')}")

def train_weibull_diffusion(X_train, y_train, epochs=10, batch_size=128, device='cpu', weibull_shape=2.0):
    cond_dim = X_train.shape[1]
    y_dim = y_train.shape[1]
    input_dim = cond_dim + y_dim + 1  # +1 for timestep
    model = SimpleDiffusionModelWeibull(input_dim=input_dim, output_dim=y_dim).to(device)
    diffusion = WeibullDiffusion(model, device=device, weibull_shape=weibull_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = diffusion.train_loss(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"[WeibullDiffusion] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'diffusion_weibull.pth'))
    print(f"Saved Weibull Diffusion model to {os.path.join(CHECKPOINT_DIR, 'diffusion_weibull.pth')}")

def train_gan(X_train, y_train, epochs=10, batch_size=128, device='cpu', latent_dim=32):
    cond_dim = X_train.shape[1]
    y_dim = y_train.shape[1]
    
    generator = GANGenerator(cond_dim=cond_dim, latent_dim=latent_dim, output_dim=y_dim).to(device)
    discriminator = GANDiscriminator(input_dim=cond_dim + y_dim).to(device)
    optim_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Train Discriminator
            z = torch.randn(x_batch.shape[0], latent_dim).to(device)
            fake_y = generator(z, x_batch)
            real_data = torch.cat([x_batch, y_batch], dim=1)
            fake_data = torch.cat([x_batch, fake_y], dim=1)
            
            d_real = discriminator(real_data)
            d_fake = discriminator(fake_data.detach())
            loss_d = -torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))
            
            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()
            
            # Train Generator
            d_fake = discriminator(fake_data)
            loss_g = -torch.mean(torch.log(d_fake + 1e-8))
            
            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

        print(f"[GAN] Epoch {epoch+1}/{epochs} D Loss: {loss_d.item():.4f} G Loss: {loss_g.item():.4f}")

    torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_discriminator.pth'))
    print(f"Saved GAN models to {CHECKPOINT_DIR}")

def train_vae(X_train, y_train, epochs=10, batch_size=128, device='cpu'):
    cond_dim = X_train.shape[1]
    y_dim = y_train.shape[1]
    
    vae = VAE(cond_dim=cond_dim, y_dim=y_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            recon_y, mu, logvar = vae(x_batch, y_batch)
            loss = vae.loss_function(recon_y, y_batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"[VAE] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
    
    torch.save(vae.state_dict(), os.path.join(CHECKPOINT_DIR, 'vae.pth'))
    print(f"Saved VAE model to {os.path.join(CHECKPOINT_DIR, 'vae.pth')}")


def get_default_device():
    """Pick GPU if available, else CPU. Priority: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    # Check for Apple Silicon (M1/M2)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diffusion', choices=['diffusion', 'gan', 'vae', 'weibull_diffusion'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default=get_default_device(),
                        help="Device to run training on (e.g., 'cpu', 'cuda', 'mps'). "
                             "Automatically detects available hardware.")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    # 只用前 10000 条数据训练（可根据内存情况调整）
    df = load_data(max_samples=10000)
    X, y, _, _ = preprocess_features(df, seq_len=24)
    X_train, X_test, y_train, y_test = split_data(X, y)

    if args.model == 'diffusion':
        train_diffusion(X_train, y_train, epochs=args.epochs, device=args.device)
    elif args.model == 'weibull_diffusion':
        train_weibull_diffusion(X_train, y_train, epochs=args.epochs, device=args.device)
    elif args.model == 'gan':
        train_gan(X_train, y_train, epochs=args.epochs, device=args.device)
    elif args.model == 'vae':
        train_vae(X_train, y_train, epochs=args.epochs, device=args.device)