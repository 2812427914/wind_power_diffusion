import warnings
warnings.filterwarnings("ignore")
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from dataloader import load_data, preprocess_features, split_data, WindPowerCSVDataset
from diffusion import SimpleDiffusionModel, GaussianDiffusion
from baseline_gan import GANGenerator, GANDiscriminator
from baseline_vae import VAE
from diffusion_weibull import SimpleDiffusionModelWeibull, WeibullDiffusion

from seq_vae import SeqVAE
from seq_diffusion import SeqCondEncoder, SeqDiffusionModel, SeqGaussianDiffusion
from seq_ar_diffusion import SeqCondEncoder as ARCondEncoder, SeqARDiffusionModel, SeqARGaussianDiffusion

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '../checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_diffusion(X_train, y_train, epochs=10, batch_size=128, device='cpu'):
    from sklearn.model_selection import train_test_split
    cond_dim = X_train.shape[1]
    y_dim = y_train.shape[1]
    input_dim = cond_dim + y_dim + 1  # +1 for timestep
    model = SimpleDiffusionModel(input_dim=input_dim, output_dim=y_dim).to(device)
    diffusion = GaussianDiffusion(model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 划分验证集
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                           torch.tensor(y_tr, dtype=torch.float32)),
                              batch_size=256, shuffle=True, drop_last=False,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                           torch.tensor(y_val, dtype=torch.float32)),
                              batch_size=256, drop_last=False,
                              num_workers=4, pin_memory=False)

    best_val_loss = float('inf')
    best_epoch = -1
    patience = 15
    pat_counter = 0
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = diffusion.train_loss(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(diffusion.cond_encoder.parameters(), max_norm=5.0)
            optimizer.step()
            # OneCycleLR: step per batch
            scheduler.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val_ in val_loader:
                x_val, y_val_ = x_val.to(device), y_val_.to(device)
                val_loss = diffusion.train_loss(x_val, y_val_)
                val_losses.append(val_loss.item())
        avg_val_loss = float(np.mean(val_losses))
        print(f"[Diffusion] Epoch {epoch+1}/{epochs} Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'diffusion_best.pth'))
            print(f"Saved new best Diffusion model at epoch {best_epoch} with val loss {best_val_loss:.4f}")

    # 保存最后一个 epoch 的模型
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'diffusion_last.pth'))
    print(f"Saved last Diffusion model to {os.path.join(CHECKPOINT_DIR, 'diffusion_last.pth')}")

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
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(diffusion.cond_encoder.parameters(), max_norm=5.0)
            optimizer.step()
            # OneCycleLR: step per batch
            scheduler.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"[WeibullDiffusion] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'diffusion_weibull.pth'))
    print(f"Saved Weibull Diffusion model to {os.path.join(CHECKPOINT_DIR, 'diffusion_weibull.pth')}")

def train_gan(X_train, y_train, epochs=10, batch_size=128, device='cpu', latent_dim=32):
    from sklearn.model_selection import train_test_split
    cond_dim = X_train.shape[1]
    y_dim = y_train.shape[1]
    
    generator = GANGenerator(cond_dim=cond_dim, latent_dim=latent_dim, output_dim=y_dim).to(device)
    discriminator = GANDiscriminator(input_dim=cond_dim + y_dim).to(device)
    optim_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    
    # 划分验证集
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                           torch.tensor(y_tr, dtype=torch.float32)),
                              batch_size=256, shuffle=True, drop_last=False,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                           torch.tensor(y_val, dtype=torch.float32)),
                              batch_size=256, drop_last=False,
                              num_workers=4, pin_memory=False)

    best_val_loss = float('inf')
    best_epoch = -1
    patience = 15
    pat_counter = 0
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        for x_batch, y_batch in train_loader:
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

        # 验证
        generator.eval()
        discriminator.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val_ in val_loader:
                x_val, y_val_ = x_val.to(device), y_val_.to(device)
                z = torch.randn(x_val.shape[0], latent_dim).to(device)
                fake_y = generator(z, x_val)
                fake_data = torch.cat([x_val, fake_y], dim=1)
                d_fake = discriminator(fake_data)
                # 验证用判别器的 loss
                val_loss = -torch.mean(torch.log(d_fake + 1e-8)).item()
                val_losses.append(val_loss)
        avg_val_loss = float(np.mean(val_losses))
        print(f"[GAN] Epoch {epoch+1}/{epochs} D Loss: {loss_d.item():.4f} G Loss: {loss_g.item():.4f} | Val G Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_generator_best.pth'))
            torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_discriminator_best.pth'))
            print(f"Saved new best GAN generator/discriminator at epoch {best_epoch} with val G loss {best_val_loss:.4f}")

    torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_generator_last.pth'))
    torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_discriminator_last.pth'))
    print(f"Saved last GAN models to {CHECKPOINT_DIR}")

def train_vae(X_train, y_train, epochs=10, batch_size=128, device='cpu'):
    from sklearn.model_selection import train_test_split
    cond_dim = X_train.shape[1]
    y_dim = y_train.shape[1]
    
    model = VAE(cond_dim=cond_dim, y_dim=y_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 划分验证集
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                           torch.tensor(y_tr, dtype=torch.float32)),
                              batch_size=256, shuffle=True, drop_last=False,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                           torch.tensor(y_val, dtype=torch.float32)),
                              batch_size=256, drop_last=False,
                              num_workers=4, pin_memory=False)

    best_val_loss = float('inf')
    best_epoch = -1
    patience = 15
    pat_counter = 0
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            recon_y, mu, logvar = model(x_batch, y_batch)
            loss = model.loss_function(recon_y, y_batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # torch.nn.utils.clip_grad_norm_(cond_encoder.parameters(), max_norm=5.0)
            optimizer.step()
            # OneCycleLR: step per batch
            try:
                scheduler.step()
            except Exception:
                pass
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val_ in val_loader:
                x_val, y_val_ = x_val.to(device), y_val_.to(device)
                recon_y, mu, logvar = model(x_val, y_val_)
                val_loss = model.loss_function(recon_y, y_val_, mu, logvar)
                val_losses.append(val_loss.item())
        avg_val_loss = float(np.mean(val_losses))
        print(f"[VAE] Epoch {epoch+1}/{epochs} Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'vae_best.pth'))
            print(f"Saved new best VAE model at epoch {best_epoch} with val loss {best_val_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'vae_last.pth'))
    print(f"Saved last VAE model to {os.path.join(CHECKPOINT_DIR, 'vae_last.pth')}")


def train_seq_vae(X_train, y_train, epochs=10, batch_size=128, device='cpu', hist_len=24, seq_len=24, feature_dim=9):
    from sklearn.model_selection import train_test_split
    y_dim = 1  # 只预测功率
    # y_train shape: (N, seq_len) -> (N, seq_len, 1)
    y_train = y_train[..., None]
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                           torch.tensor(y_tr, dtype=torch.float32)),
                              batch_size=256, shuffle=True, drop_last=False,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                           torch.tensor(y_val, dtype=torch.float32)),
                              batch_size=256, drop_last=False,
                              num_workers=4, pin_memory=False)
    model = SeqVAE(feature_dim=feature_dim, hist_len=hist_len, y_dim=y_dim, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float('inf')
    best_epoch = -1
    patience = 15
    pat_counter = 0
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.squeeze(-1).to(device)
            recon_y, mu, logvar = model(x_batch)
            loss = model.loss_function(recon_y, y_batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(diffusion.cond_encoder.parameters(), max_norm=5.0)
            optimizer.step()
            # OneCycleLR: step per batch
            scheduler.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val_ in val_loader:
                x_val, y_val_ = x_val.to(device), y_val_.squeeze(-1).to(device)
                recon_y, mu, logvar = model(x_val)
                val_loss = model.loss_function(recon_y, y_val_, mu, logvar)
                val_losses.append(val_loss.item())
        avg_val_loss = float(np.mean(val_losses))
        print(f"[SeqVAE] Epoch {epoch+1}/{epochs} Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'seq_vae_best.pth'))
            print(f"Saved new best SeqVAE model at epoch {best_epoch} with val loss {best_val_loss:.4f}")
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'seq_vae_last.pth'))
    print(f"Saved last SeqVAE model to {os.path.join(CHECKPOINT_DIR, 'seq_vae_last.pth')}")

def train_seq_diffusion(X_train, y_train, epochs=10, batch_size=128, device='cpu', hist_len=24, seq_len=24, feature_dim=9):
    from sklearn.model_selection import train_test_split
    y_dim = seq_len
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                           torch.tensor(y_tr, dtype=torch.float32)),
                              batch_size=256, shuffle=True, drop_last=False,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                           torch.tensor(y_val, dtype=torch.float32)),
                              batch_size=256, drop_last=False,
                              num_workers=4, pin_memory=False)
    cond_encoder = SeqCondEncoder(feature_dim=feature_dim, hist_len=hist_len).to(device)
    model = SeqDiffusionModel(cond_dim=cond_encoder.hidden_dim, y_dim=y_dim).to(device)
    diffusion = SeqGaussianDiffusion(model, cond_encoder, device=device)
    # 使用更保守的优化策略
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cond_encoder.parameters()), 
        lr=1e-4,  # 降低学习率
        weight_decay=1e-4,
        betas=(0.9, 0.999)  # 使用默认的beta值
    )
    # 使用简单的学习率衰减
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_val_loss = float('inf')
    best_epoch = -1
    patience = 15
    pat_counter = 0
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        cond_encoder.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = diffusion.train_loss(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(diffusion.cond_encoder.parameters(), max_norm=5.0)
            optimizer.step()
            # OneCycleLR: step per batch
            scheduler.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        # 验证
        model.eval()
        cond_encoder.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val_ in val_loader:
                x_val, y_val_ = x_val.to(device), y_val_.to(device)
                val_loss = diffusion.train_loss(x_val, y_val_)
                val_losses.append(val_loss.item())
        avg_val_loss = float(np.mean(val_losses))
        print(f"[SeqDiffusion] Epoch {epoch+1}/{epochs} Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save({'cond_encoder': cond_encoder.state_dict(), 'model': model.state_dict()},
                       os.path.join(CHECKPOINT_DIR, 'seq_diffusion_best.pth'))
            print(f"Saved new best SeqDiffusion model at epoch {best_epoch} with val loss {best_val_loss:.4f}")
    torch.save({'cond_encoder': cond_encoder.state_dict(), 'model': model.state_dict()},
               os.path.join(CHECKPOINT_DIR, 'seq_diffusion_last.pth'))
    print(f"Saved last SeqDiffusion model to {os.path.join(CHECKPOINT_DIR, 'seq_diffusion_last.pth')}")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diffusion', choices=['diffusion', 'gan', 'vae', 'weibull_diffusion', 'seq_vae', 'seq_diffusion', 'seq_ar_diffusion'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default=get_default_device(),
                        help="Device to run training on (e.g., 'cpu', 'cuda', 'mps'). "
                             "Automatically detects available hardware.")
    parser.add_argument('--use_lazy_dataset', action='store_true', help="Use torch Dataset lazy loading for large CSV.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--max_samples', type=int, default=None, help="Only use the first N samples for quick test (lazy dataset mode).")
    parser.add_argument('--dry_run_fit_train', action='store_true', help="Use all data as both train and test for dry-run overfit check.")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    if args.use_lazy_dataset:
        # 使用懒加载 Dataset
        from torch.utils.data import random_split, DataLoader
        # 支持快速测试：只用前 N 条数据
        max_samples = getattr(args, "max_samples", None)
        # 针对序列模型自动切片
        if args.model in ['seq_vae', 'seq_diffusion', 'seq_ar_diffusion']:
            hist_len = 24
            seq_len = 24
            seq_mode = True
        else:
            hist_len = 1
            seq_len = 1
            seq_mode = False
        # 只用前 max_samples 条数据（如 1000），否则用全部
        if max_samples is not None:
            # 只取前 max_samples 条
            dataset = WindPowerCSVDataset(
                os.path.join(os.path.dirname(__file__), '../data/wtbdata_hourly.csv'),
                hist_len=hist_len, seq_len=seq_len, seq_mode=seq_mode
            )
            # 截断
            dataset = torch.utils.data.Subset(dataset, range(max_samples))
            n_total = len(dataset)
        else:
            dataset = WindPowerCSVDataset(
                os.path.join(os.path.dirname(__file__), '../data/wtbdata_hourly.csv'),
                hist_len=hist_len, seq_len=seq_len, seq_mode=seq_mode
            )
            n_total = len(dataset)
        n_val = int(n_total * 0.1)
        n_train = n_total - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"训练集样本数: {len(train_set)}")
        cond_dim = len(dataset.dataset.feature_cols) if isinstance(dataset, torch.utils.data.Subset) else len(dataset.feature_cols)
        y_dim = 1 if args.model in ['seq_vae', 'seq_ar_diffusion'] else seq_len if args.model == 'seq_diffusion' else 1

        if args.model == 'diffusion':
            input_dim = cond_dim + y_dim + 1
            model = SimpleDiffusionModel(input_dim=input_dim, output_dim=y_dim).to(args.device)
            diffusion = GaussianDiffusion(model, device=args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            best_val_loss = float('inf')
            best_epoch = -1
            for epoch in range(args.epochs):
                epoch_loss = 0
                model.train()
                for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} train"):
                    x_batch, y_batch = x_batch.float().to(args.device), y_batch.unsqueeze(1).float().to(args.device)
                    loss = diffusion.train_loss(x_batch, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_loader)
                # 验证
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for x_val, y_val_ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} val"):
                        x_val, y_val_ = x_val.float().to(args.device), y_val_.unsqueeze(1).float().to(args.device)
                        val_loss = diffusion.train_loss(x_val, y_val_)
                        val_losses.append(val_loss.item())
                avg_val_loss = float(np.mean(val_losses))
                print(f"[Diffusion-Lazy] Epoch {epoch+1}/{args.epochs} Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'diffusion_best.pth'))
                    print(f"Saved new best Diffusion model at epoch {best_epoch} with val loss {best_val_loss:.4f}")
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'diffusion_last.pth'))
            print(f"Saved last Diffusion model to {os.path.join(CHECKPOINT_DIR, 'diffusion_last.pth')}")

        elif args.model == 'gan':
            latent_dim = 32
            generator = GANGenerator(cond_dim=cond_dim, latent_dim=latent_dim, output_dim=y_dim).to(args.device)
            discriminator = GANDiscriminator(input_dim=cond_dim + y_dim).to(args.device)
            optim_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
            optim_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
            best_val_loss = float('inf')
            best_epoch = -1
            for epoch in range(args.epochs):
                generator.train()
                discriminator.train()
                for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} train"):
                    x_batch, y_batch = x_batch.float().to(args.device), y_batch.unsqueeze(1).float().to(args.device)
                    # Train Discriminator
                    z = torch.randn(x_batch.shape[0], latent_dim).float().to(args.device)
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
                # 验证
                generator.eval()
                discriminator.eval()
                val_losses = []
                with torch.no_grad():
                    for x_val, y_val_ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} val"):
                        x_val, y_val_ = x_val.float().to(args.device), y_val_.unsqueeze(1).float().to(args.device)
                        z = torch.randn(x_val.shape[0], latent_dim).float().to(args.device)
                        fake_y = generator(z, x_val)
                        fake_data = torch.cat([x_val, fake_y], dim=1)
                        d_fake = discriminator(fake_data)
                        val_loss = -torch.mean(torch.log(d_fake + 1e-8)).item()
                        val_losses.append(val_loss)
                avg_val_loss = float(np.mean(val_losses))
                print(f"[GAN-Lazy] Epoch {epoch+1}/{args.epochs} D Loss: {loss_d.item():.4f} G Loss: {loss_g.item():.4f} | Val G Loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_generator_best.pth'))
                    torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_discriminator_best.pth'))
                    print(f"Saved new best GAN generator/discriminator at epoch {best_epoch} with val G loss {best_val_loss:.4f}")
            torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_generator_last.pth'))
            torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_discriminator_last.pth'))
            print(f"Saved last GAN models to {CHECKPOINT_DIR}")

        elif args.model == 'vae':
            vae = VAE(cond_dim=cond_dim, y_dim=y_dim).to(args.device)
            optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
            best_val_loss = float('inf')
            best_epoch = -1
            for epoch in range(args.epochs):
                epoch_loss = 0
                vae.train()
                for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} train"):
                    x_batch, y_batch = x_batch.float().to(args.device), y_batch.unsqueeze(1).float().to(args.device)
                    recon_y, mu, logvar = vae(x_batch, y_batch)
                    loss = vae.loss_function(recon_y, y_batch, mu, logvar)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_loader)
                # 验证
                vae.eval()
                val_losses = []
                with torch.no_grad():
                    for x_val, y_val_ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} val"):
                        x_val, y_val_ = x_val.float().to(args.device), y_val_.unsqueeze(1).float().to(args.device)
                        recon_y, mu, logvar = vae(x_val, y_val_)
                        val_loss = vae.loss_function(recon_y, y_val_, mu, logvar)
                        val_losses.append(val_loss.item())
                avg_val_loss = float(np.mean(val_losses))
                print(f"[VAE-Lazy] Epoch {epoch+1}/{args.epochs} Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save(vae.state_dict(), os.path.join(CHECKPOINT_DIR, 'vae_best.pth'))
                    print(f"Saved new best VAE model at epoch {best_epoch} with val loss {best_val_loss:.4f}")
            torch.save(vae.state_dict(), os.path.join(CHECKPOINT_DIR, 'vae_last.pth'))
            print(f"Saved last VAE model to {os.path.join(CHECKPOINT_DIR, 'vae_last.pth')}")

        elif args.model == 'seq_vae':
            from seq_vae import SeqVAE
            feature_dim = cond_dim
            model = SeqVAE(feature_dim=feature_dim, hist_len=hist_len, y_dim=1, seq_len=seq_len).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            best_val_loss = float('inf')
            best_epoch = -1
            for epoch in range(args.epochs):
                epoch_loss = 0
                model.train()
                for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} train"):
                    x_batch, y_batch = x_batch.float().to(args.device), y_batch.float().to(args.device)
                    recon_y, mu, logvar = model(x_batch)
                    loss = model.loss_function(recon_y, y_batch, mu, logvar)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_loader)
                # 验证
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for x_val, y_val_ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} val"):
                        x_val, y_val_ = x_val.float().to(args.device), y_val_.float().to(args.device)
                        recon_y, mu, logvar = model(x_val)
                        val_loss = model.loss_function(recon_y, y_val_, mu, logvar)
                        val_losses.append(val_loss.item())
                avg_val_loss = float(np.mean(val_losses))
                print(f"[SeqVAE-Lazy] Epoch {epoch+1}/{args.epochs} Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'seq_vae_best.pth'))
                    print(f"Saved new best SeqVAE model at epoch {best_epoch} with val loss {best_val_loss:.4f}")
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'seq_vae_last.pth'))
            print(f"Saved last SeqVAE model to {os.path.join(CHECKPOINT_DIR, 'seq_vae_last.pth')}")

        elif args.model == 'seq_diffusion':
            from seq_diffusion import SeqCondEncoder, SeqDiffusionModel, SeqGaussianDiffusion
            feature_dim = cond_dim
            cond_encoder = SeqCondEncoder(feature_dim=feature_dim, hist_len=hist_len).to(args.device)
            model = SeqDiffusionModel(cond_dim=cond_encoder.hidden_dim, y_dim=seq_len).to(args.device)
            diffusion = SeqGaussianDiffusion(model, cond_encoder, device=args.device)
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            optimizer = torch.optim.Adam(list(model.parameters()) + list(cond_encoder.parameters()), lr=1e-3)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
            best_val_loss = float('inf')
            best_epoch = -1
            for epoch in range(args.epochs):
                epoch_loss = 0
                model.train()
                cond_encoder.train()
                for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} train"):
                    x_batch, y_batch = x_batch.float().to(args.device), y_batch.float().to(args.device)
                    loss = diffusion.train_loss(x_batch, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_loader)
                # 验证
                model.eval()
                cond_encoder.eval()
                val_losses = []
                with torch.no_grad():
                    for x_val, y_val_ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} val"):
                        x_val, y_val_ = x_val.float().to(args.device), y_val_.float().to(args.device)
                        val_loss = diffusion.train_loss(x_val, y_val_)
                        val_losses.append(val_loss.item())
                avg_val_loss = float(np.mean(val_losses))
                print(f"[SeqDiffusion-Lazy] Epoch {epoch+1}/{args.epochs} Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save({'cond_encoder': cond_encoder.state_dict(), 'model': model.state_dict()}, os.path.join(CHECKPOINT_DIR, 'seq_diffusion_best.pth'))
                    print(f"Saved new best SeqDiffusion model at epoch {best_epoch} with val loss {best_val_loss:.4f}")
            torch.save({'cond_encoder': cond_encoder.state_dict(), 'model': model.state_dict()}, os.path.join(CHECKPOINT_DIR, 'seq_diffusion_last.pth'))
            print(f"Saved last SeqDiffusion model to {os.path.join(CHECKPOINT_DIR, 'seq_diffusion_last.pth')}")

        elif args.model == 'seq_ar_diffusion':
            from seq_ar_diffusion import SeqCondEncoder as ARCondEncoder, SeqARDiffusionModel, SeqARGaussianDiffusion
            feature_dim = cond_dim
            cond_encoder = ARCondEncoder(feature_dim=feature_dim, hist_len=hist_len).to(args.device)
            # 自回归：把上一时刻y拼到条件中 => cond_dim + 1
            model = SeqARDiffusionModel(cond_dim=cond_encoder.hidden_dim + 1, y_dim=1).to(args.device)
            diffusion = SeqARGaussianDiffusion(model, cond_encoder, device=args.device, seq_len=seq_len)
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            optimizer = torch.optim.Adam(list(model.parameters()) + list(cond_encoder.parameters()), lr=1e-3)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
            best_val_loss = float('inf')
            best_epoch = -1
            for epoch in range(args.epochs):
                epoch_loss = 0
                model.train()
                cond_encoder.train()
                for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} train"):
                    x_batch, y_batch = x_batch.float().to(args.device), y_batch.float().to(args.device)
                    loss = diffusion.train_loss(x_batch, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_loader)
                # 验证
                model.eval()
                cond_encoder.eval()
                val_losses = []
                with torch.no_grad():
                    for x_val, y_val_ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} val"):
                        x_val, y_val_ = x_val.float().to(args.device), y_val_.float().to(args.device)
                        val_loss = diffusion.train_loss(x_val, y_val_)
                        val_losses.append(val_loss.item())
                avg_val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
                print(f"[SeqAR-Diffusion-Lazy] Epoch {epoch+1}/{args.epochs} Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save({'cond_encoder': cond_encoder.state_dict(), 'model': model.state_dict()}, os.path.join(CHECKPOINT_DIR, 'seq_ar_diffusion_best.pth'))
                    print(f"Saved new best SeqAR-Diffusion model at epoch {best_epoch} with val loss {best_val_loss:.4f}")
            torch.save({'cond_encoder': cond_encoder.state_dict(), 'model': model.state_dict()}, os.path.join(CHECKPOINT_DIR, 'seq_ar_diffusion_last.pth'))
            print(f"Saved last SeqAR-Diffusion model to {os.path.join(CHECKPOINT_DIR, 'seq_ar_diffusion_last.pth')}")

        elif args.model == 'gan':
            latent_dim = 32
            generator = GANGenerator(cond_dim=cond_dim, latent_dim=latent_dim, output_dim=y_dim).to(args.device)
            discriminator = GANDiscriminator(input_dim=cond_dim + y_dim).to(args.device)
            optim_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
            optim_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
            best_val_loss = float('inf')
            best_epoch = -1
            for epoch in range(args.epochs):
                generator.train()
                discriminator.train()
                for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} train"):
                    x_batch, y_batch = x_batch.float().to(args.device), y_batch.unsqueeze(1).float().to(args.device)
                    # Train Discriminator
                    z = torch.randn(x_batch.shape[0], latent_dim).float().to(args.device)
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
                # 验证
                generator.eval()
                discriminator.eval()
                val_losses = []
                with torch.no_grad():
                    for x_val, y_val_ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} val"):
                        x_val, y_val_ = x_val.float().to(args.device), y_val_.unsqueeze(1).float().to(args.device)
                        z = torch.randn(x_val.shape[0], latent_dim).float().to(args.device)
                        fake_y = generator(z, x_val)
                        fake_data = torch.cat([x_val, fake_y], dim=1)
                        d_fake = discriminator(fake_data)
                        val_loss = -torch.mean(torch.log(d_fake + 1e-8)).item()
                        val_losses.append(val_loss)
                avg_val_loss = float(np.mean(val_losses))
                print(f"[GAN-Lazy] Epoch {epoch+1}/{args.epochs} D Loss: {loss_d.item():.4f} G Loss: {loss_g.item():.4f} | Val G Loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_generator_best.pth'))
                    torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_discriminator_best.pth'))
                    print(f"Saved new best GAN generator/discriminator at epoch {best_epoch} with val G loss {best_val_loss:.4f}")
            torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_generator_last.pth'))
            torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, 'gan_discriminator_last.pth'))
            print(f"Saved last GAN models to {CHECKPOINT_DIR}")

        elif args.model == 'vae':
            vae = VAE(cond_dim=cond_dim, y_dim=y_dim).to(args.device)
            optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
            best_val_loss = float('inf')
            best_epoch = -1
            for epoch in range(args.epochs):
                epoch_loss = 0
                vae.train()
                for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} train"):
                    x_batch, y_batch = x_batch.float().to(args.device), y_batch.unsqueeze(1).float().to(args.device)
                    recon_y, mu, logvar = vae(x_batch, y_batch)
                    loss = vae.loss_function(recon_y, y_batch, mu, logvar)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_loader)
                # 验证
                vae.eval()
                val_losses = []
                with torch.no_grad():
                    for x_val, y_val_ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} val"):
                        x_val, y_val_ = x_val.float().to(args.device), y_val_.unsqueeze(1).float().to(args.device)
                        recon_y, mu, logvar = vae(x_val, y_val_)
                        val_loss = vae.loss_function(recon_y, y_val_, mu, logvar)
                        val_losses.append(val_loss.item())
                avg_val_loss = float(np.mean(val_losses))
                print(f"[VAE-Lazy] Epoch {epoch+1}/{args.epochs} Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    torch.save(vae.state_dict(), os.path.join(CHECKPOINT_DIR, 'vae_best.pth'))
                    print(f"Saved new best VAE model at epoch {best_epoch} with val loss {best_val_loss:.4f}")
            torch.save(vae.state_dict(), os.path.join(CHECKPOINT_DIR, 'vae_last.pth'))
            print(f"Saved last VAE model to {os.path.join(CHECKPOINT_DIR, 'vae_last.pth')}")

        # 懒加载 SeqDiffusion/SeqAR-Diffusion/SeqVAE 可扩展，需支持序列切片

    else:
        # 只用前 10000 条数据训练（可根据内存情况调整）
        df = load_data(max_samples=10000)
        if args.model == 'seq_vae':
            import numpy as np
            X = np.load(os.path.join(os.path.dirname(__file__), '../data/X_seq.npy'))
            y = np.load(os.path.join(os.path.dirname(__file__), '../data/y_seq.npy'))
            # 1. 长度对齐
            min_samples = min(len(X), len(y))
            X = X[:min_samples]
            y = y[:min_samples]
            # 2. 过滤异常样本
            mask = (~np.isnan(X).any(axis=(1, 2)) & ~np.isinf(X).any(axis=(1, 2)) &
                    ~np.isnan(y).any(axis=1) & ~np.isinf(y).any(axis=1))
            X, y = X[mask], y[mask]
            print(f"[Clean] 剩余样本数: {len(X)}")
            # 3. MinMax 归一化
            from sklearn.preprocessing import MinMaxScaler
            from dataloader import split_data
            X_2d = X.reshape(-1, X.shape[-1])
            scaler_x = MinMaxScaler()
            X_scaled_2d = scaler_x.fit_transform(X_2d)
            X_scaled = X_scaled_2d.reshape(X.shape)
            y_flat = y.reshape(-1, 1)
            scaler_y = MinMaxScaler()
            y_scaled_flat = scaler_y.fit_transform(y_flat)
            y_scaled = y_scaled_flat.reshape(y.shape)
            # 4. 断言确保干净
            assert not np.isnan(X_scaled).any(), "X_scaled 仍含 nan"
            assert not np.isinf(X_scaled).any(), "X_scaled 仍含 inf"
            assert not np.isnan(y_scaled).any(), "y_scaled 仍含 nan"
            assert not np.isinf(y_scaled).any(), "y_scaled 仍含 inf"
            # 5. dry-run: 用全部数据做训练和评估（可选 max_samples 截断）
            if getattr(args, "dry_run_fit_train", False):
                max_samples = getattr(args, "max_samples", None)
                if max_samples is not None:
                    X_train, y_train = X_scaled[:max_samples], y_scaled[:max_samples]
                    X_test, y_test = X_scaled[:max_samples], y_scaled[:max_samples]
                    print(f"[Dry-run] 使用前 {max_samples} 条数据做训练和评估，样本数: {X_train.shape[0]}")
                else:
                    X_train, y_train = X_scaled, y_scaled
                    X_test, y_test = X_scaled, y_scaled
                    print(f"[Dry-run] 使用全部数据做训练和评估，样本数: {X_train.shape[0]}")
            else:
                X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
                print(f"训练集样本数: {X_train.shape[0]}")
            feature_dim = X.shape[-1]
            train_seq_vae(X_train, y_train, epochs=args.epochs, device=args.device, hist_len=24, seq_len=24, feature_dim=feature_dim)
        elif args.model == 'seq_diffusion':
            import numpy as np
            X = np.load(os.path.join(os.path.dirname(__file__), '../data/X_seq.npy'))
            y = np.load(os.path.join(os.path.dirname(__file__), '../data/y_seq.npy'))
            min_samples = min(len(X), len(y))
            X = X[:min_samples]
            y = y[:min_samples]
            mask = (~np.isnan(X).any(axis=(1, 2)) & ~np.isinf(X).any(axis=(1, 2)) &
                    ~np.isnan(y).any(axis=1) & ~np.isinf(y).any(axis=1))
            X, y = X[mask], y[mask]
            print(f"[Clean] 剩余样本数: {len(X)}")
            from sklearn.preprocessing import MinMaxScaler
            from dataloader import split_data
            X_2d = X.reshape(-1, X.shape[-1])
            scaler_x = MinMaxScaler()
            X_scaled_2d = scaler_x.fit_transform(X_2d)
            X_scaled = X_scaled_2d.reshape(X.shape)
            y_flat = y.reshape(-1, 1)
            scaler_y = MinMaxScaler()
            y_scaled_flat = scaler_y.fit_transform(y_flat)
            y_scaled = y_scaled_flat.reshape(y.shape)
            assert not np.isnan(X_scaled).any(), "X_scaled 仍含 nan"
            assert not np.isinf(X_scaled).any(), "X_scaled 仍含 inf"
            assert not np.isnan(y_scaled).any(), "y_scaled 仍含 nan"
            assert not np.isinf(y_scaled).any(), "y_scaled 仍含 inf"
            X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
            print(f"训练集样本数: {X_train.shape[0]}")
            feature_dim = X.shape[-1]
            train_seq_diffusion(X_train, y_train, epochs=args.epochs, device=args.device, hist_len=24, seq_len=24, feature_dim=feature_dim)
        elif args.model == 'seq_ar_diffusion':
            import numpy as np
            X = np.load(os.path.join(os.path.dirname(__file__), '../data/X_seq.npy'))
            y = np.load(os.path.join(os.path.dirname(__file__), '../data/y_seq.npy'))
            min_samples = min(len(X), len(y))
            X = X[:min_samples]
            y = y[:min_samples]
            mask = (~np.isnan(X).any(axis=(1, 2)) & ~np.isinf(X).any(axis=(1, 2)) &
                    ~np.isnan(y).any(axis=1) & ~np.isinf(y).any(axis=1))
            X, y = X[mask], y[mask]
            print(f"[Clean] 剩余样本数: {len(X)}")
            from sklearn.preprocessing import MinMaxScaler
            from dataloader import split_data
            X_2d = X.reshape(-1, X.shape[-1])
            scaler_x = MinMaxScaler()
            X_scaled_2d = scaler_x.fit_transform(X_2d)
            X_scaled = X_scaled_2d.reshape(X.shape)
            y_flat = y.reshape(-1, 1)
            scaler_y = MinMaxScaler()
            y_scaled_flat = scaler_y.fit_transform(y_flat)
            y_scaled = y_scaled_flat.reshape(y.shape)
            assert not np.isnan(X_scaled).any(), "X_scaled 仍含 nan"
            assert not np.isinf(X_scaled).any(), "X_scaled 仍含 inf"
            assert not np.isnan(y_scaled).any(), "y_scaled 仍含 nan"
            assert not np.isinf(y_scaled).any(), "y_scaled 仍含 inf"
            # dry-run: 用全部数据做训练和评估
            if getattr(args, "dry_run_fit_train", False):
                X_train, y_train = X_scaled, y_scaled
                X_test, y_test = X_scaled, y_scaled
                print(f"[Dry-run] 使用全部数据做训练和评估，样本数: {X_train.shape[0]}")
            else:
                X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
                print(f"训练集样本数: {X_train.shape[0]}")
            feature_dim = X.shape[-1]
            train_seq_ar_diffusion(X_train, y_train, epochs=args.epochs, device=args.device, hist_len=24, seq_len=24, feature_dim=feature_dim)
        else:
            X, y, _, _ = preprocess_features(df, seq_len=24)
            # dry-run: 用全部数据做训练和评估
            if getattr(args, "dry_run_fit_train", False):
                X_train, y_train = X, y
                X_test, y_test = X, y
                print(f"[Dry-run] 使用全部数据做训练和评估，样本数: {X_train.shape[0]}")
            else:
                X_train, X_test, y_train, y_test = split_data(X, y)
                print(f"训练集样本数: {X_train.shape[0]}")
            if args.model == 'diffusion':
                train_diffusion(X_train, y_train, epochs=args.epochs, device=args.device)
            elif args.model == 'weibull_diffusion':
                train_weibull_diffusion(X_train, y_train, epochs=args.epochs, device=args.device)
            elif args.model == 'gan':
                train_gan(X_train, y_train, epochs=args.epochs, device=args.device)
            elif args.model == 'vae':
                train_vae(X_train, y_train, epochs=args.epochs, device=args.device)
