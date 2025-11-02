import warnings
warnings.filterwarnings("ignore")
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
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
from seq_lstm import SeqLSTMModel

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

def train_seq_lstm(X_train, y_train, epochs=10, batch_size=128, device='cpu', hist_len=24, seq_len=24, feature_dim=1, hidden_dim=128, dropout=0.1):
    from sklearn.model_selection import train_test_split
    y_dim = seq_len
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                           torch.tensor(y_tr, dtype=torch.float32)),
                              batch_size=batch_size, shuffle=True, drop_last=False,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                           torch.tensor(y_val, dtype=torch.float32)),
                              batch_size=batch_size, drop_last=False,
                              num_workers=4, pin_memory=False)
    model = SeqLSTMModel(feature_dim=feature_dim, hist_len=hist_len, seq_len=seq_len, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float('inf')
    best_epoch = -1
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = model.loss_function(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val_ in val_loader:
                x_val, y_val_ = x_val.to(device), y_val_.to(device)
                val_loss = model.loss_function(x_val, y_val_)
                val_losses.append(val_loss.item())
        avg_val_loss = float(np.mean(val_losses))
        print(f"[SeqLSTM] Epoch {epoch+1}/{epochs} Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save({'model': model.state_dict(),
                        'meta': {
                            'feature_dim': feature_dim,
                            'hist_len': hist_len,
                            'seq_len': seq_len,
                            'hidden_dim': hidden_dim,
                            'dropout': dropout,
                        }},
                       os.path.join(CHECKPOINT_DIR, 'seq_lstm_best.pth'))
            print(f"Saved new best SeqLSTM model at epoch {best_epoch} with val loss {best_val_loss:.4f}")

            # ====== 计算并打印验证集指标 ======
            model.eval()
            y_true_list = []
            y_pred_list = []
            for x_val, y_val_ in val_loader:
                x_val = x_val.to(device)
                y_val_ = y_val_.to(device)
                with torch.no_grad():
                    y_pred = model(x_val).cpu().numpy()
                    y_true = y_val_.cpu().numpy()
                    y_true_list.append(y_true)
                    y_pred_list.append(y_pred)
            import joblib
            DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
            scaler_y = joblib.load(os.path.join(DATA_DIR, 'scaler_y.joblib'))
            y_true_all = np.concatenate(y_true_list, axis=0)
            y_pred_all = np.concatenate(y_pred_list, axis=0)
            # 先裁剪到归一化区间，避免反归一化后出现异常
            y_pred_all = np.clip(y_pred_all, 0.0, 1.0)
            # 反归一化
            y_true_inv = scaler_y.inverse_transform(y_true_all)
            y_pred_inv = scaler_y.inverse_transform(y_pred_all)
            # 物理约束：功率非负
            y_true_eval = np.clip(y_true_inv, 0.0, None)
            y_pred_eval = np.clip(y_pred_inv, 0.0, None)
            # 展平并过滤非有限值
            yt = y_true_eval.reshape(-1)
            yp = y_pred_eval.reshape(-1)
            finite_mask = np.isfinite(yt) & np.isfinite(yp)
            yt = yt[finite_mask]
            yp = yp[finite_mask]
            mae = float(np.mean(np.abs(yp - yt))) if yt.size > 0 else float('nan')
            rmse = float(np.sqrt(np.mean((yp - yt) ** 2))) if yt.size > 0 else float('nan')
            # 稳健 MAPE：容量阈值取 scaler 的最大值（所有时间步最大）
            try:
                data_max = getattr(scaler_y, 'data_max_', None)
                if data_max is not None:
                    y_cap = float(np.max(data_max))
                else:
                    y_cap = float(np.max(y_true_eval))
            except Exception:
                y_cap = float(np.max(y_true_eval))
            den = np.maximum(np.abs(yt), max(1e-8, 0.05 * y_cap))
            mape = float(np.mean(np.abs(yp - yt) / den)) if yt.size > 0 else float('nan')
            smape = float(np.mean(2.0 * np.abs(yp - yt) / (np.abs(yp) + np.abs(yt) + 1e-8))) if yt.size > 0 else float('nan')
            print(f"[Best SeqLSTM Metrics] Epoch {best_epoch} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.4f} | sMAPE: {smape:.4f}")

    torch.save({'model': model.state_dict(),
                'meta': {
                    'feature_dim': feature_dim,
                    'hist_len': hist_len,
                    'seq_len': seq_len,
                    'hidden_dim': hidden_dim,
                    'dropout': dropout,
                }},
               os.path.join(CHECKPOINT_DIR, 'seq_lstm_last.pth'))
    print(f"Saved last SeqLSTM model to {os.path.join(CHECKPOINT_DIR, 'seq_lstm_last.pth')}")


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
    parser.add_argument('--model', type=str, default='diffusion', choices=['diffusion', 'gan', 'vae', 'weibull_diffusion', 'seq_vae', 'seq_diffusion', 'seq_ar_diffusion', 'seq_lstm'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default=get_default_device(),
                        help="Device to run training on (e.g., 'cpu', 'cuda', 'mps'). "
                             "Automatically detects available hardware.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--max_samples', type=int, default=None, help="Only use the first N samples for quick test (lazy dataset mode).")
    parser.add_argument('--dry_run_fit_train', action='store_true', help="Use all data as both train and test for dry-run overfit check.")
    parser.add_argument('--workers', type=int, default=4, help="Number of DataLoader worker processes for faster data loading.")
    # SeqAR-Diffusion 相关超参
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension for SeqAR-Diffusion MLP/LSTM.")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate for SeqAR-Diffusion MLP.")
    parser.add_argument('--t_embed_dim', type=int, default=32, help="Time embedding dimension (sinusoidal).")
    parser.add_argument('--timesteps', type=int, default=300, help="Number of diffusion timesteps.")
    parser.add_argument('--beta_start', type=float, default=1e-4, help="Beta start for diffusion (linear schedule).")
    parser.add_argument('--beta_end', type=float, default=0.02, help="Beta end for diffusion (linear schedule).")
    parser.add_argument('--schedule', type=str, default='cosine', choices=['linear', 'cosine'], help="Noise schedule type.")
    parser.add_argument('--k_steps', type=int, default=4, help="Number of random AR steps per batch for training.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for optimizer.")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # 统一从预处理好的 .npy 文件加载数据（优先使用时间一致的预分割 train/val）
    DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
    train_X_path = os.path.join(DATA_DIR, 'X_seq_train.npy')
    train_y_path = os.path.join(DATA_DIR, 'y_seq_train.npy')
    val_X_path   = os.path.join(DATA_DIR, 'X_seq_val.npy')
    val_y_path   = os.path.join(DATA_DIR, 'y_seq_val.npy')

    pre_split_available = (os.path.exists(train_X_path) and os.path.exists(train_y_path)
                           and os.path.exists(val_X_path) and os.path.exists(val_y_path))

    if pre_split_available:
        X_train = np.load(train_X_path)
        y_train = np.load(train_y_path)
        X_val   = np.load(val_X_path)
        y_val   = np.load(val_y_path)
        # 预分割数据已在构建时清洗与缩放，仍做一次健壮性检查
        def _clean_once(X, y):
            mask = (~np.isnan(X).any(axis=(1, 2)) & ~np.isinf(X).any(axis=(1, 2)) &
                    ~np.isnan(y).any(axis=1) & ~np.isinf(y).any(axis=1))
            return X[mask], y[mask]
        X_train, y_train = _clean_once(X_train, y_train)
        X_val, y_val     = _clean_once(X_val, y_val)
        print(f"Loaded pre-split data. Train X: {X_train.shape}, Train y: {y_train.shape} | Val X: {X_val.shape}, Val y: {y_val.shape}")
    else:
        # 回退到合并版，并使用随机划分（不推荐用于“预测未来”，仅兼容）
        X = np.load(os.path.join(DATA_DIR, 'X_seq.npy'))
        y = np.load(os.path.join(DATA_DIR, 'y_seq.npy'))
        # 清理 NaN/inf 值
        mask = (~np.isnan(X).any(axis=(1, 2)) & ~np.isinf(X).any(axis=(1, 2)) &
                ~np.isnan(y).any(axis=1) & ~np.isinf(y).any(axis=1))
        X, y = X[mask], y[mask]
        print(f"Loaded and cleaned data. X shape: {X.shape}, y shape: {y.shape}")
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # 构建 DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    loader_kwargs = dict(num_workers=max(0, args.workers), pin_memory=str(args.device).startswith('cuda'))
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, **loader_kwargs)
    print(f"训练集样本数: {len(X_train)}, 验证集样本数: {len(X_val)}")

    # 形状元信息（基于训练集）
    hist_len = X_train.shape[1]
    seq_len = y_train.shape[1]
    feature_dim = X_train.shape[2]

    if args.model == 'seq_ar_diffusion':
        from seq_ar_diffusion import SeqCondEncoder as ARCondEncoder, SeqARDiffusionModel, SeqARGaussianDiffusion
        cond_encoder = ARCondEncoder(feature_dim=feature_dim, hist_len=hist_len).to(args.device)
        model = SeqARDiffusionModel(
            cond_dim=cond_encoder.hidden_dim + 1,
            y_dim=1,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            t_embed_dim=args.t_embed_dim
        ).to(args.device)
        diffusion = SeqARGaussianDiffusion(
            model,
            cond_encoder,
            timesteps=args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            device=args.device,
            seq_len=seq_len,
            schedule=args.schedule,
            k_steps=args.k_steps
        )
        meta = {
            "model": "seq_ar_diffusion",
            "timesteps": args.timesteps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "schedule": args.schedule,
            "t_embed_dim": args.t_embed_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "k_steps": args.k_steps,
            "feature_dim": feature_dim,
            "hist_len": hist_len,
            "seq_len": seq_len,
            "lr": args.lr,
        }
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        optimizer = torch.optim.Adam(list(model.parameters()) + list(cond_encoder.parameters()), lr=args.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
        scaler = torch.cuda.amp.GradScaler(enabled=str(args.device).startswith('cuda'))
        best_val_loss = float('inf')
        best_epoch = -1
        for epoch in range(args.epochs):
            epoch_loss = 0
            model.train()
            cond_encoder.train()
            for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} train"), start=0):
                x_batch = x_batch.float().to(args.device, non_blocking=True)
                y_batch = y_batch.float().to(args.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=str(args.device).startswith('cuda')):
                    loss = diffusion.train_loss(x_batch, y_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(cond_encoder.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(epoch + (batch_idx + 1) / max(1, len(train_loader)))
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            model.eval()
            cond_encoder.eval()
            val_losses = []
            with torch.no_grad():
                for x_val, y_val_ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} val"):
                    x_val = x_val.float().to(args.device, non_blocking=True)
                    y_val_ = y_val_.float().to(args.device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=str(args.device).startswith('cuda')):
                        val_loss = diffusion.train_loss(x_val, y_val_)
                    val_losses.append(val_loss.item())
            avg_val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
            print(f"[SeqAR-Diffusion] Epoch {epoch+1}/{args.epochs} Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                torch.save({'cond_encoder': cond_encoder.state_dict(), 'model': model.state_dict(), 'meta': meta}, os.path.join(CHECKPOINT_DIR, 'seq_ar_diffusion_best.pth'))
                print(f"Saved new best SeqAR-Diffusion model at epoch {best_epoch} with val loss {best_val_loss:.4f}")

                # ====== 计算并打印验证集指标 ======
                model.eval()
                cond_encoder.eval()
                y_true_list = []
                y_pred_list = []
                for x_val, y_val_ in tqdm(val_loader, desc=f"[Eval] Best Model at Epoch {best_epoch}"):
                    x_val = x_val.float().to(args.device, non_blocking=True)
                    y_val_ = y_val_.float().to(args.device, non_blocking=True)
                    with torch.no_grad():
                        y_pred = diffusion.sample(x_val, seq_len=seq_len, deterministic=True).cpu().numpy()
                        y_true = y_val_.cpu().numpy()
                        y_true_list.append(y_true)
                        y_pred_list.append(y_pred)
                y_true_all = np.concatenate(y_true_list, axis=0)  # (N, seq_len)
                y_pred_all = np.concatenate(y_pred_list, axis=0)  # (N, seq_len)
                import joblib
                DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
                scaler_y = joblib.load(os.path.join(DATA_DIR, 'scaler_y.joblib'))

                # 先裁剪到归一化区间，避免反归一化后出现异常
                y_pred_all = np.clip(y_pred_all, 0.0, 1.0)

                # 反归一化
                y_true_inv = scaler_y.inverse_transform(y_true_all)
                y_pred_inv = scaler_y.inverse_transform(y_pred_all)

                # 物理约束：功率非负
                y_true_eval = np.clip(y_true_inv, 0.0, None)
                y_pred_eval = np.clip(y_pred_inv, 0.0, None)

                # 展平并过滤非有限值
                yt = y_true_eval.reshape(-1)
                yp = y_pred_eval.reshape(-1)
                finite_mask = np.isfinite(yt) & np.isfinite(yp)
                yt = yt[finite_mask]
                yp = yp[finite_mask]

                mae = float(np.mean(np.abs(yp - yt))) if yt.size > 0 else float('nan')
                rmse = float(np.sqrt(np.mean((yp - yt) ** 2))) if yt.size > 0 else float('nan')

                # 稳健 MAPE：容量阈值取 scaler 的最大值（所有时间步最大）
                try:
                    data_max = getattr(scaler_y, 'data_max_', None)
                    if data_max is not None:
                        y_cap = float(np.max(data_max))
                    else:
                        y_cap = float(np.max(y_true_eval))
                except Exception:
                    y_cap = float(np.max(y_true_eval))
                den = np.maximum(np.abs(yt), max(1e-8, 0.05 * y_cap))
                mape = float(np.mean(np.abs(yp - yt) / den)) if yt.size > 0 else float('nan')
                smape = float(np.mean(2.0 * np.abs(yp - yt) / (np.abs(yp) + np.abs(yt) + 1e-8))) if yt.size > 0 else float('nan')

                print(f"[Best Model Metrics] Epoch {best_epoch} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.4f} | sMAPE: {smape:.4f}")
                # 调试打印反归一化后的数据范围，确认尺度合理
                try:
                    print(f"[Debug] y_true_inv range: min={np.min(y_true_inv):.2f}, max={np.max(y_true_inv):.2f}, mean={np.mean(y_true_inv):.2f}")
                    print(f"[Debug] y_pred_inv range: min={np.min(y_pred_inv):.2f}, max={np.max(y_pred_inv):.2f}, mean={np.mean(y_pred_inv):.2f}")
                except Exception as _e:
                    print(f"[Debug] range print failed: {_e}")
        torch.save({'cond_encoder': cond_encoder.state_dict(), 'model': model.state_dict(), 'meta': meta}, os.path.join(CHECKPOINT_DIR, 'seq_ar_diffusion_last.pth'))
        print(f"Saved last SeqAR-Diffusion model to {os.path.join(CHECKPOINT_DIR, 'seq_ar_diffusion_last.pth')}")
    elif args.model == 'seq_lstm':
        train_seq_lstm(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, device=args.device,
                       hist_len=hist_len, seq_len=seq_len, feature_dim=feature_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
#         # 只用前 10000 条数据训练（可根据内存情况调整）
#         df = load_data(max_samples=10000)
#         if args.model == 'seq_vae':
#             import numpy as np
#             X = np.load(os.path.join(os.path.dirname(__file__), '../data/X_seq.npy'))
#             y = np.load(os.path.join(os.path.dirname(__file__), '../data/y_seq.npy'))
#             # 1. 长度对齐
#             min_samples = min(len(X), len(y))
#             X = X[:min_samples]
#             y = y[:min_samples]
#             # 2. 过滤异常样本
#             mask = (~np.isnan(X).any(axis=(1, 2)) & ~np.isinf(X).any(axis=(1, 2)) &
#                     ~np.isnan(y).any(axis=1) & ~np.isinf(y).any(axis=1))
#             X, y = X[mask], y[mask]
#             print(f"[Clean] 剩余样本数: {len(X)}")
#             # 3. MinMax 归一化
#             from sklearn.preprocessing import MinMaxScaler
#             from dataloader import split_data
#             X_2d = X.reshape(-1, X.shape[-1])
#             scaler_x = MinMaxScaler()
#             X_scaled_2d = scaler_x.fit_transform(X_2d)
#             X_scaled = X_scaled_2d.reshape(X.shape)
#             y_flat = y.reshape(-1, 1)
#             scaler_y = MinMaxScaler()
#             y_scaled_flat = scaler_y.fit_transform(y_flat)
#             y_scaled = y_scaled_flat.reshape(y.shape)
#             # 4. 断言确保干净
#             assert not np.isnan(X_scaled).any(), "X_scaled 仍含 nan"
#             assert not np.isinf(X_scaled).any(), "X_scaled 仍含 inf"
#             assert not np.isnan(y_scaled).any(), "y_scaled 仍含 nan"
#             assert not np.isinf(y_scaled).any(), "y_scaled 仍含 inf"
#             # 5. dry-run: 用全部数据做训练和评估（可选 max_samples 截断）
#             if getattr(args, "dry_run_fit_train", False):
#                 max_samples = getattr(args, "max_samples", None)
#                 if max_samples is not None:
#                     X_train, y_train = X_scaled[:max_samples], y_scaled[:max_samples]
#                     X_test, y_test = X_scaled[:max_samples], y_scaled[:max_samples]
#                     print(f"[Dry-run] 使用前 {max_samples} 条数据做训练和评估，样本数: {X_train.shape[0]}")
#                 else:
#                     X_train, y_train = X_scaled, y_scaled
#                     X_test, y_test = X_scaled, y_scaled
#                     print(f"[Dry-run] 使用全部数据做训练和评估，样本数: {X_train.shape[0]}")
#             else:
#                 X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
#                 print(f"训练集样本数: {X_train.shape[0]}")
#             feature_dim = X.shape[-1]
#             train_seq_vae(X_train, y_train, epochs=args.epochs, device=args.device, hist_len=24, seq_len=24, feature_dim=feature_dim)
#         elif args.model == 'seq_diffusion':
#             import numpy as np
#             X = np.load(os.path.join(os.path.dirname(__file__), '../data/X_seq.npy'))
#             y = np.load(os.path.join(os.path.dirname(__file__), '../data/y_seq.npy'))
#             min_samples = min(len(X), len(y))
#             X = X[:min_samples]
#             y = y[:min_samples]
#             mask = (~np.isnan(X).any(axis=(1, 2)) & ~np.isinf(X).any(axis=(1, 2)) &
#                     ~np.isnan(y).any(axis=1) & ~np.isinf(y).any(axis=1))
#             X, y = X[mask], y[mask]
#             print(f"[Clean] 剩余样本数: {len(X)}")
#             from sklearn.preprocessing import MinMaxScaler
#             from dataloader import split_data
#             X_2d = X.reshape(-1, X.shape[-1])
#             scaler_x = MinMaxScaler()
#             X_scaled_2d = scaler_x.fit_transform(X_2d)
#             X_scaled = X_scaled_2d.reshape(X.shape)
#             y_flat = y.reshape(-1, 1)
#             scaler_y = MinMaxScaler()
#             y_scaled_flat = scaler_y.fit_transform(y_flat)
#             y_scaled = y_scaled_flat.reshape(y.shape)
#             assert not np.isnan(X_scaled).any(), "X_scaled 仍含 nan"
#             assert not np.isinf(X_scaled).any(), "X_scaled 仍含 inf"
#             assert not np.isnan(y_scaled).any(), "y_scaled 仍含 nan"
#             assert not np.isinf(y_scaled).any(), "y_scaled 仍含 inf"
#             X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
#             print(f"训练集样本数: {X_train.shape[0]}")
#             feature_dim = X.shape[-1]
#             train_seq_diffusion(X_train, y_train, epochs=args.epochs, device=args.device, hist_len=24, seq_len=24, feature_dim=feature_dim)
#         elif args.model == 'seq_ar_diffusion':
#             import numpy as np
#             X = np.load(os.path.join(os.path.dirname(__file__), '../data/X_seq.npy'))
#             y = np.load(os.path.join(os.path.dirname(__file__), '../data/y_seq.npy'))
#             min_samples = min(len(X), len(y))
#             X = X[:min_samples]
#             y = y[:min_samples]
#             mask = (~np.isnan(X).any(axis=(1, 2)) & ~np.isinf(X).any(axis=(1, 2)) &
#                     ~np.isnan(y).any(axis=1) & ~np.isinf(y).any(axis=1))
#             X, y = X[mask], y[mask]
#             print(f"[Clean] 剩余样本数: {len(X)}")
#             from sklearn.preprocessing import MinMaxScaler
#             from dataloader import split_data
#             X_2d = X.reshape(-1, X.shape[-1])
#             scaler_x = MinMaxScaler()
#             X_scaled_2d = scaler_x.fit_transform(X_2d)
#             X_scaled = X_scaled_2d.reshape(X.shape)
#             y_flat = y.reshape(-1, 1)
#             scaler_y = MinMaxScaler()
#             y_scaled_flat = scaler_y.fit_transform(y_flat)
#             y_scaled = y_scaled_flat.reshape(y.shape)
#             assert not np.isnan(X_scaled).any(), "X_scaled 仍含 nan"
#             assert not np.isinf(X_scaled).any(), "X_scaled 仍含 inf"
#             assert not np.isnan(y_scaled).any(), "y_scaled 仍含 nan"
#             assert not np.isinf(y_scaled).any(), "y_scaled 仍含 inf"
#             # dry-run: 用全部数据做训练和评估
#             if getattr(args, "dry_run_fit_train", False):
#                 X_train, y_train = X_scaled, y_scaled
#                 X_test, y_test = X_scaled, y_scaled
#                 print(f"[Dry-run] 使用全部数据做训练和评估，样本数: {X_train.shape[0]}")
#             else:
#                 X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
#                 print(f"训练集样本数: {X_train.shape[0]}")
            
#         else:
#             X, y, _, _ = preprocess_features(df, seq_len=24)
#             # dry-run: 用全部数据做训练和评估
#             if getattr(args, "dry_run_fit_train", False):
#                 X_train, y_train = X, y
#                 X_test, y_test = X, y
#                 print(f"[Dry-run] 使用全部数据做训练和评估，样本数: {X_train.shape[0]}")
#             else:
#                 X_train, X_test, y_train, y_test = split_data(X, y)
#                 print(f"训练集样本数: {X_train.shape[0]}")
#             if args.model == 'diffusion':
#                 train_diffusion(X_train, y_train, epochs=args.epochs, device=args.device)
#             elif args.model == 'weibull_diffusion':
#                 train_weibull_diffusion(X_train, y_train, epochs=args.epochs, device=args.device)
#             elif args.model == 'gan':
#                 train_gan(X_train, y_train, epochs=args.epochs, device=args.device)
#             elif args.model == 'vae':
#                 train_vae(X_train, y_train, epochs=args.epochs, device=args.device)
