import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqCondEncoder(nn.Module):
    def __init__(self, feature_dim, hist_len, hidden_dim=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x_hist):
        # x_hist: (batch, hist_len, feature_dim)
        _, (h_n, _) = self.lstm(x_hist)
        h_last = h_n[-1]  # (batch, hidden_dim)
        return h_last

class SeqARDiffusionModel(nn.Module):
    def __init__(self, cond_dim, y_dim=1, hidden_dim=256, n_layers=4, dropout=0.2):
        super().__init__()
        layers = []
        input_dim = cond_dim + y_dim + 1  # +1 for timestep
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, y_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, cond, y_prev, t):
        # cond: (batch, cond_dim), y_prev: (batch, 1), t: (batch,)
        t_embed = t.unsqueeze(-1).float() / 1000.0
        x_in = torch.cat([cond, y_prev, t_embed], dim=1)
        return self.net(x_in)

class SeqARGaussianDiffusion:
    def __init__(self, model, cond_encoder, timesteps=300, beta_start=1e-4, beta_end=0.1, device='cpu', seq_len=24):
        self.model = model
        self.cond_encoder = cond_encoder
        self.timesteps = timesteps
        self.device = device
        self.seq_len = seq_len
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, y_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(y_start)
        sqrt_alpha_bar = self.alpha_bars[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bars[t]).sqrt().unsqueeze(-1)
        return sqrt_alpha_bar * y_start + sqrt_one_minus_alpha_bar * noise

    def p_sample(self, cond, y_prev, t):
        pred_noise = self.model(cond, y_prev, t)
        beta_t = self.betas[t].unsqueeze(-1)
        alpha_t = self.alphas[t].unsqueeze(-1)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
        mean = (1 / alpha_t.sqrt()) * (y_prev - beta_t / (1 - alpha_bar_t).sqrt() * pred_noise)
        noise = torch.randn_like(y_prev) if (t > 0).any() else 0
        return mean + beta_t.sqrt() * noise

    def sample(self, x_hist, seq_len=None):
        # x_hist: (batch, hist_len, feature_dim)
        if seq_len is None:
            seq_len = self.seq_len
        cond_base = self.cond_encoder(x_hist)  # (batch, hidden_dim)
        batch_size = x_hist.shape[0]
        y_seq = []
        # 自回归：上一时刻的y（初始为0）
        y_prev_seq = torch.zeros((batch_size, 1), device=self.device)
        for step in range(seq_len):
            # 把上一步的y拼到条件向量
            cond = torch.cat([cond_base, y_prev_seq], dim=1)
            # 扩散反推：从噪声开始
            y = torch.randn((batch_size, 1), device=self.device)
            for t in reversed(range(self.timesteps)):
                t_tensor = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
                y = self.p_sample(cond, y, t_tensor)
            y_seq.append(y)
            # 更新上一时刻y
            y_prev_seq = y
        y_seq = torch.cat(y_seq, dim=1)  # (batch, seq_len)
        return y_seq

    def train_loss(self, x_hist, y_start):
        cond_base = self.cond_encoder(x_hist)  # (batch, hidden_dim)
        batch_size = x_hist.shape[0]
        seq_len = y_start.shape[1]
        # device check
        assert cond_base.device == y_start.device, f"cond.device={cond_base.device}, y_start.device={y_start.device}"
        # 逐步加权：后面的步权重更大，缓解误差累积
        weights = torch.linspace(1.0, float(seq_len), steps=seq_len, device=self.device)
        weights = weights / weights.sum()
        loss_total = 0.0
        for step in range(seq_len):
            # teacher forcing：第 step 步使用上一时刻真值作为条件
            if step == 0:
                y_prev_seq = torch.zeros((batch_size, 1), device=self.device, dtype=y_start.dtype)
            else:
                y_prev_seq = y_start[:, step-1:step]
            cond = torch.cat([cond_base, y_prev_seq], dim=1)  # (batch, cond_dim+1)
            y_true = y_start[:, step:step+1]  # (batch, 1)
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
            noise = torch.randn_like(y_true)
            y_noisy = self.q_sample(y_true, t, noise)
            pred_noise = self.model(cond, y_noisy, t)
            loss = F.mse_loss(pred_noise, noise)
            loss_total += weights[step] * loss
        return loss_total