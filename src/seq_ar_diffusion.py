import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    t: (batch,) long or float
    return: (batch, dim) sinusoidal embedding
    """
    if t.dtype != torch.float32 and t.dtype != torch.float64:
        t = t.float()
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(float(max_period), device=device)) * torch.arange(0, half, device=device).float() / max(1, half)
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class SeqCondEncoder(nn.Module):
    def __init__(self, feature_dim, hist_len, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x_hist):
        # x_hist: (batch, hist_len, feature_dim)
        _, (h_n, _) = self.lstm(x_hist)
        h_last = h_n[-1]  # (batch, hidden_dim)
        return h_last

class SeqARDiffusionModel(nn.Module):
    def __init__(self, cond_dim, y_dim=1, hidden_dim=128, n_layers=2, dropout=0.1, t_embed_dim=32, h_embed_dim=None):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.h_embed_dim = h_embed_dim if h_embed_dim is not None else t_embed_dim
        in_dim = cond_dim + y_dim + self.t_embed_dim + self.h_embed_dim
        # 简化网络结构
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, y_dim)
        )

    def forward(self, cond, x_t, t, h=None):
        # cond: (batch, cond_dim), x_t: (batch, 1), t: (batch,), h: (batch,) 预测步索引
        t_embed = sinusoidal_time_embedding(t, self.t_embed_dim)
        if h is not None:
            h_embed = sinusoidal_time_embedding(h, self.h_embed_dim)
        else:
            h_embed = torch.zeros((cond.shape[0], self.h_embed_dim), device=cond.device, dtype=t_embed.dtype)
        x_in = torch.cat([cond, x_t, t_embed, h_embed], dim=1)
        return self.net(x_in)

class SeqARGaussianDiffusion:
    def __init__(self, model, cond_encoder, timesteps=300, beta_start=1e-4, beta_end=0.1, device='cpu', seq_len=24, schedule: str = 'cosine', k_steps: int = 4):
        self.model = model
        self.cond_encoder = cond_encoder
        self.timesteps = timesteps
        self.device = device
        self.seq_len = seq_len
        self.schedule = schedule
        self.k_steps = max(1, int(k_steps))

        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
            self.alphas = 1. - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        elif schedule == 'cosine':
            s = 0.008
            t = torch.linspace(0, timesteps, timesteps + 1, device=device) / timesteps
            f = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_bar = f / f[0]
            # 转换为逐步 alpha、beta
            self.alphas = (alpha_bar[1:] / alpha_bar[:-1]).clamp(min=1e-5, max=1.0)
            self.betas = (1.0 - self.alphas).clamp(min=1e-5, max=0.999)
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def q_sample(self, y_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(y_start)
        sqrt_alpha_bar = self.alpha_bars[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bars[t]).sqrt().unsqueeze(-1)
        return sqrt_alpha_bar * y_start + sqrt_one_minus_alpha_bar * noise

    def p_sample(self, cond, x_t, t, add_noise: bool = True, h=None):
        pred_noise = self.model(cond, x_t, t, h)
        beta_t = self.betas[t].unsqueeze(-1)
        alpha_t = self.alphas[t].unsqueeze(-1)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)

        # 当 t==0（单步或最后一步）时，直接用封闭式公式预测 x0，提升 timesteps 很小时的稳定性
        # 公式来源：x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        # => x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        pred_x0 = (x_t - (1.0 - alpha_bar_t).sqrt() * pred_noise) / (alpha_bar_t.sqrt() + 1e-8)
        if (t == 0).all():
            return pred_x0

        # 使用 DDPM 后验方差（beta tilde）常规反推
        prev_t = torch.clamp(t - 1, min=0)
        alpha_bar_prev = self.alpha_bars[prev_t].unsqueeze(-1)
        posterior_var_t = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-8)
        mean = (1 / (alpha_t.sqrt() + 1e-8)) * (x_t - beta_t / ((1 - alpha_bar_t).sqrt() + 1e-8) * pred_noise)
        noise = torch.randn_like(x_t) if ((t > 0).any() and add_noise) else 0
        y_next = mean + posterior_var_t.sqrt() * noise
        return y_next

    def train_loss(self, x_hist, y_start):
        """
        多步训练：随机采样 K 个时间步，计算噪声回归 MSE 的加权平均。
        适配 y_start 已经缩放到 [0,1] 的情形（若未缩放也可工作）。
        """
        cond_base = self.cond_encoder(x_hist)  # (batch, hidden_dim)
        batch_size = x_hist.shape[0]
        seq_len = y_start.shape[1]

        k = min(self.k_steps, seq_len)
        # 随机选取 k 个不同的时间位点
        steps = torch.randperm(seq_len, device=self.device)[:k].tolist()

        total_loss = 0.0
        for step in steps:
            # teacher forcing：使用上一时刻真值作为条件
            if step == 0:
                y_prev_seq = torch.zeros((batch_size, 1), device=self.device, dtype=y_start.dtype)
            else:
                y_prev_seq = y_start[:, step-1:step]
            cond = torch.cat([cond_base, y_prev_seq], dim=1)  # (batch, cond_dim+1)
            y_true = y_start[:, step:step+1]  # (batch, 1)

            # 扩散过程：随机时间 t
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
            noise = torch.randn_like(y_true)
            y_noisy = self.q_sample(y_true, t, noise)
            # 传入当前预测步索引作为额外条件
            h = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
            pred_noise = self.model(cond, y_noisy, t, h)
            # 纯 MSE（epsilon loss）
            mse = F.mse_loss(pred_noise, noise, reduction='mean')
            total_loss += mse

        loss = total_loss / float(k)
        return loss

    def sample(self, x_hist, seq_len=None, deterministic: bool = False):
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
            # 传入当前预测步索引
            h = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
            for t in reversed(range(self.timesteps)):
                t_tensor = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
                y = self.p_sample(cond, y, t_tensor, add_noise=(not deterministic), h=h)
            # 仅在完成反推后进行裁剪，避免破坏反推过程
            y = torch.clamp(y, 0.0, 1.0)
            y_seq.append(y)
            # 更新上一时刻y
            y_prev_seq = y
        y_seq = torch.cat(y_seq, dim=1)  # (batch, seq_len)
        return y_seq

