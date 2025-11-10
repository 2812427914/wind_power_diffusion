from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    t: (batch,) long or float
    return: (batch, dim) sinusoidal embedding
    """
    if t.dtype not in (torch.float32, torch.float64):
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


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T_in, F]
        outputs, (h, c) = self.lstm(x)
        return h, c


class NoisePredictor(nn.Module):
    """
    Epsilon predictor network for diffusion reverse process.
    Input: concat(cond, x_t, t_embed, h_embed) -> predict noise (epsilon).
    """
    def __init__(self, cond_dim: int, y_dim: int = 1, hidden_dim: int = 128, dropout: float = 0.1, t_embed_dim: int = 32, h_embed_dim: Optional[int] = None):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.h_embed_dim = h_embed_dim if h_embed_dim is not None else t_embed_dim
        in_dim = cond_dim + y_dim + self.t_embed_dim + self.h_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, y_dim)
        )

    def forward(self, cond: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        # cond: [B, cond_dim], x_t: [B, 1], t: [B], h: [B] (forecast step index)
        t_embed = sinusoidal_time_embedding(t, self.t_embed_dim)
        if h is not None:
            h_embed = sinusoidal_time_embedding(h, self.h_embed_dim)
        else:
            h_embed = torch.zeros((cond.shape[0], self.h_embed_dim), device=cond.device, dtype=t_embed.dtype)
        x_in = torch.cat([cond, x_t, t_embed, h_embed], dim=1)
        return self.net(x_in)


class Seq2SeqARDiffusion(nn.Module):
    """
    True AR Diffusion model within Seq2Seq interface:
    - Encoder: LSTM over history + turbine embedding.
    - Condition per forecast step: [enc_out, exo_future_t_with_emb, prev_y].
    - Train: epsilon loss over randomly sampled diffusion timesteps.
    - Sample: reverse diffusion from noise per forecast step (AR with prev_y).
    """
    def __init__(
        self,
        input_dim: int,
        exo_dim: int,
        n_turbines: int,
        emb_dim: int = 16,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        # diffusion params
        timesteps: int = 100,  # Reduced from 100
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",  # Use linear for simplicity
        t_embed_dim: int = 16,    # Reduced from 32
        h_embed_dim: Optional[int] = None,
        k_steps: int = 2,         # Reduced from 4
    ):
        super().__init__()
        self.turb_emb = nn.Embedding(n_turbines, emb_dim)
        # encoder consumes history features + turbine embedding
        self.encoder = Encoder(input_dim + emb_dim, hidden_size, num_layers, dropout)

        # cond = [enc_out(hidden_size), exo_future_t( exo_dim + emb_dim ), prev_y(1)]
        self.cond_dim = hidden_size + (exo_dim + emb_dim) + 1
        self.noise_pred = NoisePredictor(
            cond_dim=self.cond_dim,
            y_dim=1,
            hidden_dim=hidden_size,
            dropout=dropout,
            t_embed_dim=t_embed_dim,
            h_embed_dim=h_embed_dim if h_embed_dim is not None else t_embed_dim,
        )

        # diffusion buffers
        self.timesteps = int(timesteps)
        self.k_steps = max(1, int(k_steps))
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.timesteps)
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)
        elif schedule == "cosine":
            s = 0.008
            t = torch.linspace(0, self.timesteps, self.timesteps + 1) / self.timesteps
            f = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_bar = f / f[0]
            alphas = (alpha_bar[1:] / alpha_bar[:-1]).clamp(min=1e-5, max=1.0)
            betas = (1.0 - alphas).clamp(min=1e-5, max=0.999)
            alpha_bars = torch.cumprod(alphas, dim=0)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, y_start: torch.Tensor, t_idx: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        # y_start: [B, 1], t_idx: [B]
        if noise is None:
            noise = torch.randn_like(y_start)
        sqrt_alpha_bar = self.alpha_bars[t_idx].sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha_bar = (1.0 - self.alpha_bars[t_idx]).sqrt().unsqueeze(-1)
        return sqrt_alpha_bar * y_start + sqrt_one_minus_alpha_bar * noise

    def p_sample(self, cond: torch.Tensor, x_t: torch.Tensor, t_idx: torch.Tensor, add_noise: bool = True, h_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        # predict noise
        pred_noise = self.noise_pred(cond, x_t, t_idx, h_idx)  # [B, 1]
        beta_t = self.betas[t_idx].unsqueeze(-1)
        alpha_t = self.alphas[t_idx].unsqueeze(-1)
        alpha_bar_t = self.alpha_bars[t_idx].unsqueeze(-1)

        # closed-form x0 estimate
        pred_x0 = (x_t - (1.0 - alpha_bar_t).sqrt() * pred_noise) / (alpha_bar_t.sqrt() + 1e-8)
        if (t_idx == 0).all():
            return pred_x0

        prev_t = torch.clamp(t_idx - 1, min=0)
        alpha_bar_prev = self.alpha_bars[prev_t].unsqueeze(-1)
        posterior_var_t = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-8)
        mean = (1.0 / (alpha_t.sqrt() + 1e-8)) * (x_t - beta_t / ((1.0 - alpha_bar_t).sqrt() + 1e-8) * pred_noise)
        noise = torch.randn_like(x_t) if ((t_idx > 0).any() and add_noise) else torch.zeros_like(x_t)
        return mean + posterior_var_t.sqrt() * noise

    def _build_conditions(self, x_hist: torch.Tensor, x_future: torch.Tensor, turb_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # turbine embeddings
        emb = self.turb_emb(turb_idx)  # [B, emb_dim]
        emb_hist = emb.unsqueeze(1).expand(-1, x_hist.size(1), -1)   # [B, T_in, emb_dim]
        emb_future = emb.unsqueeze(1).expand(-1, x_future.size(1), -1)  # [B, T_out, emb_dim]

        x_hist_aug = torch.cat([x_hist, emb_hist], dim=-1)
        x_future_aug = torch.cat([x_future, emb_future], dim=-1)

        h, c = self.encoder(x_hist_aug)
        enc_out = h[-1]  # top layer hidden state: [B, hidden_size]
        return enc_out, x_hist_aug, x_future_aug, emb

    def train_loss(self, x_hist: torch.Tensor, x_future: torch.Tensor, y_scaled: torch.Tensor, turb_idx: torch.Tensor) -> torch.Tensor:
        """
        Epsilon loss averaged over k randomly selected forecast steps per batch.
        x_hist: [B, T_in, F]
        x_future: [B, T_out, exo_dim]
        y_scaled: [B, T_out, 1]
        turb_idx: [B]
        """
        enc_out, _, x_future_aug, _ = self._build_conditions(x_hist, x_future, turb_idx)  # enc_out [B, H], x_future_aug [B, T_out, exo_dim+emb]
        B, T_out = y_scaled.size(0), y_scaled.size(1)
        device = x_hist.device

        k = min(self.k_steps, T_out)
        steps = torch.randperm(T_out, device=device)[:k].tolist()

        total_loss = 0.0
        for step in steps:
            # previous true y for teacher forcing in condition
            if step == 0:
                prev_y = torch.zeros((B, 1), device=device, dtype=y_scaled.dtype)  # [B, 1]
            else:
                prev_y = y_scaled[:, step - 1 : step, :].squeeze(1)  # [B, 1]
            exo_t = x_future_aug[:, step : step + 1, :]  # [B, 1, exo_dim+emb]
            exo_t_flat = exo_t.squeeze(1)                # [B, exo_dim+emb]
            cond = torch.cat([enc_out, exo_t_flat, prev_y], dim=1)  # [B, cond_dim]

            # Ensure 2D shape [B, 1] for diffusion inputs
            y_true = y_scaled[:, step : step + 1, :].squeeze(-1)  # [B, 1]
            t_idx = torch.randint(0, self.timesteps, (B,), device=device)  # [B]
            noise = torch.randn_like(y_true)  # [B, 1]
            y_noisy = self.q_sample(y_true, t_idx, noise)  # [B, 1]

            h_idx = torch.full((B,), step, dtype=torch.long, device=device)
            pred_noise = self.noise_pred(cond, y_noisy, t_idx, h_idx)
            mse = F.mse_loss(pred_noise, noise, reduction="mean")
            total_loss += mse

        loss = total_loss / float(k)
        return loss

    def sample(self, x_hist: torch.Tensor, x_future: torch.Tensor, turb_idx: torch.Tensor, seq_len: int, y0: Optional[torch.Tensor] = None, deterministic: bool = False) -> torch.Tensor:
        """
        Autoregressive sampling with reverse diffusion for each forecast step.
        Returns scaled predictions: [B, seq_len, 1]
        """
        enc_out, _, x_future_aug, _ = self._build_conditions(x_hist, x_future, turb_idx)
        B = x_hist.size(0)
        y_seq = []
        prev_y = y0 if (y0 is not None) else torch.zeros((B, 1), device=x_hist.device)
        for step in range(seq_len):
            exo_t = x_future_aug[:, step : step + 1, :]  # [B, 1, exo_dim+emb]
            exo_t_flat = exo_t.squeeze(1)                # [B, exo_dim+emb]
            cond = torch.cat([enc_out, exo_t_flat, prev_y], dim=1)  # [B, cond_dim]

            y = torch.randn((B, 1), device=x_hist.device)
            h_idx = torch.full((B,), step, dtype=torch.long, device=x_hist.device)
            for t in reversed(range(self.timesteps)):
                t_tensor = torch.full((B,), t, dtype=torch.long, device=x_hist.device)
                y = self.p_sample(cond, y, t_tensor, add_noise=(not deterministic), h_idx=h_idx)
            y_seq.append(y)
            prev_y = y
        return torch.cat(y_seq, dim=1).unsqueeze(-1)  # [B, seq_len, 1]

    def forward(
        self,
        x_hist: torch.Tensor,
        x_future: torch.Tensor,
        y0: torch.Tensor,
        turb_idx: torch.Tensor,
        pred_steps: int,
        teacher_forcing: float = 0.0,
        y_truth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Keep Seq2Seq-compatible interface. During eval -> deterministic sampling; train -> stochastic sampling.
        """
        deterministic = not self.training
        return self.sample(x_hist, x_future, turb_idx, seq_len=pred_steps, y0=y0, deterministic=deterministic)