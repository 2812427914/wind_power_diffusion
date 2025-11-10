from typing import Optional, Tuple

import torch
import torch.nn as nn


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

    def forward(self, x):
        outputs, (h, c) = self.lstm(x)
        return h, c


class Decoder(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int = 1, exo_dim: int = 0, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=out_dim + exo_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_size, out_dim)
        self.dropout = nn.Dropout(dropout)
        # Latent noise projection (GAN-style stochastic generator)
        self.z_dim = 16
        self.z_proj = nn.Linear(self.z_dim, out_dim + exo_dim)

    def forward(
        self,
        y0: torch.Tensor,
        x_future: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        steps: int,
        teacher_forcing: float = 0.0,
        y_truth: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
        z: Optional[torch.Tensor] = None,
    ):
        """
        Decoder that injects a per-sequence latent code z as an additive bias to inputs.
        Also supports optional Gaussian input noise via noise_std.
        """
        outputs = []
        h, c = hidden
        prev_y = y0.unsqueeze(1)  # [B, 1, 1]
        B = y0.size(0)
        # Prepare latent bias once per sequence
        z_bias = None
        if z is not None:
            z_bias = self.z_proj(z).unsqueeze(1)  # [B, 1, in_size]
        for t in range(steps):
            if self.training and y_truth is not None and torch.rand(1).item() < teacher_forcing:
                y_in = y_truth[:, t : t + 1, :]
            else:
                y_in = prev_y
            exo_t = x_future[:, t : t + 1, :]
            in_t = torch.cat([y_in, exo_t], dim=2)  # [B, 1, in_size]
            if z_bias is not None:
                in_t = in_t + z_bias  # broadcast along time=1
            if noise_std > 0:
                in_t = in_t + torch.randn_like(in_t) * noise_std
            out, (h, c) = self.lstm(self.dropout(in_t), (h, c))
            yhat = self.proj(out)
            outputs.append(yhat)
            prev_y = yhat.detach()
        return torch.cat(outputs, dim=1)


class Seq2SeqGAN(nn.Module):
    """
    GAN-style generator only for evaluation/prediction.
    Keeps interface compatible with training/eval pipeline.
    """
    def __init__(self, input_dim: int, exo_dim: int, n_turbines: int, emb_dim: int = 16, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.turb_emb = nn.Embedding(n_turbines, emb_dim)
        self.encoder = Encoder(input_dim + emb_dim, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size, out_dim=1, exo_dim=exo_dim + emb_dim, num_layers=num_layers, dropout=dropout)

    def forward(
        self,
        x_hist: torch.Tensor,
        x_future: torch.Tensor,
        y0: torch.Tensor,
        turb_idx: torch.Tensor,
        pred_steps: int,
        teacher_forcing: float = 0.0,
        y_truth: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
    ):
        emb = self.turb_emb(turb_idx)
        emb_hist = emb.unsqueeze(1).expand(-1, x_hist.size(1), -1)
        emb_future = emb.unsqueeze(1).expand(-1, x_future.size(1), -1)

        x_hist_aug = torch.cat([x_hist, emb_hist], dim=-1)
        x_future_aug = torch.cat([x_future, emb_future], dim=-1)

        h, c = self.encoder(x_hist_aug)
        # Stochastic latent code: sample when training for diversity; deterministic zero in eval for repeatability
        B = x_hist.size(0)
        device = x_hist.device
        if self.training:
            z = torch.randn(B, self.decoder.z_dim, device=device)
        else:
            z = torch.zeros(B, self.decoder.z_dim, device=device)
        out = self.decoder(
            y0, x_future_aug, (h, c),
            steps=pred_steps,
            teacher_forcing=teacher_forcing,
            y_truth=y_truth,
            noise_std=noise_std,
            z=z,
        )
        return out