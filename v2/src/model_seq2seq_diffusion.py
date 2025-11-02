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

    def forward(
        self,
        y0: torch.Tensor,
        x_future: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        steps: int,
        teacher_forcing: float = 0.0,
        y_truth: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
    ):
        outputs = []
        h, c = hidden
        prev_y = y0.unsqueeze(1)  # [B, 1, 1]
        for t in range(steps):
            if self.training and y_truth is not None and torch.rand(1).item() < teacher_forcing:
                y_in = y_truth[:, t : t + 1, :]  # [B, 1, 1]
            else:
                y_in = prev_y
            exo_t = x_future[:, t : t + 1, :]  # [B, 1, exo_dim]
            step_in = torch.cat([y_in, exo_t], dim=2)  # [B, 1, 1+exo]
            if noise_std > 0.0:
                step_in = step_in + torch.randn_like(step_in) * noise_std
            out, (h, c) = self.lstm(self.dropout(step_in), (h, c))
            yhat = self.proj(out)
            outputs.append(yhat)
            prev_y = yhat.detach()
        return torch.cat(outputs, dim=1)  # [B, steps, 1]


class Seq2SeqDiffusion(nn.Module):
    """
    A diffusion-style variant using Gaussian noise injected into decoder inputs.
    Shares the same interface as Seq2Seq for compatibility.
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
        emb = self.turb_emb(turb_idx)  # [B, emb_dim]
        emb_hist = emb.unsqueeze(1).expand(-1, x_hist.size(1), -1)
        emb_future = emb.unsqueeze(1).expand(-1, x_future.size(1), -1)

        x_hist_aug = torch.cat([x_hist, emb_hist], dim=-1)
        x_future_aug = torch.cat([x_future, emb_future], dim=-1)

        h, c = self.encoder(x_hist_aug)
        out = self.decoder(y0, x_future_aug, (h, c), steps=pred_steps, teacher_forcing=teacher_forcing, y_truth=y_truth, noise_std=noise_std)
        return out