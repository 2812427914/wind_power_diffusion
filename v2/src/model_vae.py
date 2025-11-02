from typing import Optional, Tuple

import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, z_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.mu = nn.Linear(hidden_size, z_dim)
        self.logvar = nn.Linear(hidden_size, z_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs, (h, c) = self.lstm(x)
        # use last layer's hidden state
        h_last = h[-1]  # [B, H]
        mu = self.mu(h_last)
        logvar = self.logvar(h_last)
        return mu, logvar, (h, c)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAEDecoder(nn.Module):
    def __init__(self, hidden_size: int, z_dim: int, out_dim: int = 1, exo_dim: int = 0, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.z_to_h = nn.Linear(z_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=out_dim + exo_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.num_layers = num_layers
        self.proj = nn.Linear(hidden_size, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,
        y0: torch.Tensor,
        x_future: torch.Tensor,
        steps: int,
        teacher_forcing: float = 0.0,
        y_truth: Optional[torch.Tensor] = None,
    ):
        # init hidden from z for all LSTM layers
        h0_single = torch.tanh(self.z_to_h(z))  # [B, H]
        h0 = h0_single.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, B, H]
        c0 = torch.zeros_like(h0)  # [num_layers, B, H]
        outputs = []
        prev_y = y0.unsqueeze(1)
        h, c = h0, c0
        for t in range(steps):
            if self.training and y_truth is not None and torch.rand(1).item() < teacher_forcing:
                y_in = y_truth[:, t : t + 1, :]
            else:
                y_in = prev_y
            exo_t = x_future[:, t : t + 1, :]
            inp = torch.cat([y_in, exo_t], dim=2)
            out, (h, c) = self.lstm(self.dropout(inp), (h, c))
            yhat = self.proj(out)
            outputs.append(yhat)
            prev_y = yhat.detach()
        return torch.cat(outputs, dim=1)


class Seq2SeqVAE(nn.Module):
    """
    VAE-style model: encoder outputs mu/logvar -> sample z -> decoder generates sequence.
    Interface kept compatible.
    """
    def __init__(self, input_dim: int, exo_dim: int, n_turbines: int, emb_dim: int = 16, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2, z_dim: int = 32):
        super().__init__()
        self.turb_emb = nn.Embedding(n_turbines, emb_dim)
        self.encoder = VAEEncoder(input_dim + emb_dim, hidden_size, z_dim, num_layers, dropout)
        self.decoder = VAEDecoder(hidden_size, z_dim, out_dim=1, exo_dim=exo_dim + emb_dim, num_layers=num_layers, dropout=dropout)

    def forward(
        self,
        x_hist: torch.Tensor,
        x_future: torch.Tensor,
        y0: torch.Tensor,
        turb_idx: torch.Tensor,
        pred_steps: int,
        teacher_forcing: float = 0.0,
        y_truth: Optional[torch.Tensor] = None,
    ):
        emb = self.turb_emb(turb_idx)  # [B, emb_dim]
        emb_hist = emb.unsqueeze(1).expand(-1, x_hist.size(1), -1)
        emb_future = emb.unsqueeze(1).expand(-1, x_future.size(1), -1)

        x_hist_aug = torch.cat([x_hist, emb_hist], dim=-1)
        x_future_aug = torch.cat([x_future, emb_future], dim=-1)

        mu, logvar, _hc = self.encoder(x_hist_aug)
        z = reparameterize(mu, logvar)

        out = self.decoder(z, y0, x_future_aug, steps=pred_steps, teacher_forcing=teacher_forcing, y_truth=y_truth)
        return out