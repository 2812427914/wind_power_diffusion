import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDiffusionModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=3):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        # x: [batch, cond_dim + y_dim]
        # t: [batch]
        t_embed = t.unsqueeze(-1).float() / 1000.0  # normalize t
        x_in = torch.cat([x, t_embed], dim=1)
        return self.net(x_in)

class GaussianDiffusion:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, y_start, t, noise=None):
        # y_start: [batch, y_dim]
        if noise is None:
            noise = torch.randn_like(y_start)
        sqrt_alpha_bar = self.alpha_bars[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bars[t]).sqrt().unsqueeze(-1)
        return sqrt_alpha_bar * y_start + sqrt_one_minus_alpha_bar * noise

    def p_sample(self, x_cond, y, t):
        # x_cond: [batch, cond_dim], y: [batch, y_dim], t: [batch]
        model_in = torch.cat([x_cond, y], dim=1)
        pred_noise = self.model(model_in, t)
        beta_t = self.betas[t].unsqueeze(-1)
        alpha_t = self.alphas[t].unsqueeze(-1)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
        mean = (1 / alpha_t.sqrt()) * (y - beta_t / (1 - alpha_bar_t).sqrt() * pred_noise)
        noise = torch.randn_like(y) if (t > 0).any() else 0
        return mean + beta_t.sqrt() * noise

    def sample(self, x_cond, y_shape):
        # x_cond: [batch, cond_dim], y_shape: (batch, y_dim)
        y = torch.randn(y_shape).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((y_shape[0],), t, dtype=torch.long).to(self.device)
            y = self.p_sample(x_cond, y, t_tensor)
        return y

    def train_loss(self, x_cond, y_start):
        # x_cond: [batch, cond_dim], y_start: [batch, y_dim]
        t = torch.randint(0, self.timesteps, (y_start.shape[0],), device=self.device)
        noise = torch.randn_like(y_start)
        y_noisy = self.q_sample(y_start, t, noise)
        model_in = torch.cat([x_cond, y_noisy], dim=1)
        pred_noise = self.model(model_in, t)
        return F.mse_loss(pred_noise, noise)