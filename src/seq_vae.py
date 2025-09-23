import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqVAE(nn.Module):
    def __init__(self, feature_dim, hist_len, y_dim, seq_len=24, latent_dim=8, hidden_dim=128, num_layers=1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hist_len = hist_len
        self.y_dim = y_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # 编码器：LSTM编码历史序列
        self.encoder_lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器：LSTM解码未来序列
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder_out = nn.Sequential(
            nn.Linear(hidden_dim, y_dim),
            nn.ReLU()
        )

    def encode(self, x_hist):
        # x_hist: (batch, hist_len, feature_dim)
        _, (h_n, _) = self.encoder_lstm(x_hist)
        h_last = h_n[-1]  # (batch, hidden_dim)
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z: (batch, latent_dim)
        dec_in = self.decoder_input(z).unsqueeze(1)  # (batch, 1, hidden_dim)
        dec_in = dec_in.repeat(1, self.seq_len, 1)   # (batch, seq_len, hidden_dim)
        dec_out, _ = self.decoder_lstm(dec_in)
        y_pred = self.decoder_out(dec_out)  # (batch, seq_len, y_dim)
        return y_pred.squeeze(-1)  # (batch, seq_len)

    def forward(self, x_hist, y_future=None):
        mu, logvar = self.encode(x_hist)
        z = self.reparameterize(mu, logvar)
        y_pred = self.decode(z)
        return y_pred, mu, logvar

    def loss_function(self, recon_y, y, mu, logvar):
        # recon_y, y: (batch, seq_len)
        recon_loss = F.mse_loss(recon_y, y, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld