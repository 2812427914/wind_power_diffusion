import torch
import torch.nn as nn

class GANGenerator(nn.Module):
    def __init__(self, cond_dim, latent_dim, hidden_dim=128, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z, c):
        # z: [batch, latent_dim], c: [batch, cond_dim]
        net_in = torch.cat((z, c), dim=1)
        return self.net(net_in)

class GANDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)