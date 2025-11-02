import torch
import torch.nn as nn

class SeqLSTMModel(nn.Module):
    def __init__(self, feature_dim=1, hist_len=24, seq_len=24, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hist_len = hist_len
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True, dropout=dropout, num_layers=1)
        self.fc = nn.Linear(hidden_dim, seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, hist_len, feature_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.dropout(out)
        out = self.fc(out)   # (batch, seq_len)
        return out

    def loss_function(self, x, y):
        y_pred = self.forward(x)
        return torch.nn.functional.mse_loss(y_pred, y)