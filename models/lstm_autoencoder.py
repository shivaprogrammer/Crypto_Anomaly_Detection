import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=32, dropout=0.3):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            batch_first=True
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = self.dropout(h)
        h = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        out, _ = self.decoder(h)
        return out
