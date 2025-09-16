import torch
import torch.nn as nn

class TransformerAnomalyDetector(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super(TransformerAnomalyDetector, self).__init__()
        self.embedding = nn.Linear(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, n_features)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.decoder(x)
