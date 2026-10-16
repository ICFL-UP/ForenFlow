import math
import torch
import torch.nn as nn

from config import NUM_FEATURES, TF_D_MODEL, TF_NHEAD, TF_NUM_LAYERS, TF_DIM_FF, TF_DROPOUT


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):

    def __init__(
        self,
        n_features: int = NUM_FEATURES,
        d_model: int = TF_D_MODEL,
        nhead: int = TF_NHEAD,
        num_layers: int = TF_NUM_LAYERS,
        dim_ff: int = TF_DIM_FF,
        dropout: float = TF_DROPOUT,
    ):
        super().__init__()
        if d_model % nhead != 0:
            nhead = 1

        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, 1),
        )

    def forward(self, x):
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.classifier(x).squeeze(-1)

    @staticmethod
    def loss_fn():
        return nn.BCEWithLogitsLoss()
