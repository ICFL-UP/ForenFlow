import torch.nn as nn

from config import NUM_FEATURES, LSTM_BASE_HIDDEN, LSTM_BASE_LAYERS, LSTM_BASE_DROPOUT


class LSTMBaseline(nn.Module):

    def __init__(
        self,
        n_features=NUM_FEATURES,
        hidden=LSTM_BASE_HIDDEN,
        num_layers=LSTM_BASE_LAYERS,
        dropout=LSTM_BASE_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.fc(out).squeeze(-1)

    @staticmethod
    def loss_fn():
        return nn.BCEWithLogitsLoss()
