import torch.nn as nn

from config import NUM_FEATURES, LSTM_IMP_HIDDEN_1, LSTM_IMP_HIDDEN_2, LSTM_IMP_DROPOUT


class LSTMImproved(nn.Module):

    def __init__(
        self,
        n_features=NUM_FEATURES,
        hidden1=LSTM_IMP_HIDDEN_1,
        hidden2=LSTM_IMP_HIDDEN_2,
        dropout=LSTM_IMP_DROPOUT,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out.mean(dim=1))
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)

    @staticmethod
    def loss_fn():
        return nn.BCEWithLogitsLoss()
