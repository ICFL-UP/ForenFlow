import torch
import torch.nn as nn

from config import NUM_FEATURES, SEQUENCE_LENGTH


class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_size=NUM_FEATURES,
        hidden_size=64,
        bottleneck_size=32,
        sequence_length=SEQUENCE_LENGTH,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.sequence_length = sequence_length

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.encoder_fully_connected = nn.Linear(hidden_size, bottleneck_size)

        self.decoder_fully_connected = nn.Linear(bottleneck_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output_fully_connected = nn.Linear(hidden_size, input_size)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name and "lstm" in name:
                param.data.fill_(0)
                # Forget gate bias = 1.0 to prevent early forgetting
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for fc in [self.encoder_fully_connected,
                   self.decoder_fully_connected,
                   self.output_fully_connected]:
            nn.init.xavier_uniform_(fc.weight)
            fc.bias.data.fill_(0)

    def encode(self, x):
        _, (h, _) = self.encoder_lstm(x)
        return torch.tanh(self.encoder_fully_connected(h.squeeze(0)))

    def decode(self, z):
        h = torch.relu(self.decoder_fully_connected(z))
        repeated = h.unsqueeze(1).repeat(1, self.sequence_length, 1)
        out, _ = self.decoder_lstm(repeated)
        return self.output_fully_connected(out)

    def forward(self, x):
        return self.decode(self.encode(x))

    def reconstruction_error(self, x):
        return ((self.forward(x) - x) ** 2).mean(dim=(1, 2))

    @staticmethod
    def loss_fn():
        return nn.MSELoss()
