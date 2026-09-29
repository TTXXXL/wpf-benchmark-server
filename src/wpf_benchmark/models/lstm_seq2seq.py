"""Two-layer LSTM encoder with a multi-step decoder head."""
from __future__ import annotations

from .neural import NeuralForecaster
from .registry import register_model


@register_model("lstm_seq2seq")
class LSTMSeq2SeqForecaster(NeuralForecaster):
    def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.1,
                 lr: float = 0.001, batch: int = 256, epochs: int = 30,
                 patience: int = 5, stride: int = 6, weight_decay: float = 0.0):
        super().__init__(hidden, layers, dropout, lr, batch, epochs, patience,
                         stride, weight_decay)

    def _build_network(self, torch, n_turbines: int):
        nn = torch.nn
        hidden, layers, horizon = self.hidden, self.layers, self.config.horizon

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.LSTM(4, hidden, num_layers=layers,
                                       dropout=self_dropout if layers > 1 else 0,
                                       batch_first=True)
                self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                             nn.Linear(hidden, horizon))

            def forward(self, x):
                _, (state, _) = self.encoder(x.transpose(1, 2))
                return self.decoder(state[-1])

        self_dropout = self.dropout
        return Network()
