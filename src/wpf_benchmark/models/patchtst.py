"""Channel-independent patch Transformer with instance normalization."""
from __future__ import annotations

from .neural import NeuralForecaster
from .registry import register_model


@register_model("patchtst")
class PatchTSTForecaster(NeuralForecaster):
    def __init__(self, hidden: int = 128, layers: int = 2, patch: int = 12,
                 patch_stride: int = 6, heads: int = 4, dropout: float = 0.1,
                 lr: float = 0.001, batch: int = 256, epochs: int = 30,
                 patience: int = 5, stride: int = 6, weight_decay: float = 0.0):
        super().__init__(hidden, layers, dropout, lr, batch, epochs, patience,
                         stride, weight_decay)
        self.patch = int(patch)
        self.patch_stride = int(patch_stride)
        self.heads = int(heads)
        if min(self.patch, self.patch_stride, self.heads) <= 0 or hidden % heads:
            raise ValueError("Invalid patch geometry or attention heads")

    @property
    def model_config(self) -> dict:
        return dict(super().model_config, patch=self.patch,
                    patch_stride=self.patch_stride, heads=self.heads)

    def _build_network(self, torch, n_turbines: int):
        nn = torch.nn
        w, h = self.config.input_window, self.config.horizon
        if self.patch > w:
            raise ValueError("patch must not exceed input_window")
        count = 1 + (w - self.patch) // self.patch_stride
        patch, step, dim = self.patch, self.patch_stride, self.hidden
        heads, layers, dropout = self.heads, self.layers, self.dropout

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(patch, dim)
                self.position = nn.Parameter(torch.zeros(1, count, dim))
                block = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                                   dim_feedforward=4 * dim,
                                                   dropout=dropout,
                                                   batch_first=True)
                self.encoder = nn.TransformerEncoder(block, num_layers=layers)
                self.head = nn.Linear(4 * count * dim, h)

            def forward(self, x):
                b, channels, _ = x.shape
                mean = x.mean(dim=-1, keepdim=True)
                std = x.var(dim=-1, unbiased=False, keepdim=True).add(1e-5).sqrt()
                x = (x - mean) / std
                patches = x.unfold(-1, patch, step).reshape(b * channels, count, patch)
                encoded = self.encoder(self.projection(patches) + self.position)
                output = self.head(encoded.reshape(b, channels * count * dim))
                return output * std[:, 3, :] + mean[:, 3, :]

        return Network()
