"""AGCRN paper architecture adapted to the common wind forecasting protocol."""
from __future__ import annotations

from .neural import NeuralForecaster
from .registry import register_model


@register_model("agcrn")
class AGCRNForecaster(NeuralForecaster):
    """NAPL + DAGG graph recurrence and a direct multi-horizon forecast head."""

    graph_model = True
    loss_name = "mae"

    def __init__(self, hidden: int = 64, layers: int = 2, emb: int = 10,
                 cheb_k: int = 2, loss: str = "mae", dropout: float = 0.0,
                 lr: float = 0.001, batch: int = 16, epochs: int = 30,
                 patience: int = 5, stride: int = 6, weight_decay: float = 0.0):
        super().__init__(hidden, layers, dropout, lr, batch, epochs, patience,
                         stride, weight_decay)
        self.emb = int(emb)
        self.cheb_k = int(cheb_k)
        self.loss_name = str(loss)
        if self.emb <= 0 or self.cheb_k < 2:
            raise ValueError("emb must be positive and cheb_k must be at least 2")
        if self.loss_name not in ("mae", "mse"):
            raise ValueError("AGCRN loss must be 'mae' or 'mse'")

    @property
    def model_config(self) -> dict:
        return dict(super().model_config, emb=self.emb,
                    cheb_k=self.cheb_k, loss=self.loss_name)

    def _build_network(self, torch, n_turbines: int):
        from .agcrn_original import PaperAGCRNNetwork
        return PaperAGCRNNetwork(n_turbines, len(self.features), self.hidden,
                                 self.layers, self.emb, self.cheb_k,
                                 self.config.horizon)
