"""Historical pooled graph recurrent baseline, now registered as agcrn_lite."""
from __future__ import annotations

from .neural import NeuralForecaster
from .registry import register_model


@register_model("agcrn_lite")
class AGCRNLiteForecaster(NeuralForecaster):
    graph_model = True

    def __init__(self, hidden: int = 64, layers: int = 2, emb: int = 10,
                 temporal_stride: int = 6, dropout: float = 0.1,
                 lr: float = 0.001, batch: int = 16, epochs: int = 30,
                 patience: int = 5, stride: int = 6, weight_decay: float = 0.0,
                 loss: str = "mse", current_power_skip: bool = False):
        super().__init__(hidden, layers, dropout, lr, batch, epochs, patience,
                         stride, weight_decay)
        self.emb = int(emb)
        self.temporal_stride = int(temporal_stride)
        self.loss_name = str(loss)
        if self.loss_name not in ("mae", "mse"):
            raise ValueError("AGCRN Lite loss must be 'mae' or 'mse'")
        if not isinstance(current_power_skip, bool):
            raise ValueError("current_power_skip must be a boolean")
        self.current_power_skip = current_power_skip
        if self.emb <= 0 or self.temporal_stride <= 0:
            raise ValueError("emb and temporal_stride must be positive")

    @property
    def model_config(self) -> dict:
        return dict(super().model_config, emb=self.emb,
                    temporal_stride=self.temporal_stride, loss=self.loss_name,
                    current_power_skip=self.current_power_skip)

    def _build_network(self, torch, n_turbines: int):
        from .networks import AGCRNLiteNetwork

        return AGCRNLiteNetwork(n_turbines, self.hidden, self.layers, self.emb,
                                self.config.horizon, self.temporal_stride,
                                current_power_skip=self.current_power_skip,
                                power_fallback=self._normalized_zero_power())
