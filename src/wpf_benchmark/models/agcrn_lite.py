"""Historical pooled graph recurrent baseline, now registered as agcrn_lite."""
from __future__ import annotations

import math
from typing import Optional, Sequence

from .neural import NeuralForecaster
from .registry import register_model


@register_model("agcrn_lite")
class AGCRNLiteForecaster(NeuralForecaster):
    graph_model = True
    base_features = ("Wspd", "Wsin", "Wcos", "Patv")

    def __init__(self, hidden: int = 64, layers: int = 2, emb: int = 10,
                 temporal_stride: int = 6, dropout: float = 0.1,
                 lr: float = 0.001, batch: int = 16, epochs: int = 30,
                 patience: int = 5, stride: int = 6, weight_decay: float = 0.0,
                 loss: str = "mse", current_power_skip: bool = False,
                 low_power_head: bool = False, low_power_threshold_kw: float = 10.0,
                 state_loss_weight: float = 0.05, state_history_steps: int = 12,
                 input_features: Optional[Sequence[str]] = None,
                 selection_mode: str = "native",
                 selection_days: Optional[Sequence[int]] = None):
        super().__init__(hidden, layers, dropout, lr, batch, epochs, patience,
                         stride, weight_decay)
        self.emb = int(emb)
        if selection_mode not in ("native", "fixed_budget_dual_mask"):
            raise ValueError("Unknown Lite selection_mode")
        self.selection_mode = selection_mode
        self.selection_days = list(selection_days) if selection_days is not None else [197, 208]
        if (len(self.selection_days) != 2 or any(type(day) is not int for day in self.selection_days) or
                self.selection_days[0] > self.selection_days[1]):
            raise ValueError("selection_days must be an ordered pair of integer days")
        if selection_mode == "native" and selection_days is not None:
            raise ValueError("selection_days requires fixed_budget_dual_mask")
        self.temporal_stride = int(temporal_stride)
        self.loss_name = str(loss)
        if self.loss_name not in ("mae", "mse"):
            raise ValueError("AGCRN Lite loss must be 'mae' or 'mse'")
        if not isinstance(current_power_skip, bool):
            raise ValueError("current_power_skip must be a boolean")
        self.current_power_skip = current_power_skip
        if not isinstance(low_power_head, bool):
            raise ValueError("low_power_head must be a boolean")
        if low_power_head and not current_power_skip:
            raise ValueError("low_power_head requires current_power_skip=True")
        self.low_power_head = low_power_head
        self.low_power_threshold_kw = float(low_power_threshold_kw)
        self.state_loss_weight = float(state_loss_weight)
        if not math.isfinite(self.low_power_threshold_kw) or self.low_power_threshold_kw <= 0:
            raise ValueError("low_power_threshold_kw must be finite and positive")
        if not math.isfinite(self.state_loss_weight) or self.state_loss_weight < 0:
            raise ValueError("state_loss_weight must be finite and nonnegative")
        if isinstance(state_history_steps, bool) or not isinstance(state_history_steps, int) or state_history_steps <= 0:
            raise ValueError("state_history_steps must be a positive integer")
        self.state_history_steps = state_history_steps
        # The power anchor and local state head retain their established indices.
        # Additional historical channels enter only the pooled graph encoder.
        if input_features is not None:
            if not isinstance(input_features, (list, tuple)) or any(
                    not isinstance(name, str) for name in input_features):
                raise ValueError("input_features must be a list or tuple of feature names")
            chosen = tuple(input_features)
            if (chosen[:4] != self.base_features or len(set(chosen)) != len(chosen) or
                    any(name not in ("Pab1", "Pab2", "Pab3") for name in chosen[4:])):
                raise ValueError("input_features must preserve Wspd,Wsin,Wcos,Patv first, "
                                 "with distinct optional Pab1/Pab2/Pab3 channels")
            self.input_features = chosen
        else:
            self.input_features = None
        self.features = self.base_features if self.input_features is None else self.input_features
        if self.emb <= 0 or self.temporal_stride <= 0:
            raise ValueError("emb and temporal_stride must be positive")

    @property
    def model_config(self) -> dict:
        options = dict(super().model_config, emb=self.emb,
                    temporal_stride=self.temporal_stride, loss=self.loss_name,
                    current_power_skip=self.current_power_skip,
                    low_power_head=self.low_power_head,
                    low_power_threshold_kw=self.low_power_threshold_kw,
                    state_loss_weight=self.state_loss_weight,
                    state_history_steps=self.state_history_steps)
        # Keep the legacy default recipe and checkpoint payload unchanged.
        if self.input_features is not None:
            options["input_features"] = list(self.input_features)
        if self.selection_mode != "native":
            options["selection_mode"] = self.selection_mode
            options["selection_days"] = list(self.selection_days)
        return options

    def configure(self, config, scaler):
        super().configure(config, scaler)
        if self.selection_mode == "fixed_budget_dual_mask" and (
                config.validation_mask is not None or config.m2_extra_target_weight != 1 or
                self.loss_name != "mae"):
            raise ValueError("Fixed dual selection requires MAE, unit weights and no validation_mask")
        if self.selection_mode == "fixed_budget_dual_mask" and not (
                config.train_days < self.selection_days[0] <= self.selection_days[1] <=
                config.train_days + config.val_days):
            raise ValueError("Selection days must lie within the development period")
        if self.low_power_head and not self.low_power_threshold_kw < config.rated_power_kw:
            raise ValueError("low_power_threshold_kw must be below rated_power_kw")

    def _normalized_power(self, kw):
        j = self.features.index("Patv")
        span = max(self.scaler.maximum[j] - self.scaler.minimum[j], 1e-6)
        return float((kw - self.scaler.minimum[j]) / span)

    def _loss_sums(self, histories, target, mask):
        if not self.low_power_head:
            return super()._loss_sums(histories, target, mask)
        from torch.nn.functional import binary_cross_entropy_with_logits

        outputs = self.network(histories, return_aux=True)
        sums = self._power_loss_sums(outputs["power"], target, mask)
        selected = mask > 0 if mask.is_floating_point() else mask
        labels = (target[selected] < self._normalized_power(self.low_power_threshold_kw)).to(target.dtype)
        state_loss = binary_cross_entropy_with_logits(outputs["state_logits"][selected], labels,
                         reduction="none" if mask.is_floating_point() else "sum")
        if mask.is_floating_point():
            state_loss = (state_loss * mask[selected]).sum()
        sums["state_bce"] = state_loss
        sums["total"] = sums[self.loss_name] + self.state_loss_weight * state_loss
        return sums

    def _training_log_lines(self):
        inputs = (["input_features={} input_size={}".format(
            ",".join(self.features), len(self.features))] if self.input_features is not None else [])
        if not self.low_power_head:
            return inputs
        return inputs + ["low_power_head=True low_power_threshold_kw={} state_loss_weight={} "
                "state_history_steps={} early_stopping={} state_target=power_lt_threshold "
                "state_mask=target_mask".format(self.low_power_threshold_kw, self.state_loss_weight,
                                               self.state_history_steps, self.loss_name)]

    def _build_network(self, torch, n_turbines: int):
        from .networks import AGCRNLiteNetwork

        return AGCRNLiteNetwork(n_turbines, self.hidden, self.layers, self.emb,
                                self.config.horizon, self.temporal_stride,
                                current_power_skip=self.current_power_skip,
                                power_fallback=self._normalized_zero_power(),
                                low_power_head=self.low_power_head,
                                low_power_threshold=self._normalized_power(self.low_power_threshold_kw),
                                rated_power=self._normalized_power(self.config.rated_power_kw),
                                state_history_steps=self.state_history_steps,
                                input_size=len(self.features))
