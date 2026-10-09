"""Matched shared linear/TCN forecasts of increments from a causal power anchor."""
from __future__ import annotations

from .neural import NeuralForecaster
from .registry import register_model


class IncrementForecaster(NeuralForecaster):
    loss_name = "mae"
    current_power_skip = True
    architecture = "linear"

    def __init__(self, input_mode="power", hidden=32, layers=6, kernel_size=3,
                 dropout=0.0, lr=0.001, batch=256, epochs=30, patience=5,
                 stride=6, weight_decay=0.0):
        if input_mode not in ("power", "power_wind"):
            raise ValueError("input_mode must be power or power_wind")
        if type(kernel_size) is not int or kernel_size < 2:
            raise ValueError("kernel_size must be an integer >= 2")
        super().__init__(hidden, layers, dropout, lr, batch, epochs, patience,
                         stride, weight_decay)
        self.input_mode = input_mode
        self.kernel_size = kernel_size

    @property
    def model_config(self):
        return dict(super().model_config, input_mode=self.input_mode,
                    kernel_size=self.kernel_size)

    @property
    def receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.layers - 1)

    def _training_log_lines(self):
        return ["architecture={} input_mode={} shared_turbine_parameters=True "
                "power_centering=anchor output=direct_signed_increments "
                "receptive_field={}".format(self.architecture, self.input_mode,
                    self.receptive_field if self.architecture == "tcn" else "full_window")]

    def _build_network(self, torch, n_turbines):
        if self.architecture == "tcn" and self.receptive_field < self.config.input_window:
            raise ValueError("TCN receptive field does not cover input_window")
        nn = torch.nn
        window, horizon = self.config.input_window, self.config.horizon
        mode, zero = self.input_mode, self._normalized_zero_power()
        hidden, layers, kernel, dropout = (self.hidden, self.layers,
                                           self.kernel_size, self.dropout)
        architecture = self.architecture

        class CausalBlock(nn.Module):
            def __init__(self, dilation):
                super().__init__()
                self.padding = (kernel - 1) * dilation
                self.conv1 = nn.Conv1d(hidden, hidden, kernel, dilation=dilation)
                self.conv2 = nn.Conv1d(hidden, hidden, kernel, dilation=dilation)
                self.dropout = nn.Dropout(dropout)

            def forward(self, value):
                residual = value
                for convolution in (self.conv1, self.conv2):
                    value = convolution(torch.nn.functional.pad(value, (self.padding, 0)))
                    value = self.dropout(torch.relu(value))
                return torch.relu(value + residual)

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                if architecture == "linear":
                    self.head = nn.Linear(4 * window, horizon)
                else:
                    self.projection = nn.Conv1d(4, hidden, 1)
                    self.blocks = nn.Sequential(*[CausalBlock(2 ** i) for i in range(layers)])
                    self.head = nn.Linear(hidden, horizon)
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)

            def prepare(self, histories):
                if histories.ndim != 3 or histories.shape[1:] != (4, window):
                    raise ValueError("Expected (batch,4,input_window) histories")
                power = histories[:, 3]
                finite = torch.isfinite(power)
                ticks = torch.arange(window, device=power.device)
                last = torch.where(finite, ticks, ticks.new_full((), -1)).max(-1).values
                safe_power = torch.where(finite, power, torch.full_like(power, zero))
                anchor = safe_power.gather(-1, last.clamp(min=0)[:, None])
                anchor = torch.where(last[:, None] >= 0, anchor, torch.full_like(anchor, zero))
                value = torch.where(torch.isfinite(histories), histories,
                                    torch.zeros_like(histories)).clone()
                # Missing power conveys no invented departure from the anchor.
                value[:, 3] = torch.where(finite, power - anchor, torch.zeros_like(power))
                if mode == "power":
                    value[:, :3] = 0
                return value, anchor

            def forward(self, histories):
                value, anchor = self.prepare(histories)
                if architecture == "linear":
                    increment = self.head(value.flatten(1))
                else:
                    encoded = self.blocks(self.projection(value))
                    increment = self.head(encoded[:, :, -1])
                return anchor + increment

        return Network()


@register_model("increment_linear")
class LinearIncrementForecaster(IncrementForecaster):
    architecture = "linear"


@register_model("increment_tcn")
class TCNIncrementForecaster(IncrementForecaster):
    architecture = "tcn"
