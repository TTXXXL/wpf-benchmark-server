"""The historical graph recurrent network retained for lite and PIN runs.

This module is imported only when a deep model is built, so PyTorch remains an
optional dependency for the rest of the benchmark.
"""
from __future__ import annotations

import torch
from torch import nn

from .current_power import latest_history_power


class GraphCell(nn.Module):
    def __init__(self, input_size: int, hidden: int):
        super().__init__()
        self.gates = nn.Linear(input_size + 2 * hidden, 2 * hidden)
        self.candidate = nn.Linear(input_size + 2 * hidden, hidden)

    def forward(self, x, state, adjacency):
        neighbor = torch.einsum("ij,bjh->bih", adjacency, state)
        gates = self.gates(torch.cat((x, state, neighbor), dim=-1)).sigmoid()
        reset, update = gates.chunk(2, dim=-1)
        candidate = torch.tanh(self.candidate(
            torch.cat((x, reset * state, neighbor), dim=-1)))
        return update * state + (1 - update) * candidate


class AGCRNLiteNetwork(nn.Module):
    """The established pooled graph decoder, optionally with a wind head."""

    def __init__(self, n_turbines: int, hidden: int, layers: int, emb: int,
                 horizon: int, temporal_stride: int, output_wind: bool = False,
                 current_power_skip: bool = False, power_fallback: float = 0.0):
        super().__init__()
        self.hidden = hidden
        self.horizon = horizon
        self.temporal_stride = temporal_stride
        self.output_wind = output_wind
        self.current_power_skip = current_power_skip
        self.power_fallback = power_fallback
        self.node_left = nn.Parameter(torch.randn(n_turbines, emb) * 0.1)
        self.node_right = nn.Parameter(torch.randn(emb, n_turbines) * 0.1)
        self.cells = nn.ModuleList([GraphCell(4 if i == 0 else hidden, hidden)
                                    for i in range(layers)])
        self.decoder = GraphCell(1, hidden)
        self.readout = nn.Linear(hidden, 1)
        if self.current_power_skip:
            nn.init.zeros_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)
        if output_wind:
            self.wind_readout = nn.Linear(hidden, 1)

    def adaptive_adjacency(self):
        return torch.softmax(torch.relu(self.node_left @ self.node_right), dim=-1)

    def forward(self, x):
        # Capture the unpooled issue-time power before any missing-value fill.
        if self.current_power_skip:
            anchor = latest_history_power(x, fallback=self.power_fallback)
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        b0, n0, f0, w0 = x.shape
        step = min(self.temporal_stride, w0)
        pooled = torch.nn.functional.avg_pool1d(
            x.reshape(b0 * n0 * f0, 1, w0), kernel_size=step, stride=step)
        x = pooled.reshape(b0, n0, f0, -1).permute(0, 3, 1, 2)
        b, _, n, _ = x.shape
        adjacency = self.adaptive_adjacency()
        states = [x.new_zeros(b, n, self.hidden) for _ in self.cells]
        for tick in range(x.shape[1]):
            value = x[:, tick]
            for i, cell in enumerate(self.cells):
                states[i] = cell(value, states[i], adjacency)
                value = states[i]
        state = states[-1]
        previous = anchor if self.current_power_skip else x[:, -1, :, 3:4]
        forecasts = []
        wind_forecasts = []
        for _ in range(self.horizon):
            state = self.decoder(previous, state, adjacency)
            residual = self.readout(state)
            # Each horizon is relative to the same P(t), not a cumulative delta.
            previous = anchor + residual if self.current_power_skip else residual
            forecasts.append(previous[..., 0])
            if self.output_wind:
                wind_forecasts.append(self.wind_readout(state)[..., 0])
        power = torch.stack(forecasts, dim=-1)
        if not self.output_wind:
            return power
        wind = torch.stack(wind_forecasts, dim=-1)
        return torch.cat((power, wind), dim=-1)
