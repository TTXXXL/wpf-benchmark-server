"""Causal, per-horizon low-power mixture head for the Lite decoder."""
from __future__ import annotations

import torch
from torch import nn


class LowPowerHead(nn.Module):
    def __init__(self, hidden: int, history_steps: int, zero: float,
                 threshold: float, rated: float):
        super().__init__()
        if history_steps <= 0 or not zero < threshold < rated:
            raise ValueError("Low-power head requires history_steps > 0 and zero < threshold < rated")
        self.history_steps = history_steps
        self.zero = zero
        self.threshold = threshold
        self.rated = rated
        # Eight local history features, decoder context, and relative horizon.
        self.context = nn.Sequential(nn.Linear(hidden + 9, hidden), nn.Tanh())
        self.state_readout = nn.Linear(hidden, 1)
        self.low_readout = nn.Linear(hidden, 1)
        # Start near the residual branch, without freezing the state classifier.
        nn.init.zeros_(self.state_readout.weight)
        nn.init.constant_(self.state_readout.bias, -4.0)

    def history_features(self, histories):
        """Summarize original-resolution history before pooling or filling.

        Features: latest power/wind, endpoint slopes per tick, mean power,
        trailing low-power duration, and power/wind observation fractions.
        Missing ticks break the trailing run; an entirely missing series has
        zero slope, zero observation fraction, and a finite fallback.
        """
        recent = histories[..., -self.history_steps:]
        power, wind = recent[:, :, 3], recent[:, :, 0]

        def summarize(values, fallback):
            finite = torch.isfinite(values)
            ticks = torch.arange(values.shape[-1], device=values.device)
            first = torch.where(finite, ticks, ticks.new_full((), values.shape[-1])).min(-1).values
            last = torch.where(finite, ticks, ticks.new_full((), -1)).max(-1).values
            safe = torch.where(finite, values, torch.zeros_like(values))
            start = safe.gather(-1, first.clamp(max=values.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
            end = safe.gather(-1, last.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            count = finite.sum(-1)
            latest = torch.where(count > 0, end, torch.full_like(end, fallback))
            slope = torch.where(count > 1, (end - start) / (last - first).clamp(min=1),
                                torch.zeros_like(end))
            mean = torch.where(count > 0, safe.sum(-1) / count.clamp(min=1),
                               torch.full_like(end, fallback))
            return latest, slope, mean, finite.to(values.dtype).mean(-1)

        p, dp, mean_p, p_fraction = summarize(power, self.zero)
        w, dw, _, w_fraction = summarize(wind, 0.0)
        low = torch.isfinite(power) & (power < self.threshold)
        duration = low.flip(-1).to(power.dtype).cumprod(-1).sum(-1) / power.shape[-1]
        return torch.stack((p, w, dp, dw, mean_p, duration, p_fraction, w_fraction), dim=-1)

    def forward(self, state, local, relative_horizon: float, normal):
        horizon = state.new_full((*state.shape[:-1], 1), relative_horizon)
        context = self.context(torch.cat((state, local, horizon), dim=-1))
        logits = self.state_readout(context)
        low = self.zero + (self.threshold - self.zero) * self.low_readout(context).sigmoid()
        normal = normal.clamp(min=self.zero, max=self.rated)
        q = logits.sigmoid()
        power = q * low + (1.0 - q) * normal
        return power, logits, low, normal
