"""Causal power anchor shared by the two optional graph residual paths."""
from __future__ import annotations

import torch


def latest_history_power(histories, power_index: int = 3, fallback: float = 0.0):
    """Latest finite normalized Patv in (B,N,F,W), or normalized zero kW.

    Select before sanitization and temporal pooling. No future observations,
    spatial averages, or zero-filled missing cells can become the anchor.
    """
    power = histories[:, :, power_index, :]
    finite = torch.isfinite(power)
    ticks = torch.arange(power.shape[-1], device=power.device)
    last = torch.where(finite, ticks, ticks.new_full((), -1)).max(dim=-1).values
    safe_power = torch.where(finite, power, torch.zeros_like(power))
    selected = safe_power.gather(-1, last.clamp(min=0).unsqueeze(-1))
    return torch.where(last.unsqueeze(-1) >= 0, selected,
                       torch.full_like(selected, fallback))
