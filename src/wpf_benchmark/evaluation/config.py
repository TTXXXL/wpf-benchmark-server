"""Forecast task and evaluation thresholds."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProtocolConfig:
    input_window: int = 144
    horizon: int = 12
    train_days: int = 196
    val_days: int = 25
    steps_per_day: int = 144
    exclude_flags_main: tuple = ("m_missing", "m_imputed", "m_outlier",
                                 "f_fault", "f_curtail", "f_farm")
    rated_power_kw: float = 1550.0
    farm_min_valid_frac: float = 0.85
    wind_bins: tuple = ((0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 40))
    stability_block_days: int = 7
    ramp_threshold_frac: float = 0.10
    ramp_window_steps: int = 6
    high_wind_speed: float = 18.0
    violation_sigma: float = 3.0
    curve_bin_width: float = 1.0
    curve_min_samples: int = 30
    cut_in: float = 3.0
    cut_out: float = 25.0

    def __post_init__(self) -> None:
        if min(self.input_window, self.horizon, self.train_days, self.val_days,
               self.steps_per_day, self.stability_block_days, self.ramp_window_steps,
               self.curve_min_samples) <= 0:
            raise ValueError("Window lengths, day counts and sample counts must be positive")
        if self.rated_power_kw <= 0 or self.curve_bin_width <= 0 or self.violation_sigma <= 0:
            raise ValueError("Power rating, curve width and sigma multiplier must be positive")
        if not 0 <= self.farm_min_valid_frac <= 1:
            raise ValueError("farm_min_valid_frac must lie in [0, 1]")

    def ramp_threshold_kw(self) -> float:
        return self.ramp_threshold_frac * self.rated_power_kw
