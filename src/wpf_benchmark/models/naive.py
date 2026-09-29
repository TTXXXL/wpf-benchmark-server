"""Deterministic physical-unit baselines."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register_model


@register_model("seasonal_persistence")
class SeasonalPersistenceForecaster(BaseForecaster):
    """Repeat the values at the same time one day earlier."""

    def fit(self, train: pd.DataFrame, valid: Optional[pd.DataFrame] = None) -> None:
        if self.config.input_window < self.config.steps_per_day:
            raise ValueError("Seasonal persistence needs a full preceding day and horizon")

    def predict(self, history: np.ndarray) -> np.ndarray:
        return self.predict_batch(history[None])[0]

    def predict_batch(self, histories: np.ndarray) -> np.ndarray:
        day = self.config.steps_per_day
        offsets = np.arange(1, self.config.horizon + 1) - day - 1
        return histories[:, :, 0, offsets].astype(np.float32)


@register_model("climatology")
class ClimatologyForecaster(BaseForecaster):
    """Mean training power for each turbine and time-of-day phase."""

    needs_phases = True

    def fit(self, train: pd.DataFrame, valid: Optional[pd.DataFrame] = None) -> None:
        flags = self.config.exclude_flags_main
        good = np.isfinite(train["Patv"].to_numpy(dtype=float))
        for flag in flags:
            good &= ~train[flag].to_numpy(dtype=bool)
        sample = train.loc[good, ["TurbID", "ts", "Patv"]].copy()
        if sample.empty:
            raise ValueError("No valid training rows for climatology")
        sample["phase"] = (sample["ts"].dt.total_seconds().to_numpy() // 600).astype(int) % self.config.steps_per_day
        self.turbine_ids = np.sort(train["TurbID"].unique())
        global_mean = float(sample["Patv"].mean())
        self.lookup = np.full((len(self.turbine_ids), self.config.steps_per_day),
                              global_mean, dtype=np.float32)
        grouped = sample.groupby(["TurbID", "phase"])["Patv"].mean()
        for (tid, phase), mean in grouped.items():
            index = np.searchsorted(self.turbine_ids, tid)
            self.lookup[index, int(phase)] = mean
        self.n_params = int(self.lookup.size)

    def predict(self, history: np.ndarray) -> np.ndarray:
        raise ValueError("Climatology requires issue-time phases; use predict_batch")

    def predict_batch(self, histories: np.ndarray,
                      phases: Optional[np.ndarray] = None) -> np.ndarray:
        if phases is None or len(phases) != len(histories):
            raise ValueError("Climatology requires one phase per issue time")
        if histories.shape[1] != len(self.turbine_ids):
            raise ValueError("Turbine count differs from the fitted lookup")
        values = self.lookup[:, np.asarray(phases)].T
        return np.repeat(values[:, :, None], self.config.horizon, axis=2).astype(np.float32)


@register_model("trend_persistence")
class TrendPersistenceForecaster(BaseForecaster):
    def __init__(self, k: int = 6):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)

    @property
    def model_config(self) -> dict:
        return {"k": self.k}

    def fit(self, train: pd.DataFrame, valid: Optional[pd.DataFrame] = None) -> None:
        if self.k >= self.config.input_window:
            raise ValueError("k must be shorter than input_window")

    def predict(self, history: np.ndarray) -> np.ndarray:
        return self.predict_batch(history[None])[0]

    def predict_batch(self, histories: np.ndarray) -> np.ndarray:
        last = histories[:, :, 0, -1]
        past = histories[:, :, 0, -1 - self.k]
        steps = np.arange(1, self.config.horizon + 1, dtype=np.float32)
        result = last[:, :, None] + (last - past)[:, :, None] * steps / self.k
        return np.clip(result, 0, self.config.rated_power_kw).astype(np.float32)


@register_model("farm_mean_persistence")
class FarmMeanPersistenceForecaster(TrendPersistenceForecaster):
    """Scale each turbine's latest power by the recent farm-wide power ratio.

    A zero earlier farm mean has no stable ratio, so that issue uses ratio 1.
    """

    def predict_batch(self, histories: np.ndarray) -> np.ndarray:
        last = histories[:, :, 0, -1]
        past = histories[:, :, 0, -1 - self.k]
        current_mean = np.nanmean(last, axis=1)
        past_mean = np.nanmean(past, axis=1)
        ratio = np.divide(current_mean, past_mean,
                          out=np.ones_like(current_mean),
                          where=np.abs(past_mean) > 1e-6)
        values = np.clip(last * ratio[:, None], 0, self.config.rated_power_kw)
        return np.repeat(values[:, :, None], self.config.horizon, axis=2).astype(np.float32)
