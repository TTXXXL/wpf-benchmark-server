"""Per-turbine multi-output ridge regression on normalized power windows."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register_model
from .training import issue_indices, training_cube


@register_model("linear")
class LinearForecaster(BaseForecaster):
    features = ("Patv",)
    history_scale = "normalized"

    def __init__(self, alpha: float = 1.0, stride: int = 6):
        if alpha < 0 or stride <= 0:
            raise ValueError("alpha must be nonnegative and stride positive")
        self.alpha = float(alpha)
        self.stride = int(stride)

    @property
    def model_config(self) -> dict:
        return {"alpha": self.alpha, "stride": self.stride,
                "fallback_turbines": getattr(self, "fallback_turbines", [])}

    def fit(self, train: pd.DataFrame, valid: Optional[pd.DataFrame] = None) -> None:
        _, self.turbine_ids, cube, mask, _ = training_cube(
            train, self.features, self.config, self.scaler)
        power = cube[:, :, 0]
        w, h = self.config.input_window, self.config.horizon
        issues = issue_indices(len(cube), w, h, self.stride)
        if not len(issues):
            raise ValueError("Training split is too short for linear windows")
        future = issues[:, None] + np.arange(1, h + 1)[None, :]
        self.weights = np.empty((len(self.turbine_ids), w + 1, h), np.float32)
        self.fallback_turbines = []
        penalty = np.eye(w + 1, dtype=np.float64) * self.alpha
        penalty[-1, -1] = 0
        for j in range(len(self.turbine_ids)):
            x = np.lib.stride_tricks.sliding_window_view(power[:, j], w)[issues - w + 1]
            y = power[future, j]
            good = mask[future, j].all(axis=1) & np.isfinite(y).all(axis=1)
            if not good.any():
                # Some turbines have no fully valid H-step training window.
                # Use normalized persistence instead of training on flagged targets.
                self.weights[j] = 0
                self.weights[j, -2, :] = 1
                self.fallback_turbines.append(int(self.turbine_ids[j]))
                continue
            x = np.nan_to_num(x[good]).astype(np.float64)
            x = np.column_stack((x, np.ones(len(x))))
            y = y[good].astype(np.float64)
            self.weights[j] = np.linalg.solve(x.T @ x + penalty, x.T @ y).astype(np.float32)
        self.n_params = int((len(self.turbine_ids) - len(self.fallback_turbines)) *
                            (w + 1) * h)

    def predict(self, history: np.ndarray) -> np.ndarray:
        return self.predict_batch(history[None])[0]

    def predict_batch(self, histories: np.ndarray) -> np.ndarray:
        if histories.shape[1] != len(self.turbine_ids):
            raise ValueError("Turbine count differs from fitted coefficients")
        x = np.nan_to_num(histories[:, :, 0, :], nan=0.0, posinf=0.0, neginf=0.0)
        result = np.einsum("bnw,nwh->bnh", x, self.weights[:, :-1, :]) + self.weights[None, :, -1, :]
        low = self.scaler.minimum[0]
        span = max(self.scaler.maximum[0] - low, 1e-6)
        return np.clip(result * span + low, 0, self.config.rated_power_kw).astype(np.float32)
