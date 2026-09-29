"""Persistence baseline used to validate the model extension path."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register_model


@register_model("persistence")
class PersistenceForecaster(BaseForecaster):
    features = ("Patv",)

    def fit(self, train: pd.DataFrame, valid: Optional[pd.DataFrame] = None) -> None:
        return None

    def predict(self, history: np.ndarray) -> np.ndarray:
        last = history[:, 0, -1]
        return np.repeat(last[:, None], self.config.horizon, axis=1)

    def predict_batch(self, histories: np.ndarray) -> np.ndarray:
        last = histories[:, :, 0, -1]
        return np.repeat(last[:, :, None], self.config.horizon, axis=2)
