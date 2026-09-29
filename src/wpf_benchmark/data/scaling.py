"""Feature scales fitted by preprocessing on training days only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureScaler:
    features: tuple
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def from_meta(cls, meta: Dict[str, Any], features: Sequence[str]) -> "FeatureScaler":
        if "scale" not in meta:
            raise ValueError("Preprocessing metadata has no training scales")
        names = tuple(features)
        try:
            minimum = np.asarray([meta["scale"][name]["min"] for name in names], dtype=np.float32)
            maximum = np.asarray([meta["scale"][name]["max"] for name in names], dtype=np.float32)
        except KeyError as exc:
            raise ValueError("Feature has no training scale: " + str(exc)) from exc
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError("Training scales must be finite")
        return cls(names, minimum, maximum)

    @classmethod
    def from_training(cls, train: pd.DataFrame, features: Sequence[str]) -> "FeatureScaler":
        names = tuple(features)
        minimum = train.loc[:, list(names)].min().to_numpy(dtype=np.float32)
        maximum = train.loc[:, list(names)].max().to_numpy(dtype=np.float32)
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError("Training data cannot define finite feature scales")
        return cls(names, minimum, maximum)

    def transform_histories(self, histories: np.ndarray) -> np.ndarray:
        """Scale (B,N,F,W) histories to [0,1], preserving missing values."""
        if histories.ndim != 4 or histories.shape[2] != len(self.features):
            raise ValueError("Expected histories with shape (B,N,F,W)")
        lo = self.minimum[None, None, :, None]
        span = np.maximum(self.maximum - self.minimum, 1e-6)[None, None, :, None]
        return ((histories - lo) / span).astype(np.float32)
