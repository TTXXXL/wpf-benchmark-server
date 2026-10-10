"""Contract for point-forecast models."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..data.scaling import FeatureScaler
from ..evaluation.config import ProtocolConfig


class BaseForecaster:
    """Train on chronological data and predict power in kW.

    History is (N,F,input_window) in physical units by default. Set
    history_scale to "normalized" to receive train-only min/max scaled input.
    """

    name = "base"
    features = ("Patv",)
    history_scale = "physical"
    needs_phases = False
    n_params = 0
    flops_per_sample = None
    validation_metrics = None
    additional_columns = ()

    def configure(self, config: ProtocolConfig, scaler: FeatureScaler) -> None:
        if self.history_scale not in ("physical", "normalized"):
            raise ValueError("history_scale must be physical or normalized")
        if self.name != "agcrn_lite" and (config.validation_mask is not None or
                                         config.m2_extra_target_weight != 1):
            raise ValueError("Separate validation masks and M2 target weighting currently require agcrn_lite")
        self.config = config
        self.scaler = scaler

    @property
    def model_config(self) -> dict:
        """Effective constructor settings recorded with every evaluation."""
        return {}

    def fit(self, train: pd.DataFrame, valid: Optional[pd.DataFrame] = None) -> None:
        raise NotImplementedError

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Return (N,horizon) power in kW for one issue time."""
        raise NotImplementedError

    def predict_batch(self, histories: np.ndarray,
                      phases: Optional[np.ndarray] = None) -> np.ndarray:
        """Default batch adapter; GPU models should override this method."""
        return np.stack([self.predict(history) for history in histories], axis=0)
