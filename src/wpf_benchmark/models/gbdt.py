"""Shared gradient-boosted regressors, one for each forecast step."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register_model
from .training import issue_indices, training_cube, window_batches


@register_model("gbdt")
class GBDTForecaster(BaseForecaster):
    features = ("Patv", "Wspd", "Wsin", "Wcos")
    needs_phases = True

    def __init__(self, stride: int = 6, n_estimators: int = 100,
                 learning_rate: float = 0.05, max_depth: int = 6,
                 num_leaves: int = 31, n_jobs: int = 4):
        self.stride = int(stride)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.num_leaves = int(num_leaves)
        self.n_jobs = int(n_jobs)
        if min(self.stride, self.n_estimators, self.max_depth,
               self.num_leaves, self.n_jobs) <= 0 or self.learning_rate <= 0:
            raise ValueError("GBDT hyperparameters must be positive")

    @property
    def model_config(self) -> dict:
        config = {key: getattr(self, key) for key in (
            "stride", "n_estimators", "learning_rate", "max_depth",
            "num_leaves", "n_jobs")}
        config["backend"] = getattr(self, "backend", "auto")
        return config

    def _backend(self):
        try:
            import lightgbm
            self.backend = "lightgbm"
            return lightgbm
        except ImportError:
            try:
                import xgboost
                self.backend = "xgboost"
                return xgboost
            except ImportError as exc:
                raise ImportError("gbdt needs lightgbm or xgboost; install wpf-benchmark[gbdt]") from exc

    def _features(self, histories: np.ndarray, phases: np.ndarray) -> np.ndarray:
        # Input is (B,N,F,W), in physical units for this model.
        x = np.nan_to_num(histories, nan=0.0, posinf=0.0, neginf=0.0)
        last = x[:, :, :, -1]
        mean = x.mean(axis=-1)
        std = x.std(axis=-1)
        minimum = x.min(axis=-1)
        maximum = x.max(axis=-1)
        positions = np.arange(x.shape[-1], dtype=np.float32)
        centered = positions - positions.mean()
        slope = np.sum(x * centered, axis=-1) / np.sum(centered ** 2)
        b, n = x.shape[:2]
        phase_col = np.broadcast_to(phases[:, None, None], (b, n, 1)) / self.config.steps_per_day
        tid_col = np.broadcast_to(self.turbine_ids[None, :, None], (b, n, 1))
        return np.concatenate((last, mean, std, minimum, maximum, slope,
                               phase_col, tid_col), axis=-1).reshape(b * n, -1).astype(np.float32)

    def fit(self, train: pd.DataFrame, valid: Optional[pd.DataFrame] = None) -> None:
        backend = self._backend()
        times, self.turbine_ids, cube, valid_target, target, _ = training_cube(
            train, self.features, self.config, self.scaler)
        issues = issue_indices(len(times), self.config.input_window,
                               self.config.horizon, self.stride)
        if not len(issues):
            raise ValueError("Training split is too short for GBDT windows")
        x_parts, y_parts, mask_parts = [], [], []
        power_lo = self.scaler.minimum[0]
        power_span = max(self.scaler.maximum[0] - power_lo, 1e-6)
        # Restore physical units for the engineered features and kW targets.
        for offset in range(0, len(issues), 64):
            selected = issues[offset:offset + 64]
            x, y, mask = next(window_batches(
                cube, valid_target, target, selected, self.config.input_window,
                self.config.horizon, len(selected)))
            x = x * np.maximum(self.scaler.maximum - self.scaler.minimum, 1e-6)[None, None, :, None]
            x += self.scaler.minimum[None, None, :, None]
            ns = np.asarray(times[selected], dtype="timedelta64[ns]").astype(np.int64)
            phases = ((ns // 600_000_000_000) % self.config.steps_per_day).astype(np.float32)
            x_parts.append(self._features(x, phases))
            y_parts.append((y * power_span + power_lo).reshape(-1, self.config.horizon))
            mask_parts.append(mask.reshape(-1, self.config.horizon))
        X = np.concatenate(x_parts)
        Y = np.concatenate(y_parts)
        good = np.concatenate(mask_parts)
        self.estimators = []
        self.n_params = 0
        for h in range(self.config.horizon):
            if not good[:, h].any():
                raise ValueError("No valid GBDT target for horizon {}".format(h + 1))
            if self.backend == "lightgbm":
                dataset = backend.Dataset(X[good[:, h]], label=Y[good[:, h], h])
                model = backend.train({"objective": "regression",
                                       "learning_rate": self.learning_rate,
                                       "max_depth": self.max_depth,
                                       "num_leaves": self.num_leaves,
                                       "num_threads": self.n_jobs,
                                       "verbosity": -1}, dataset,
                                      num_boost_round=self.n_estimators)
            else:
                dataset = backend.DMatrix(X[good[:, h]], label=Y[good[:, h], h])
                model = backend.train({"objective": "reg:squarederror",
                                       "eta": self.learning_rate,
                                       "max_depth": self.max_depth,
                                       "nthread": self.n_jobs,
                                       "tree_method": "hist"}, dataset,
                                      num_boost_round=self.n_estimators)
            self.estimators.append(model)
            if self.backend == "lightgbm":
                self.n_params += sum(tree["num_leaves"] for tree in
                                     model.dump_model()["tree_info"])
            else:
                self.n_params += sum(line.count("leaf=") for line in
                                     model.get_dump())

    def predict(self, history: np.ndarray) -> np.ndarray:
        raise ValueError("GBDT requires issue-time phases; use predict_batch")

    def predict_batch(self, histories: np.ndarray,
                      phases: Optional[np.ndarray] = None) -> np.ndarray:
        if phases is None or len(phases) != len(histories):
            raise ValueError("GBDT requires one phase per issue time")
        X = self._features(histories, np.asarray(phases, dtype=np.float32))
        if self.backend == "xgboost":
            import xgboost
            X = xgboost.DMatrix(X)
        y = np.column_stack([model.predict(X) for model in self.estimators])
        return np.clip(y.reshape(len(histories), len(self.turbine_ids), -1),
                       0, self.config.rated_power_kw).astype(np.float32)
