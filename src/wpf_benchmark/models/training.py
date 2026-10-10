"""Shared chronological windows for train/validation only."""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from ..data.windows import to_feature_cube
from ..data.masks import target_valid, target_values


def training_cube(frame: pd.DataFrame, features: tuple, config, scaler, weighted=False):
    """Return normalized features and a main-table target mask on an aligned grid."""
    tids = np.sort(frame["TurbID"].unique())
    clean = frame.copy()
    good = target_valid(clean, config.target_mask, config.exclude_flags_main)
    clean["_valid_target"] = good.astype(np.float32)
    use_weights = weighted and config.target_mask == "m2" and config.m2_extra_target_weight != 1
    if weighted and config.target_mask == "m2":
        common = target_valid(clean, "m1", config.exclude_flags_main)
        if np.any(common & ~good) or not np.array_equal(
                target_values(clean, "m1")[common], target_values(clean, "m2")[common]):
            raise ValueError("Weighted M2 study requires M1 to be a subset of M2 with identical common targets")
        if use_weights:
            clean.loc[good & ~common, "_valid_target"] = config.m2_extra_target_weight
    clean["_target_power"] = target_values(clean, config.target_mask)
    times, grid = to_feature_cube(clean, features + ("_valid_target", "_target_power"), tids)
    data = grid[:, :, :-2]
    valid = np.nan_to_num(grid[:, :, -2], nan=0.0) if use_weights else grid[:, :, -2] > 0.5
    lo = scaler.minimum[None, None, :]
    span = np.maximum(scaler.maximum - scaler.minimum, 1e-6)[None, None, :]
    data = (data - lo) / span
    power_index = features.index("Patv")
    target = ((grid[:, :, -1] - scaler.minimum[power_index]) /
              max(scaler.maximum[power_index] - scaler.minimum[power_index], 1e-6))
    return times, tids, data.astype(np.float32), valid, target.astype(np.float32), power_index


def issue_indices(length: int, window: int, horizon: int,
                  stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("stride must be positive")
    return np.arange(window - 1, length - horizon, stride, dtype=np.int64)


def window_batches(data: np.ndarray, valid: np.ndarray, target: np.ndarray,
                   issues: np.ndarray, window: int, horizon: int,
                   batch_size: int, shuffle: bool = False
                   ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield normalized (B,N,F,W), target (B,N,H), validity (B,N,H)."""
    if shuffle:
        issues = np.random.permutation(issues)
    offsets = np.arange(window, dtype=np.int64) - window + 1
    future = np.arange(1, horizon + 1, dtype=np.int64)
    for first in range(0, len(issues), batch_size):
        selected = issues[first:first + batch_size]
        x = data[selected[:, None] + offsets[None, :]].transpose(0, 2, 3, 1)
        y = target[selected[:, None] + future[None, :]].transpose(0, 2, 1)
        mask = valid[selected[:, None] + future[None, :]].transpose(0, 2, 1)
        mask &= np.isfinite(y)
        yield np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(y), mask
