"""Shared chronological windows for train/validation only."""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from ..data.windows import to_feature_cube


def training_cube(frame: pd.DataFrame, features: tuple, config, scaler):
    """Return normalized features and a main-table target mask on an aligned grid."""
    tids = np.sort(frame["TurbID"].unique())
    clean = frame.copy()
    good = np.isfinite(clean["Patv"].to_numpy(dtype=float))
    for flag in config.exclude_flags_main:
        good &= ~clean[flag].to_numpy(dtype=bool)
    clean["_valid_target"] = good.astype(np.float32)
    times, grid = to_feature_cube(clean, features + ("_valid_target",), tids)
    data = grid[:, :, :-1]
    valid = grid[:, :, -1] > 0.5
    lo = scaler.minimum[None, None, :]
    span = np.maximum(scaler.maximum - scaler.minimum, 1e-6)[None, None, :]
    data = (data - lo) / span
    power_index = features.index("Patv")
    return times, tids, data.astype(np.float32), valid, power_index


def issue_indices(length: int, window: int, horizon: int,
                  stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("stride must be positive")
    return np.arange(window - 1, length - horizon, stride, dtype=np.int64)


def window_batches(data: np.ndarray, valid: np.ndarray, power_index: int,
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
        y = data[selected[:, None] + future[None, :], :, power_index].transpose(0, 2, 1)
        mask = valid[selected[:, None] + future[None, :]].transpose(0, 2, 1)
        mask &= np.isfinite(y)
        yield np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(y), mask
