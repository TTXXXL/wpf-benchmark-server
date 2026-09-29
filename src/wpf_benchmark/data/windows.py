"""Build chronological model inputs without crossing the test boundary forward."""
from __future__ import annotations

from typing import Iterator, Sequence, Tuple

import numpy as np
import pandas as pd


def to_feature_cube(df: pd.DataFrame, features: Sequence[str],
                    turbine_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted time values and an aligned (T,N,F) physical-unit cube."""
    time_codes, times = pd.factorize(df["ts"], sort=True)
    turbine_codes = pd.Index(turbine_ids).get_indexer(df["TurbID"])
    if np.any(turbine_codes < 0):
        raise ValueError("Window data contains an unknown turbine ID")
    if len(df) != len(times) * len(turbine_ids):
        raise ValueError("Window data must contain one row per time and turbine")
    if df.duplicated(["ts", "TurbID"]).any():
        raise ValueError("Window data contains duplicate time/turbine pairs")
    cube = np.full((len(times), len(turbine_ids), len(features)),
                   np.nan, dtype=np.float32)
    for index, name in enumerate(features):
        cube[time_codes, turbine_codes, index] = df[name].to_numpy(dtype=np.float32)
    return np.asarray(times), cube


def iter_history_batches(cube: np.ndarray, prefix_steps: int, t_eff: int,
                         input_window: int, batch_size: int) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (first issue index, histories[B,N,F,W]) in time order."""
    if batch_size <= 0 or input_window <= 0:
        raise ValueError("batch_size and input_window must be positive")
    if prefix_steps < input_window - 1:
        raise ValueError("Not enough prior observations for the first test issue")
    if prefix_steps + t_eff > cube.shape[0]:
        raise ValueError("Issue times exceed the available time grid")
    for first in range(0, t_eff, batch_size):
        stop = min(first + batch_size, t_eff)
        histories = np.stack([
            cube[prefix_steps + t - input_window + 1:prefix_steps + t + 1]
            .transpose(1, 2, 0)
            for t in range(first, stop)], axis=0)
        yield first, histories
