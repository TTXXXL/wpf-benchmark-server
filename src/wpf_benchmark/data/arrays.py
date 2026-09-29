"""Chronological split and turbine-aligned wide arrays."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..evaluation.config import ProtocolConfig


def split_by_day(df: pd.DataFrame, cfg: ProtocolConfig):
    """Return chronological train, validation and test portions, without shuffling."""
    days = np.sort(df["Day"].unique())
    stop = cfg.train_days + cfg.val_days
    if len(days) <= stop:
        raise ValueError("No test days remain after the configured split")
    train = df[df["Day"].isin(days[:cfg.train_days])]
    valid = df[df["Day"].isin(days[cfg.train_days:stop])]
    test = df[df["Day"].isin(days[stop:])]
    return train, valid, test


def to_wide(df: pd.DataFrame, cfg: ProtocolConfig) -> Dict[str, Any]:
    """Convert a complete long-table time grid to float32 (T,N) arrays."""
    tids = np.sort(df["TurbID"].unique())
    times, unique_times = pd.factorize(df["ts"], sort=True)
    turbines = pd.Index(tids).get_indexer(df["TurbID"])
    T, N = len(unique_times), len(tids)

    def widen(col: str) -> np.ndarray:
        out = np.full((T, N), np.nan, dtype=np.float32)
        out[times, turbines] = df[col].to_numpy(dtype=np.float32, copy=False)
        return out

    data: Dict[str, Any] = {"tids": tids, "times": np.asarray(unique_times), "T": T, "N": N,
                            "Patv": widen("Patv"), "Wspd": widen("Wspd")}
    for flag in cfg.exclude_flags_main:
        if flag not in df:
            raise ValueError("Configured exclusion flag absent from data: " + flag)
        mask = np.zeros((T, N), dtype=bool)
        mask[times, turbines] = df[flag].fillna(False).to_numpy(dtype=bool)
        data[flag] = mask
    return data
