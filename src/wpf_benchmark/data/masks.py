"""Target definitions derived from the original SDWPF observations."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


TARGET_MASKS = ("m1", "m2")
OFFICIAL_COLUMNS = ("o_pitch", "o_zero_windy", "o_dir_abnormal",
                    "o_bad", "raw_valid", "Patv_obs")


def required_columns(mask: str, base_flags: Sequence[str]) -> tuple:
    if mask == "m1":
        return tuple(base_flags) + ("o_bad",)
    if mask == "m2":
        return ("raw_valid", "o_bad", "Patv_obs")
    raise ValueError("target mask must be m1 or m2")


def require_official_columns(frame: pd.DataFrame, mask: str) -> None:
    absent = [name for name in required_columns(mask, ()) if name not in frame]
    if absent:
        raise ValueError("Re-run preprocess to create official-rule columns: " +
                         ", ".join(absent))


def base_valid(frame: pd.DataFrame, base_flags: Sequence[str]) -> np.ndarray:
    """The historical clean-target gate, retained only for M1 and legacy checks."""
    absent = [name for name in base_flags if name not in frame]
    if absent:
        raise ValueError("Configured exclusion flag absent from data: " +
                         ", ".join(absent))
    good = np.isfinite(frame["Patv"].to_numpy(dtype=float))
    for flag in base_flags:
        good &= ~frame[flag].to_numpy(dtype=bool)
    return good


def target_valid(frame: pd.DataFrame, mask: str,
                 base_flags: Sequence[str]) -> np.ndarray:
    require_official_columns(frame, mask)
    if mask == "m1":
        return base_valid(frame, base_flags) & ~frame["o_bad"].to_numpy(dtype=bool)
    return (frame["raw_valid"].to_numpy(dtype=bool) &
            ~frame["o_bad"].to_numpy(dtype=bool) &
            np.isfinite(frame["Patv_obs"].to_numpy(dtype=float)))


def target_values(frame: pd.DataFrame, mask: str) -> np.ndarray:
    require_official_columns(frame, mask)
    return frame["Patv_obs"].to_numpy(dtype=np.float32) if mask == "m2" else (
        frame["Patv"].to_numpy(dtype=np.float32))
