"""Per-turbine four-parameter logistic power curves, fitted on training rows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def logistic_numpy(wind: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Parameters are lower kW, upper kW, midpoint m/s and slope 1/(m/s)."""
    wind = np.asarray(wind)
    params = np.asarray(params)
    if params.ndim == 2 and wind.ndim >= 2:
        n = params.shape[0]
        if wind.shape[-2] == n:
            params = params.reshape((1,) * (wind.ndim - 2) + (n, 1, 4))
        elif wind.shape[-1] == n:
            params = params.reshape((1,) * (wind.ndim - 1) + (n, 4))
    lower, upper, midpoint, slope = np.moveaxis(params, -1, 0)
    z = np.clip((wind - midpoint) * slope, -50, 50)
    return lower + (upper - lower) / (1.0 + np.exp(-z))


def logistic_torch(wind, params):
    """Differentiable counterpart; import torch only when called."""
    import torch
    if params.ndim == 2 and wind.ndim >= 2:
        n = params.shape[0]
        if wind.shape[-2] == n:
            params = params.reshape((1,) * (wind.ndim - 2) + (n, 1, 4))
        elif wind.shape[-1] == n:
            params = params.reshape((1,) * (wind.ndim - 1) + (n, 4))
    lower, upper, midpoint, slope = params.unbind(dim=-1)
    return lower + (upper - lower) * torch.sigmoid((wind - midpoint) * slope)


def _fit_params(wind: np.ndarray, power: np.ndarray, rated: float):
    # Bin first so the small two-dimensional grid fit stays inexpensive.
    bins = np.clip((wind * 2).astype(np.int32), 0, 60)
    count = np.bincount(bins, minlength=61)
    total = np.bincount(bins, weights=power, minlength=61)
    used = count > 0
    speeds = np.arange(61)[used] / 2.0 + 0.25
    means = total[used] / count[used]
    weights = count[used].astype(float)
    best = (float("inf"), None)
    for midpoint in np.arange(5.0, 16.01, 0.5):
        for slope in np.arange(0.3, 2.51, 0.2):
            sigmoid = 1 / (1 + np.exp(-np.clip((speeds - midpoint) * slope, -50, 50)))
            design = np.column_stack((1 - sigmoid, sigmoid))
            weighted = design * np.sqrt(weights[:, None])
            values = means * np.sqrt(weights)
            lower, upper = np.linalg.lstsq(weighted, values, rcond=None)[0]
            lower = float(np.clip(lower, 0, rated))
            upper = float(np.clip(upper, lower, rated))
            estimated = lower * (1 - sigmoid) + upper * sigmoid
            error = float(np.sum(weights * (means - estimated) ** 2))
            if error < best[0]:
                best = (error, [lower, upper, float(midpoint), float(slope)])
    return best[1]


def fit_power_curves(train: pd.DataFrame, config, meta_path: Optional[Path] = None
                     ) -> Dict[str, dict]:
    """Fit only valid training rows; optionally add R² and parameters to meta JSON."""
    required = {"TurbID", "Wspd", "Patv"} | set(config.exclude_flags_main)
    if not required.issubset(train.columns):
        raise ValueError("Power-curve fit needs wind, power, turbine and validity flags")
    good = np.isfinite(train["Wspd"].to_numpy(dtype=float)) & np.isfinite(
        train["Patv"].to_numpy(dtype=float))
    for flag in config.exclude_flags_main:
        good &= ~train[flag].to_numpy(dtype=bool)
    usable = train.loc[good]
    if usable.empty:
        raise ValueError("No valid training rows for power-curve fitting")
    global_params = _fit_params(usable["Wspd"].to_numpy(dtype=np.float64),
                                usable["Patv"].to_numpy(dtype=np.float64),
                                config.rated_power_kw)
    grouped = usable.groupby("TurbID")
    result = {}
    for tid in np.sort(train["TurbID"].unique()):
        frame = grouped.get_group(tid) if tid in grouped.groups else usable.iloc[:0]
        wind = frame["Wspd"].to_numpy(dtype=np.float64)
        power = frame["Patv"].to_numpy(dtype=np.float64)
        if len(wind) < 20:
            result[str(tid)] = {"params": global_params, "r2": None,
                                "n": int(len(power)), "source": "global_fallback"}
            continue
        params = _fit_params(wind, power, config.rated_power_kw)
        predicted = logistic_numpy(wind, np.asarray(params))
        residual = float(np.sum((power - predicted) ** 2))
        total_variation = float(np.sum((power - power.mean()) ** 2))
        r2 = 1 - residual / total_variation if total_variation > 0 else None
        result[str(tid)] = {"params": params, "r2": r2,
                            "n": int(len(power)), "source": "turbine"}
    if meta_path is not None:
        meta_path = Path(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["power_curves"] = result
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
