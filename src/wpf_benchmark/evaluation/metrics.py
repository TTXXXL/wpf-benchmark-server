"""Numerical primitives for deterministic evaluation."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .config import ProtocolConfig


def _mad_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return 5.0
    median = np.median(values)
    return max(float(1.4826 * np.median(np.abs(values - median))), 5.0)


def _fit_power_curves(wspd: np.ndarray, patv: np.ndarray, excl: np.ndarray,
                      cfg: ProtocolConfig):
    """Fit turbine curves and residual scales using training observations only."""
    edges = np.arange(0.0, 26.0 + cfg.curve_bin_width, cfg.curve_bin_width)
    centers = (edges[:-1] + edges[1:]) / 2.0
    n_bins = len(centers)
    curves = np.full((wspd.shape[1], n_bins), np.nan, dtype=np.float32)
    sigmas = np.full_like(curves, np.nan)
    for j in range(wspd.shape[1]):
        v, p = wspd[:, j], patv[:, j]
        ok = (np.isfinite(v) & np.isfinite(p) & ~excl[:, j] &
              (p > 0) & (v >= 0.25) & (v < edges[-1]))
        if not np.any(ok):
            continue
        speed, power = v[ok], p[ok]
        bins = np.clip(np.searchsorted(edges, speed, side="right") - 1, 0, n_bins - 1)
        medians = np.full(n_bins, np.nan, dtype=np.float32)
        for b in range(n_bins):
            selected = bins == b
            if int(selected.sum()) >= cfg.curve_min_samples:
                medians[b] = np.median(power[selected])
        supported = np.isfinite(medians)
        if not np.any(supported):
            continue
        if supported.sum() == 1:
            medians[:] = medians[supported][0]
        else:
            medians = np.interp(centers, centers[supported], medians[supported]).astype(np.float32)
        curves[j] = medians
        residual = power - medians[bins]
        bin_scale = np.full(n_bins, np.nan, dtype=np.float32)
        for b in range(n_bins):
            selected = bins == b
            if int(selected.sum()) >= 100:
                bin_scale[b] = _mad_sigma(residual[selected])
        # A turbine-wide typical bin dispersion is the fallback for sparse bins.
        supported_sigma = np.isfinite(bin_scale)
        whole_sigma = (float(np.median(bin_scale[supported_sigma]))
                       if np.any(supported_sigma) else _mad_sigma(residual))
        sigmas[j] = np.where(supported_sigma, bin_scale, whole_sigma)
    return centers.astype(np.float32), curves, sigmas


def _metrics(err: np.ndarray, cfg: ProtocolConfig) -> Dict[str, float]:
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"MAE_kW": float("nan"), "RMSE_kW": float("nan"),
                "NMAE_pct": float("nan")}
    mae = float(np.abs(err).mean(dtype=np.float64))
    rmse = float(np.sqrt(np.square(err).mean(dtype=np.float64)))
    return {"MAE_kW": mae, "RMSE_kW": rmse,
            "NMAE_pct": 100.0 * mae / cfg.rated_power_kw}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")
