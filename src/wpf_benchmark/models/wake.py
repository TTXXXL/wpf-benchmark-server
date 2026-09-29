"""Directional wake geometry and a train-only static graph prior."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.io import load_clean, load_meta
from ..paths import ProjectPaths


@dataclass(frozen=True)
class WakeConvention:
    sign: int
    x_axis: str
    direction: str

    @property
    def name(self) -> str:
        return "{}-{}-{}".format("plus" if self.sign == 1 else "minus",
                                 self.x_axis, self.direction)

    @classmethod
    def parse(cls, name: str) -> "WakeConvention":
        parts = name.split("-")
        if len(parts) != 3 or parts[0] not in ("plus", "minus") or (
                parts[1] not in ("north", "east")) or (
                parts[2] not in ("from", "toward")):
            raise ValueError("Unknown wind convention: " + name)
        return cls(1 if parts[0] == "plus" else -1, parts[1], parts[2])


CONVENTIONS = tuple(WakeConvention(sign, axis, direction)
                    for sign in (1, -1)
                    for axis in ("north", "east")
                    for direction in ("from", "toward"))


@dataclass(frozen=True)
class WakeConfig:
    sectors: int = 8
    sector_half_angle: float = 30.0
    d_min_km: float = 0.3
    d_max_km: float = 15.0

    def __post_init__(self):
        if self.sectors != 8:
            raise ValueError("The preregistered wake experiment uses eight sectors")
        if not 0 < self.sector_half_angle < 90 or (
                self.d_min_km <= 0 or self.d_max_km <= self.d_min_km):
            raise ValueError("Invalid wake geometry")


def circular_mean_degrees(degrees: np.ndarray, axis: int = 1) -> np.ndarray:
    """Return a circular mean; all-missing slices remain NaN."""
    rad = np.deg2rad(degrees)
    with np.errstate(invalid="ignore"):
        count = np.isfinite(rad).sum(axis=axis)
        sine = np.nansum(np.sin(rad), axis=axis)
        cosine = np.nansum(np.cos(rad), axis=axis)
    mean = np.rad2deg(np.arctan2(sine, cosine)) % 360.0
    return np.where(count > 0, mean, np.nan)


def wind_from_degrees(ndir: np.ndarray, wdir: np.ndarray,
                      convention: WakeConvention) -> np.ndarray:
    """Ndir is north-referenced; Wdir sign and from/toward remain calibrated."""
    result = (ndir + convention.sign * wdir) % 360.0
    if convention.direction == "toward":
        result = (result + 180.0) % 360.0
    return result


def direction_bins(wind_from: np.ndarray, sectors: int = 8) -> np.ndarray:
    width = 360.0 / sectors
    result = np.full(wind_from.shape, -1, dtype=np.int16)
    good = np.isfinite(wind_from)
    result[good] = np.floor((wind_from[good] + width / 2) / width).astype(
        np.int16) % sectors
    return result


def wake_matrices(xy_km: np.ndarray, convention: WakeConvention,
                  config: WakeConfig = WakeConfig(),
                  extra_rotation: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """A[target, source] is nonzero only when source is upwind of target."""
    xy = np.asarray(xy_km, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Turbine locations must be a finite N x 2 array")
    displacement = xy[:, None, :] - xy[None, :, :]  # target minus source
    distance = np.linalg.norm(displacement, axis=-1)
    possible = (distance >= config.d_min_km) & (distance <= config.d_max_km)
    possible &= ~np.eye(len(xy), dtype=bool)
    centers = np.arange(config.sectors, dtype=np.float32) * 360.0 / config.sectors
    output = np.zeros((config.sectors, len(xy), len(xy)), dtype=np.float32)
    for k, wind_from in enumerate(centers):
        # A direction marked "toward" makes the raw heading the downwind
        # direction; a meteorological "from" heading is reversed by 180°.
        downwind = (wind_from + 180.0 + extra_rotation) % 360.0
        radians = np.deg2rad(downwind)
        vector = (np.array([np.cos(radians), np.sin(radians)])
                  if convention.x_axis == "north" else
                  np.array([np.sin(radians), np.cos(radians)]))
        projection = np.einsum("ijk,k->ij", displacement, vector)
        within = projection >= distance * np.cos(np.deg2rad(config.sector_half_angle))
        adjacency = (possible & within).astype(np.float32)
        rows = adjacency.sum(axis=1, keepdims=True)
        output[k] = adjacency / np.maximum(rows, 1.0)
    return output, centers


def static_wake_prior(matrices: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float32)
    frequencies = np.asarray(frequencies, dtype=np.float32)
    if matrices.ndim != 3 or frequencies.shape != (matrices.shape[0],):
        raise ValueError("Wake matrices and directional frequencies do not align")
    if not np.isclose(frequencies.sum(), 1.0, atol=1e-6):
        raise ValueError("Directional frequencies must sum to one")
    prior = np.einsum("k,kij->ij", frequencies, matrices)
    rows = prior.sum(axis=1, keepdims=True)
    return (prior / np.where(rows > 0, rows, 1.0)).astype(np.float32)


def load_locations(paths: ProjectPaths) -> Tuple[np.ndarray, np.ndarray]:
    source = paths.raw / "sdwpf_baidukddcup2022_turb_location.csv"
    frame = pd.read_csv(source).sort_values("TurbID")
    turbine_ids = frame["TurbID"].to_numpy()
    if len(np.unique(turbine_ids)) != len(turbine_ids):
        raise ValueError("Location file has duplicate turbine IDs")
    return turbine_ids, frame[["x", "y"]].to_numpy(dtype=np.float32) / 1000.0


def prepare_wake_prior(paths: ProjectPaths,
                       convention: Optional[WakeConvention] = None,
                       config: WakeConfig = WakeConfig(),
                       train_days: int = 196) -> Path:
    """Persist directional graphs and a row-normalized train-frequency prior."""
    if convention is None:
        report = paths.reports / "wake_calibration.json"
        if not report.is_file():
            raise FileNotFoundError("Run calibrate-wake before prepare-wake-prior")
        finding = json.loads(report.read_text(encoding="utf-8"))
        if finding.get("decision") != "significant":
            raise ValueError("Wake prior is not approved for the main method")
        convention = WakeConvention.parse(finding["selected_convention"])
    turbine_ids, xy_km = load_locations(paths)
    matrices, directions = wake_matrices(xy_km, convention, config)
    frame = load_clean(paths, columns=["ts", "TurbID", "Day", "Ndir", "Wdir",
                                       "m_missing", "m_imputed"],
                       filters=[("Day", "<=", train_days)])
    frame = frame.sort_values(["ts", "TurbID"])
    observed_ids = np.sort(frame["TurbID"].unique())
    if not np.array_equal(observed_ids, turbine_ids):
        raise ValueError("Location and training turbine orders differ")
    n = len(turbine_ids)
    if len(frame) % n:
        raise ValueError("Training frame is not an aligned turbine grid")
    valid = (~frame["m_missing"].to_numpy(dtype=bool) &
             ~frame["m_imputed"].to_numpy(dtype=bool))
    north = frame["Ndir"].to_numpy(dtype=np.float32)
    relative = frame["Wdir"].to_numpy(dtype=np.float32)
    angles = wind_from_degrees(north, relative, convention)
    angles[~valid] = np.nan
    global_from = circular_mean_degrees(angles.reshape(-1, n))
    bins = direction_bins(global_from, config.sectors)
    counts = np.bincount(bins[bins >= 0], minlength=config.sectors)
    if counts.sum() == 0:
        raise ValueError("No observed training wind directions")
    frequencies = counts.astype(np.float32) / counts.sum()
    prior = static_wake_prior(matrices, frequencies)
    paths.processed.mkdir(parents=True, exist_ok=True)
    output = paths.processed / "wake_prior.npz"
    np.savez_compressed(output, A_wake=matrices, A_wake_prior=prior,
                        wind_dirs=directions, w_k=frequencies,
                        turbine_ids=turbine_ids,
                        convention=np.asarray(convention.name))
    meta_path = paths.processed / "sdwpf_meta.json"
    meta = load_meta(paths)
    meta["turbine_locations"] = {str(tid): xy_km[i].tolist()
                                  for i, tid in enumerate(turbine_ids)}
    meta["wake_prior"] = {"convention": convention.name,
                          "train_days": train_days,
                          "config": config.__dict__}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return output
