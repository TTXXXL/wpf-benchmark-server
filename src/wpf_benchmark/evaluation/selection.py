"""Frozen dense development grids and equal-horizon checkpoint selection.

Grid membership follows the previous validation_grid: issues start at the first
validation timestamp and end when all horizons are available. Subsets use target
days, so an issue before a subset boundary can contribute targets after it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.masks import target_valid, target_values
from ..data.windows import to_feature_cube


GRID_KEYS = ("issue_times", "target_times", "turbine_ids", "valid_m1",
             "valid_m2", "truth_m1", "truth_m2", "issue_power")
TOLERANCE_KW = 1e-8


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def grid_digest(grid):
    digest = hashlib.sha256()
    for key in GRID_KEYS:
        value = np.ascontiguousarray(grid[key])
        digest.update(key.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def audit_targets(frame, cfg):
    m1 = target_valid(frame, "m1", cfg.exclude_flags_main)
    m2 = target_valid(frame, "m2", cfg.exclude_flags_main)
    y1, y2 = target_values(frame, "m1"), target_values(frame, "m2")
    if np.any(m1 & ~m2) or not np.array_equal(y1[m1], y2[m1]):
        raise ValueError("M1 must be a subset of M2 with identical common targets")
    if np.any(y2[m2] < 0):
        raise ValueError("Patv_obs must use the frozen nonnegative observation convention")
    return dict(rows=len(frame), m1=int(m1.sum()), m2=int(m2.sum()),
                m2_only=int((m2 & ~m1).sum()))


def build_grid(frame, cfg, features):
    """Explicitly align keys; never assume Parquet row order."""
    audit_targets(frame, cfg)
    if frame.Day.max() > cfg.train_days + cfg.val_days:
        raise ValueError("Development grid must not contain test days")
    tids = np.sort(frame.TurbID.unique())
    fields = list(features) + ["Day"]
    clean = frame.copy()
    for mask in ("m1", "m2"):
        clean["valid_" + mask] = target_valid(clean, mask, cfg.exclude_flags_main)
        clean["truth_" + mask] = target_values(clean, mask)
        fields += ["valid_" + mask, "truth_" + mask]
    times, cube = to_feature_cube(clean, fields, tids)
    ns = times.astype("timedelta64[ns]").astype(np.int64)
    step_ns = 600_000_000_000
    if len(ns) < 2 or not np.all(np.diff(ns) == step_ns):
        raise ValueError("Dense development history must have a regular 10-minute grid")
    days = cube[:, :, fields.index("Day")]
    if not np.all(days == days[:, :1]):
        raise ValueError("Day differs between turbines")
    # Do not derive days by timestamp arithmetic: fixture and real Day origins
    # differ. Persist the aligned Day labels along with every target timestamp.
    issues = np.flatnonzero(days[:, 0] > cfg.train_days)
    issues = issues[(issues >= cfg.input_window - 1) & (issues + cfg.horizon < len(ns))]
    if not len(issues):
        raise ValueError("No complete development forecast windows")
    future = issues[:, None] + np.arange(1, cfg.horizon + 1)
    past = issues[:, None] + np.arange(cfg.input_window) - cfg.input_window + 1
    power = cube[:, :, fields.index("Patv")]
    # Bounded ffill is exactly latest finite power in the input window.
    anchor = pd.DataFrame(np.where(np.isfinite(power), power, np.nan)).ffill(
        limit=max(cfg.input_window - 1, 1)).fillna(0).to_numpy(dtype=np.float32)[issues]
    grid = dict(issue_times=ns[issues], target_times=ns[future], turbine_ids=tids.astype(np.int64),
                issue_power=power[issues], persistence_power=anchor,
                target_days=days[future, 0].astype(np.int64))
    for mask in ("m1", "m2"):
        grid["valid_" + mask] = cube[:, :, fields.index("valid_" + mask)][future].transpose(0, 2, 1) > .5
        grid["truth_" + mask] = cube[:, :, fields.index("truth_" + mask)][future].transpose(0, 2, 1)
    # Causal group definitions; target flags are separate retrospective labels.
    history = cube[:, :, :len(features)]
    recent = issues[:, None] + np.arange(min(6, cfg.input_window)) - min(6, cfg.input_window) + 1
    grid["history_low6"] = np.all(np.isfinite(power[recent]) & (power[recent] < 10), axis=1)
    wind = history[:, :, list(features).index("Wspd")]
    grid["history_calm6"] = np.all(np.isfinite(wind[recent]) & (wind[recent] >= 0) & (wind[recent] < 1), axis=1)
    grid["S1_posthoc"] = grid["history_calm6"] & np.all(
        np.isfinite(wind[future]) & (wind[future] >= 0) & (wind[future] < 1), axis=1)
    quality = ["m_missing", "m_imputed", "m_outlier", "f_fault", "f_curtail", "f_farm"]
    for flag in quality:
        _, values = to_feature_cube(clean, [flag], tids)
        values = values[:, :, 0] > .5
        grid["target_" + flag] = values[future].transpose(0, 2, 1)
        grid["history_" + flag + "_fraction"] = values[past].mean(axis=1).astype(np.float32)
    return grid, history.astype(np.float32), issues


def day_mask(grid, day_range):
    first, last = day_range
    return ((grid["target_days"] >= first) & (grid["target_days"] <= last))[:, None, :]


def predict_grid(model, cube, issues):
    predictions = []
    for first in range(0, len(issues), model.batch):
        take = issues[first:first + model.batch]
        indices = take[:, None] + np.arange(model.config.input_window) - model.config.input_window + 1
        histories = cube[indices].transpose(0, 2, 3, 1)
        predictions.append(model.predict_batch(model.scaler.transform_histories(histories)))
    return np.concatenate(predictions)


def metric_rows(prediction, truth, anchor, valid):
    rows = []
    for h in range(truth.shape[-1]):
        mask = valid[:, :, h]
        error = prediction[:, :, h][mask].astype(np.float64) - truth[:, :, h][mask].astype(np.float64)
        persistence = anchor[mask].astype(np.float64) - truth[:, :, h][mask].astype(np.float64)
        n = int(mask.sum())
        mae = float(np.abs(error).mean()) if n else None
        pmae = float(np.abs(persistence).mean()) if n else None
        rows.append(dict(horizon=h + 1, cells=n, MAE_kW=mae,
                         RMSE_kW=float(np.sqrt(np.mean(error ** 2))) if n else None,
                         ME_kW=float(error.mean()) if n else None,
                         persistence_MAE_kW=pmae,
                         skill_pct=100 * (1 - mae / pmae) if pmae else None))
    equal = {key: float(np.mean([row[key] for row in rows])) if all(
        row[key] is not None for row in rows) else None for key in (
            "MAE_kW", "RMSE_kW", "ME_kW", "persistence_MAE_kW", "skill_pct")}
    return dict(per_horizon=rows, equal_horizon=equal, cells=sum(r["cells"] for r in rows))


def validation_scores(prediction, grid, selection_days):
    # A fixed score population is essential for epoch selection. Fail closed on
    # nonfinite predictions rather than removing different members each epoch.
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite dense validation prediction")
    finite = np.isfinite(grid["truth_m2"]) & np.isfinite(grid["issue_power"])[..., None]
    result = {}
    for mask in ("m1", "m2"):
        valid = grid["valid_" + mask] & finite & day_mask(grid, selection_days)
        score = metric_rows(prediction, grid["truth_" + mask], grid["persistence_power"], valid)
        if score["equal_horizon"]["MAE_kW"] is None:
            raise ValueError("Every selection horizon requires valid targets")
        result[mask] = score
    return result


def select_epoch(rows, mask, budget):
    if [row["epoch"] for row in rows] != list(range(1, budget + 1)):
        raise ValueError("Candidate epochs must be complete, unique and increasing")
    values = np.array([row["validation"][mask]["equal_horizon"]["MAE_kW"] for row in rows], float)
    if not np.isfinite(values).all():
        raise ValueError("Selection scores must be finite")
    # Tie relative to the GLOBAL minimum; chained pairwise tolerance comparisons
    # can otherwise choose a score farther than tolerance from the minimum.
    return int(np.flatnonzero(values <= values.min() + TOLERANCE_KW)[0]) + 1
