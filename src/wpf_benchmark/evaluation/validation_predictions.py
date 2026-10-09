"""Dense validation forecasts without opening any post-validation records."""
from __future__ import annotations

import hashlib
import os
from time import perf_counter

import numpy as np
import pandas as pd

from ..data.io import load_clean
from ..data.schema import FLAG_COLUMNS, OFFICIAL_COLUMNS
from ..data.masks import target_valid, target_values
from ..data.windows import to_feature_cube, iter_history_batches


def latest_finite_power(histories):
    finite = np.isfinite(histories)
    ticks = np.arange(histories.shape[-1])
    last = np.where(finite, ticks, -1).max(axis=-1)
    safe = np.where(finite, histories, 0)
    anchor = np.take_along_axis(safe, np.maximum(last, 0)[..., None], axis=-1)[..., 0]
    return np.where(last >= 0, anchor, 0).astype(np.float32)


def metrics(prediction, truth, valid):
    n = int(valid.sum())
    if not n:
        return {"n": 0, "MAE_kW": None, "RMSE_kW": None, "ME_kW": None}
    error = prediction[valid].astype(np.float64) - truth[valid].astype(np.float64)
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite scored validation values")
    return {"n": n, "MAE_kW": float(np.abs(error).mean()),
            "RMSE_kW": float(np.sqrt(np.mean(error ** 2))), "ME_kW": float(error.mean())}


def grouped_metrics(arrays):
    truth, valid = arrays["truth"], arrays["valid"]
    p, w = arrays["history_power6"], arrays["history_wind6"]
    anchor = arrays["last_power"]
    groups = {"overall": valid,
              "H_calm": valid & (np.isfinite(w).all(-1) & ((w >= 0) & (w < 1)).all(-1))[..., None],
              "last_power_lt10kW": valid & ((anchor >= 0) & (anchor < 10))[..., None],
              "low_volatility": valid & (np.isfinite(p).all(-1) & (np.std(p, axis=-1) <= 5))[..., None],
              "truth_lt10kW": valid & (truth >= 0) & (truth < 10),
              "low_to_nonlow": valid & ((anchor >= 0) & (anchor < 10))[..., None] & (truth >= 10)}
    for h in range(truth.shape[-1]):
        selected = np.zeros_like(valid)
        selected[..., h] = valid[..., h]
        groups["h{}".format(h + 1)] = selected
    result = {}
    for name, selected in groups.items():
        model = metrics(arrays["forecasts"], truth, selected)
        baseline = metrics(arrays["persistence"], truth, selected)
        delta = None if not model["n"] else model["MAE_kW"] - baseline["MAE_kW"]
        skill = None if not baseline["n"] or baseline["MAE_kW"] == 0 else -delta / baseline["MAE_kW"]
        result[name] = {"model": model, "persistence": baseline,
                        "delta_MAE_kW": delta, "skill": skill,
                        "issue_days": int(np.unique(arrays["times"][selected.any(axis=(1, 2))] //
                                                     (86400 * 10 ** 9)).size)}
    return result


def save_dense_validation(model, cfg, paths, run_id, batch_size=32):
    features = tuple(model.features)
    columns = list(dict.fromkeys(("ts", "Day", "TurbID", "Patv", "Wspd") +
                                 features + FLAG_COLUMNS + OFFICIAL_COLUMNS))
    boundary = cfg.train_days + cfg.val_days
    frame = load_clean(paths, columns=columns,
                       filters=[("Day", "<=", boundary),
                                ("Day", ">=", max(1, cfg.train_days -
                                  (cfg.input_window - 1) // cfg.steps_per_day))])
    validation = frame[frame.Day > cfg.train_days]
    valid_times = np.sort(validation.ts.unique())
    previous = np.sort(frame.loc[frame.Day <= cfg.train_days, "ts"].unique())
    prefix = cfg.input_window - 1
    if len(previous) < prefix or len(valid_times) <= cfg.horizon:
        raise ValueError("Insufficient validation history or targets")
    keep = previous[-prefix:] if prefix else previous[:0]
    combined = pd.concat([frame[frame.ts.isin(keep)], validation], ignore_index=True)
    tids = np.sort(validation.TurbID.unique())
    times, cube = to_feature_cube(combined, features, tids)
    if not np.array_equal(times[prefix:], valid_times):
        raise ValueError("Validation time grid differs")
    if not np.all(np.diff(times.astype("timedelta64[ns]").astype(np.int64)) == 600 * 10 ** 9):
        raise ValueError("Validation time grid is not continuous at ten-minute resolution")
    _, diagnostics = to_feature_cube(combined, ("Patv", "Wspd"), tids)
    selected = validation.copy()
    selected["_truth"] = target_values(selected, cfg.target_mask)
    selected["_valid"] = target_valid(selected, cfg.target_mask, cfg.exclude_flags_main)
    _, target_cube = to_feature_cube(selected, ("_truth", "_valid"), tids)
    count = len(valid_times) - cfg.horizon
    future = np.arange(count)[:, None] + np.arange(1, cfg.horizon + 1)[None, :]
    truth = target_cube[future, :, 0].transpose(0, 2, 1)
    valid = (target_cube[future, :, 1] > .5).transpose(0, 2, 1) & np.isfinite(truth)
    shape = (count, len(tids), cfg.horizon)
    forecast = np.empty(shape, np.float32)
    last = np.empty(shape[:2], np.float32)
    recent_p = np.empty(shape[:2] + (min(6, cfg.input_window),), np.float32)
    recent_w = np.empty_like(recent_p)
    begin = perf_counter()
    for first, physical in iter_history_batches(cube, prefix, count, cfg.input_window, batch_size):
        size = len(physical)
        diag = np.stack([diagnostics[prefix + t - cfg.input_window + 1:prefix + t + 1]
                         .transpose(1, 2, 0) for t in range(first, first + size)])
        anchor = latest_finite_power(diag[:, :, 0])
        last[first:first + size] = np.clip(anchor, 0, cfg.rated_power_kw)
        recent_p[first:first + size] = diag[:, :, 0, -recent_p.shape[-1]:]
        recent_w[first:first + size] = diag[:, :, 1, -recent_w.shape[-1]:]
        if model.name == "persistence":
            batch = np.repeat(last[first:first + size, :, None], cfg.horizon, axis=-1)
        else:
            histories = model.scaler.transform_histories(physical) if model.history_scale == "normalized" else physical
            if model.needs_phases:
                phases = (valid_times[first:first + size].astype("timedelta64[ns]").astype(np.int64) //
                          (600 * 10 ** 9)) % cfg.steps_per_day
                batch = np.asarray(model.predict_batch(histories, phases.astype(np.int32)), np.float32)
            else:
                batch = np.asarray(model.predict_batch(histories), np.float32)
        if batch.shape != (size, len(tids), cfg.horizon) or not np.isfinite(batch).all():
            raise ValueError("Invalid dense validation forecast")
        forecast[first:first + size] = batch
    arrays = {"forecasts": forecast, "truth": truth, "valid": valid,
              "persistence": np.repeat(last[..., None], cfg.horizon, axis=-1),
              "last_power": last, "history_power6": recent_p, "history_wind6": recent_w,
              "times": valid_times[:count].astype("timedelta64[ns]").astype(np.int64),
              "turbine_ids": tids.astype(np.int64)}
    grouped = grouped_metrics(arrays)
    destination = paths.evaluation / (run_id + "_validation_arrays.npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(str(temporary), str(destination))
    return {"path": destination.relative_to(paths.root).as_posix(),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "split": "validation", "target_mask": cfg.target_mask,
            "grid": {"issues": count, "turbines": len(tids), "horizon": cfg.horizon},
            "anchor_policy": "latest_finite_history_fallback_zero_physical_clip",
            "predict_seconds": perf_counter() - begin, "groups": grouped}
