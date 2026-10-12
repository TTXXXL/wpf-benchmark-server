"""Bounded, dense inference exports without scoring later observations."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..data.io import load_clean
from ..data.masks import target_valid
from ..data.windows import iter_history_batches, to_feature_cube

STEP = 600_000_000_000
DAY = 86_400_000_000_000
HISTORY_COLUMNS = ("Patv", "Wspd", "Pab1", "Pab2", "Pab3", "m_imputed")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def block_frame(paths, cfg, first_day, last_day, features):
    if not 1 <= first_day <= last_day or first_day <= cfg.train_days + cfg.val_days:
        raise ValueError("Forward block must follow training and early stopping")
    columns = list(dict.fromkeys(("ts", "Day", "TurbID", "Patv", "Patv_obs", "o_bad",
        "raw_valid") + tuple(features) + HISTORY_COLUMNS + tuple(cfg.exclude_flags_main)))
    lookback = (cfg.input_window - 1 + cfg.steps_per_day - 1) // cfg.steps_per_day + 1
    frame = load_clean(paths, columns=columns, filters=[
        ("Day", ">=", max(1, first_day - lookback)), ("Day", "<=", last_day)])
    tids = np.sort(frame["TurbID"].unique())
    all_times = np.sort(frame["ts"].unique())
    start = np.searchsorted(all_times, np.timedelta64((first_day - 1) * DAY, "ns"))
    prefix = cfg.input_window - 1
    if start < prefix:
        raise ValueError("Insufficient causal history before forward block")
    times = all_times[start - prefix:]
    ns = times.astype("timedelta64[ns]").astype(np.int64)
    if not np.all(np.diff(ns) == STEP) or not len(tids):
        raise ValueError("Forward history must be a complete 10-minute grid")
    frame = frame.loc[frame["ts"].isin(times)].sort_values(["ts", "TurbID"])
    if len(frame) != len(times) * len(tids) or frame.duplicated(["ts", "TurbID"]).any():
        raise ValueError("Missing or duplicate forward turbine rows")
    count = len(times) - prefix - cfg.horizon
    if count <= 0:
        raise ValueError("Forward block is shorter than horizon")
    issue = ns[prefix:prefix + count]
    target = issue[:, None] + STEP * np.arange(1, cfg.horizon + 1)[None, :]
    if np.any(issue < (first_day - 1) * DAY) or np.any(target >= last_day * DAY):
        raise ValueError("Forward target crossed a phase boundary")
    return frame, times, tids, prefix, issue, target


def episode_starts(paths, issue, tids):
    # Episode IDs use the complete causal prefix, not the target-valid subset.
    last_day = int(issue[-1] // DAY + 1)
    frame = load_clean(paths, columns=["ts", "TurbID", "Patv"],
                       filters=[("Day", "<=", last_day)])
    power = frame.pivot(index="ts", columns="TurbID", values="Patv").reindex(columns=tids)
    times = power.index.to_numpy().astype("timedelta64[ns]").astype(np.int64)
    if not np.all(np.diff(times) == STEP):
        raise ValueError("Episode history grid is discontinuous")
    low = power.ffill(limit=143).fillna(0).to_numpy() < 10
    start = low & ~np.vstack([np.zeros((1, len(tids)), bool), low[:-1]])
    starts = np.maximum.accumulate(np.where(start, times[:, None], -1), axis=0)
    index = np.searchsorted(times, issue)
    if not np.array_equal(times[index], issue):
        raise ValueError("Episode and issue grids differ")
    return np.where(low[index], starts[index], -1)


def export_inputs(paths, cfg, days, output, features=("Wspd", "Wsin", "Wcos", "Patv")):
    frame, times, tids, prefix, issue, target = block_frame(paths, cfg, *days, features)
    count, turbines = len(issue), len(tids)
    def target_stack(values):
        wide = np.asarray(values).reshape(len(times), turbines)
        return np.stack([wide[prefix + h:prefix + h + count]
                         for h in range(1, cfg.horizon + 1)], axis=2)
    arrays = dict(times=times.astype("timedelta64[ns]").astype(np.int64),
                  issue_times=issue, target_times=target, turbine_ids=tids,
                  days=np.asarray(days), model_feature_names=np.asarray(features),
                  event_start=episode_starts(paths, issue, tids))
    for name in HISTORY_COLUMNS:
        arrays[name] = frame[name].to_numpy().reshape(len(times), turbines)
    _, arrays["model_history"] = to_feature_cube(frame, features, tids)
    for population in ("m1", "m2"):
        arrays["valid_" + population] = target_stack(target_valid(frame, population, cfg.exclude_flags_main))
        arrays["truth_" + population] = target_stack(frame[
            "Patv_obs" if population == "m2" else "Patv"].to_numpy(dtype=np.float32))
    atomic_npz(output, **arrays)
    return sha256(output)


def export_predictions(model, paths, days, output, checkpoint_sha256, inputs_sha256,
                       batch_size=32):
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive")
    frame, times, tids, prefix, issue, target = block_frame(
        paths, model.config, *days, model.features)
    if not np.array_equal(tids, model.turbine_ids):
        raise ValueError("Forward turbine order differs from checkpoint")
    _, cube = to_feature_cube(frame, model.features, tids)
    forecasts = np.empty((len(issue), len(tids), model.config.horizon), np.float32)
    batches = iter_history_batches(cube, prefix, len(issue), model.config.input_window, batch_size)
    torch = model.torch
    cuda = model.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(model.device)
    durations = []
    for first, histories in batches:
        if model.history_scale == "normalized":
            histories = model.scaler.transform_histories(histories)
        if first == 0:
            model.predict_batch(histories)  # One fixed warm-up; no labels are used.
        if cuda:
            torch.cuda.synchronize(model.device)
        start = perf_counter()
        prediction = np.asarray(model.predict_batch(histories), dtype=np.float32)
        if cuda:
            torch.cuda.synchronize(model.device)
        durations.append(perf_counter() - start)
        if prediction.shape != (len(histories), len(tids), model.config.horizon) or not np.isfinite(prediction).all():
            raise ValueError("Invalid forward prediction shape or values")
        forecasts[first:first + len(histories)] = prediction
    atomic_npz(output, forecasts=forecasts, issue_times=issue, target_times=target,
               turbine_ids=tids)
    try:
        import psutil
        import os
        rss = psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        rss = None
    info = dict(schema=1, days=list(days), checkpoint_sha256=checkpoint_sha256,
        inputs_sha256=inputs_sha256, predictions_sha256=sha256(output),
        protocol=asdict(model.config), model_config=model.model_config,
        checkpoint_metadata=model.checkpoint_metadata,
        device=str(model.device), device_name=(torch.cuda.get_device_name(model.device) if cuda else "CPU"),
        predict_seconds=sum(durations), batch_size=batch_size, warmup_batches=1,
        n_issue_windows=len(issue), n_turbines=len(tids),
        batch_seconds_p50=float(np.quantile(durations, .5)),
        batch_seconds_p95=float(np.quantile(durations, .95)),
        cpu_rss_snapshot_bytes=rss,
        cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(model.device)) if cuda else None,
        model_state_bytes=sum(v.numel() * v.element_size() for v in model.network.state_dict().values()))
    atomic_json(Path(output).with_suffix(".json"), info)
    return info
