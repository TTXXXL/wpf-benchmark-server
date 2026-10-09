"""One model workflow: split, fit, predict, evaluate and persist."""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import random
from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .data.io import load_clean, load_meta
from .data.scaling import FeatureScaler
from .data.schema import FEATURE_COLUMNS, FLAG_COLUMNS, OFFICIAL_COLUMNS
from .data.masks import target_valid
from .data.windows import iter_history_batches, to_feature_cube
from .evaluation.config import ProtocolConfig
from .evaluation.evaluator import Evaluator
from .evaluation.report import save_result
from .models.base import BaseForecaster
from .paths import ProjectPaths


def seed_all(seed: int) -> None:
    """Seed model construction and training, including CUDA when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval_forecaster(model: BaseForecaster, cfg: Optional[ProtocolConfig] = None,
                    tag: str = "", paths: Optional[ProjectPaths] = None,
                    batch_size: int = 32,
                    model_params: Optional[Dict[str, Any]] = None,
                    save_arrays: bool = True, seed: int = 0,
                    experiment: str = "standard",
                    validation_only: bool = False,
                    save_checkpoint: bool = True,
                    save_validation_arrays: bool = False) -> Dict[str, Any]:
    """Evaluate a registered model on both tables using one prediction cube."""
    cfg = cfg or ProtocolConfig()
    paths = paths or ProjectPaths.resolve()
    if save_validation_arrays and not validation_only:
        raise ValueError("save_validation_arrays requires validation_only")
    from .figures.io import append_index, config_digest, result_provenance
    data_digest, code_digest = result_provenance(paths)
    seed_all(seed)
    features = tuple(model.features)
    if not features or len(set(features)) != len(features):
        raise ValueError("Model features must be a nonempty, unique sequence")
    if any(name not in FEATURE_COLUMNS for name in features):
        raise ValueError("Model requested a feature absent from cleaned data")
    columns = list(dict.fromkeys(("ts", "Day", "TurbID", "Patv") +
                                 features + FLAG_COLUMNS + OFFICIAL_COLUMNS + tuple(model.additional_columns)))
    train = load_clean(paths, columns=columns,
                       filters=[("Day", "<=", cfg.train_days)])
    valid = load_clean(paths, columns=columns,
                       filters=[("Day", ">", cfg.train_days),
                                ("Day", "<=", cfg.train_days + cfg.val_days)])
    if train.empty or valid.empty:
        raise ValueError("Configured training or validation split is empty")
    target_counts = {"train": int(target_valid(train, cfg.target_mask, cfg.exclude_flags_main).sum()),
                     "validation": int(target_valid(valid, cfg.target_mask, cfg.exclude_flags_main).sum())}
    meta = load_meta(paths)
    use_meta = meta.get("train_days") == cfg.train_days
    scaler = (FeatureScaler.from_meta(meta, features) if use_meta
              else FeatureScaler.from_training(train, features))
    model.paths = paths
    model.configure(cfg, scaler)
    label = tag or model.name
    created = datetime.now(timezone.utc)
    stamp = created.strftime("%Y%m%d_%H%M%S_%f")
    run_id = "{}_{}".format(label, stamp)
    model.log_path = paths.reports / "train" / (run_id + ".log")
    fit_start = perf_counter()
    model.fit(train, valid)
    fit_seconds = perf_counter() - fit_start
    checkpoint = None
    if save_checkpoint and hasattr(model, "save_checkpoint"):
        from .models.checkpoint import FORMAT_VERSION
        checkpoint_path = paths.reports / "checkpoints" / (run_id + "_best.pt")
        model.save_checkpoint(checkpoint_path, metadata={
            "run_id": run_id, "seed": seed, "experiment": experiment,
            "created": created.isoformat(), "data_digest": data_digest,
            "code_digest": code_digest,
            "config_digest": config_digest(cfg, model.model_config)})
        checkpoint = {"path": checkpoint_path.relative_to(paths.root).as_posix(),
                      "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                      "format_version": FORMAT_VERSION}
    log_header = ("target_mask={} train_target_cells={} validation_target_cells={}\n".format(
        cfg.target_mask, target_counts["train"], target_counts["validation"]))
    if checkpoint is not None:
        log_header += "checkpoint={} checkpoint_sha256={}\n".format(
            checkpoint["path"], checkpoint["sha256"])
    model.log_path.parent.mkdir(parents=True, exist_ok=True)
    if model.log_path.is_file():
        model.log_path.write_text(log_header + model.log_path.read_text(encoding="utf-8"),
                                  encoding="utf-8")
    else:
        model.log_path.write_text(log_header, encoding="utf-8")
    if model.name == "persistence":
        fit_seconds = 0.0
    if hasattr(model, "export_interpret"):
        interpret_path = paths.reports / "interpret" / (run_id + "_graph.npz")
        model.export_interpret(interpret_path)
    del train, valid

    if validation_only:
        dense = None
        if save_validation_arrays:
            from .evaluation.validation_predictions import save_dense_validation
            dense = save_dense_validation(model, cfg, paths, run_id, batch_size)
        if model.validation_metrics is None and dense is None:
            raise ValueError("This model does not provide validation metrics")
        result = {"model": model.name, "table": "validation", "run_id": run_id,
                  "config": asdict(cfg), "model_config": model.model_config,
                  "target_mask": cfg.target_mask, "validation_mask": cfg.target_mask,
                  "eval_mask": None,
                  "target_counts": target_counts,
                  "validation_metrics": (model.validation_metrics if model.validation_metrics is not None
                                         else dense["groups"]["overall"]["model"]),
                  "model_size": {"n_params": int(model.n_params),
                                 "flops_per_sample": model.flops_per_sample},
                  "fit_seconds": fit_seconds, "seed": seed,
                  "experiment": experiment,
                  "data_digest": data_digest, "code_digest": code_digest}
        result["checkpoint"] = checkpoint
        result["features"] = list(features)
        result["history_scale"] = model.history_scale
        result["scaler_source"] = "sdwpf_meta.json" if use_meta else "configured_training_days"
        if dense is not None:
            result["dense_validation"] = dense
        result["result_path"] = str(save_result(
            result, "{}_validation".format(label), paths.evaluation,
            timestamp=stamp))
        append_index(paths, run_id, model.name, "validation", seed,
                     config_digest(cfg, model.model_config),
                     result["result_path"], created.isoformat(), experiment,
                     data_digest, code_digest)
        return {"validation": result}

    test = load_clean(paths, columns=columns,
                      filters=[("Day", ">", cfg.train_days + cfg.val_days)])
    if test.empty:
        raise ValueError("Configured test split is empty")
    test_times = np.sort(test["ts"].unique())
    t_eff = len(test_times) - cfg.horizon
    if t_eff <= 0:
        raise ValueError("Test portion is shorter than the forecast horizon")
    prefix_steps = cfg.input_window - 1
    first_test_day = int(test["Day"].min())
    lookback_days = (prefix_steps + cfg.steps_per_day - 1) // cfg.steps_per_day + 1
    prior = load_clean(paths, columns=columns,
                       filters=[("Day", ">=", max(1, first_test_day - lookback_days)),
                                ("Day", "<", first_test_day)]) if prefix_steps else test.iloc[:0]
    prior_times = np.sort(prior["ts"].unique())
    if len(prior_times) < prefix_steps:
        raise ValueError("Insufficient history before the test segment")
    preceding = prior[prior["ts"].isin(prior_times[-prefix_steps:])] if prefix_steps else prior
    window_frame = pd.concat([preceding, test], ignore_index=True)
    turbine_ids = np.sort(test["TurbID"].unique())
    times, cube = to_feature_cube(window_frame, features, turbine_ids)
    if not np.array_equal(times[prefix_steps:], test_times):
        raise ValueError("Test window timestamps are not aligned")
    del prior, test, preceding, window_frame

    forecasts = np.empty((t_eff, len(turbine_ids), cfg.horizon), dtype=np.float32)
    wind_forecasts = (np.empty_like(forecasts)
                      if hasattr(model, "predict_batch_with_wind") else None)
    phases = None
    if model.needs_phases:
        ns = np.asarray(test_times[:t_eff], dtype="timedelta64[ns]").astype(np.int64)
        phases = ((ns // 600_000_000_000) % cfg.steps_per_day).astype(np.int32)
    predict_start = perf_counter()
    for first, histories in iter_history_batches(
            cube, prefix_steps, t_eff, cfg.input_window, batch_size):
        if model.history_scale == "normalized":
            histories = scaler.transform_histories(histories)
        if wind_forecasts is not None:
            prediction = model.predict_batch_with_wind(histories)
            batch = np.asarray(prediction["power"], dtype=np.float32)
            wind_batch = np.asarray(prediction["wind_speed"], dtype=np.float32)
            if wind_batch.shape != (len(histories), len(turbine_ids), cfg.horizon):
                raise ValueError("Auxiliary wind prediction has the wrong shape")
            if not np.isfinite(wind_batch).all():
                raise ValueError("Auxiliary wind prediction contains nonfinite values")
            wind_forecasts[first:first + len(histories)] = wind_batch
        elif model.needs_phases:
            batch = np.asarray(model.predict_batch(
                histories, phases[first:first + len(histories)]), dtype=np.float32)
        else:
            batch = np.asarray(model.predict_batch(histories), dtype=np.float32)
        expected = (len(histories), len(turbine_ids), cfg.horizon)
        if batch.shape != expected:
            raise ValueError("Model returned shape {}, expected {}".format(batch.shape, expected))
        if not np.isfinite(batch).all():
            raise ValueError("Model returned nonfinite predictions")
        forecasts[first:first + len(histories)] = batch
    predict_seconds = perf_counter() - predict_start
    del cube

    evaluator = Evaluator(cfg, paths=paths)
    if not np.array_equal(evaluator.dte["tids"], turbine_ids) or evaluator.T_eff != t_eff:
        raise ValueError("Model and evaluator test grids differ")
    target_counts["test_m1"] = int(evaluator._valid_stack(evaluator.valid_m1).sum())
    target_counts["test_m2"] = int(evaluator._valid_stack(evaluator.valid_m2).sum())
    timing = {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
              "n_predict_samples": int(t_eff * len(turbine_ids))}
    direct_consist = None
    if wind_forecasts is not None:
        from .models.power_curve import logistic_numpy
        curve = logistic_numpy(wind_forecasts, model.curve_params)
        direct_consist = {"MAE_kW": float(np.mean(np.abs(forecasts - curve),
                                                  dtype=np.float64)),
                          "n": int(forecasts.size)}
    model_size = {"n_params": int(model.n_params),
                  "flops_per_sample": model.flops_per_sample}
    if save_arrays or model.name == "persistence":
        from .figures.io import save_run_arrays
        save_run_arrays(paths, run_id, forecasts, evaluator)
    effective_params = model.model_config
    digest = config_digest(cfg, effective_params)
    sensitivity = evaluator.mask_sensitivity(forecasts)
    results: Dict[str, Any] = {}
    for table in ("main", "all"):
        result = evaluator.evaluate(forecasts, model.name, table)
        result["features"] = list(features)
        result["history_scale"] = model.history_scale
        result["scaler_source"] = "sdwpf_meta.json" if use_meta else "configured_training_days"
        result["model_config"] = effective_params
        result["target_counts"] = target_counts
        result["mask_sensitivity"] = sensitivity
        if model.validation_metrics is not None:
            result["validation_metrics"] = model.validation_metrics
        if direct_consist is not None:
            result["direct_consist_error_kW"] = direct_consist["MAE_kW"]
            result["direct_consist_n"] = direct_consist["n"]
        result["timing"] = timing
        result["model_size"] = model_size
        result["seed"] = seed
        result["experiment"] = experiment
        result["data_digest"] = data_digest
        result["code_digest"] = code_digest
        result["run_id"] = run_id
        result["checkpoint"] = checkpoint
        result["grid"] = {"T_eff": int(t_eff), "N": int(len(turbine_ids)),
                          "H": int(cfg.horizon), "T_test": int(len(test_times))}
        result["result_path"] = str(save_result(
            result, "{}_{}".format(label, table), paths.evaluation, timestamp=stamp))
        append_index(paths, run_id, model.name, table, seed, digest,
                     result["result_path"], created.isoformat(), experiment,
                     data_digest, code_digest)
        results[table] = result
    return results
