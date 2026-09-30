"""Re-evaluate saved M1/M2 forecasts without fitting a model."""
from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np

from ..figures.io import append_index, config_digest, result_provenance, save_run_arrays
from ..paths import ProjectPaths
from .config import ProtocolConfig
from .evaluator import Evaluator
from .report import save_result


def _source_result(paths: ProjectPaths, run_id: str) -> dict:
    matches = []
    for path in paths.evaluation.glob("*_main_*.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if result.get("run_id") == run_id and result.get("table") == "main":
            matches.append(result)
    if len(matches) != 1:
        raise ValueError("Expected one main JSON for run_id {}, found {}".format(
            run_id, len(matches)))
    return matches[0]


def _check_array(name: str, actual: np.ndarray, expected: np.ndarray,
                 atol: float = 0.0) -> None:
    if actual.shape != expected.shape:
        raise ValueError("Source {} shape differs from current data".format(name))
    if not np.allclose(actual, expected, rtol=0.0, atol=atol, equal_nan=True):
        raise ValueError("Source {} differs from current data".format(name))


def rescore(paths: ProjectPaths, run_id: str, eval_mask: str,
            experiment: str, tag: str = "") -> dict:
    """Score a stored prediction cube under the other target population."""
    source = _source_result(paths, run_id)
    source_cfg = dict(source["config"])
    if source_cfg.get("target_mask") not in ("m1", "m2") or (
            source_cfg.get("eval_mask") not in ("m1", "m2")):
        raise ValueError("Legacy M0 run: use scripts/official_mask_rescore.py for historical runs")
    if eval_mask not in ("m1", "m2"):
        raise ValueError("eval_mask must be m1 or m2")
    source_cfg["wind_bins"] = tuple(tuple(x) for x in source_cfg["wind_bins"])
    source_cfg["exclude_flags_main"] = tuple(source_cfg["exclude_flags_main"])
    cfg = ProtocolConfig(**dict(source_cfg, eval_mask=eval_mask))
    old_cfg = ProtocolConfig(**source_cfg)
    array_path = paths.evaluation / (run_id + "_arrays.npz")
    if not array_path.is_file():
        raise FileNotFoundError("Saved prediction array missing: " + str(array_path))
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = ("forecasts", "truth", "valid", "times", "turbine_ids",
                "last_power", "wind_speed", "valid_m1", "valid_m2", "truth_m2")
    absent = [key for key in required if key not in arrays]
    if absent:
        raise ValueError("Saved prediction array lacks: " + ", ".join(absent))
    ev = Evaluator(cfg, paths=paths)
    preds = np.asarray(arrays["forecasts"], dtype=np.float32)
    expected_shape = (ev.T_eff, ev.dte["N"], cfg.horizon)
    if preds.shape != expected_shape or not np.isfinite(preds).all():
        raise ValueError("Saved forecast shape or values differ from current test grid")
    time_ns = np.asarray(ev.dte["times"], dtype="timedelta64[ns]").astype("int64")
    _check_array("times", arrays["times"], time_ns)
    _check_array("turbine_ids", arrays["turbine_ids"], ev.dte["tids"])
    _check_array("last_power", arrays["last_power"],
                 ev.dte["Patv"][:ev.T_eff], atol=1e-4)
    _check_array("wind_speed", arrays["wind_speed"],
                 ev._valid_stack(ev.dte["Wspd"]), atol=1e-4)
    _check_array("valid_m1", arrays["valid_m1"], ev._valid_stack(ev.valid_m1))
    _check_array("valid_m2", arrays["valid_m2"], ev._valid_stack(ev.valid_m2))
    if "truth_m1" in arrays:
        _check_array("truth_m1", arrays["truth_m1"], ev.truth_for("m1"), atol=1e-3)
    elif old_cfg.eval_mask == "m2" and eval_mask == "m1":
        raise ValueError("Saved M2 array lacks truth_m1 needed for M1 rescoring")
    _check_array("truth_m2", arrays["truth_m2"], ev.truth_for("m2"), atol=1e-3)
    source_valid = ev._valid_stack(ev.valid_m1 if old_cfg.eval_mask == "m1" else ev.valid_m2)
    _check_array("valid", arrays["valid"], source_valid)
    source_truth = ev.truth_for(old_cfg.eval_mask)
    _check_array("truth on valid cells", arrays["truth"][source_valid],
                 source_truth[source_valid], atol=1e-3)
    if "eval_mask" in arrays and str(arrays["eval_mask"]) != old_cfg.eval_mask:
        raise ValueError("Saved eval_mask disagrees with source JSON")
    if "target_mask" in arrays and str(arrays["target_mask"]) != old_cfg.target_mask:
        raise ValueError("Saved target_mask disagrees with source JSON")
    old_mae = float(np.abs(preds - source_truth)[source_valid].mean(dtype=np.float64))
    if abs(old_mae - source["A_turbine"]["MAE_kW"]) > 1e-3:
        raise ValueError("Source main MAE does not reproduce from saved forecasts")

    created = datetime.now(timezone.utc)
    stamp = created.strftime("%Y%m%d_%H%M%S_%f")
    new_id = "{}_re{}_{}".format(run_id, eval_mask, stamp)
    data_digest, code_digest = result_provenance(paths)
    save_run_arrays(paths, new_id, preds, ev)
    sensitivity = ev.mask_sensitivity(preds)
    results = {}
    for table in ("main", "all"):
        result = ev.evaluate(preds, source["model"], table)
        for field in ("features", "history_scale", "scaler_source", "model_config",
                      "model_size", "timing", "validation_metrics", "target_counts",
                      "direct_consist_error_kW", "direct_consist_n", "grid"):
            if field in source:
                result[field] = source[field]
        result["mask_sensitivity"] = sensitivity
        result.update({"run_id": new_id, "source_run_id": run_id,
                       "source_data_digest": source.get("data_digest"),
                       "rescored": True, "seed": source.get("seed", 0),
                       "experiment": experiment, "data_digest": data_digest,
                       "code_digest": code_digest})
        label = tag or source["model"] + "_re" + eval_mask
        result["result_path"] = str(save_result(
            result, "{}_{}".format(label, table), paths.evaluation, stamp))
        append_index(paths, new_id, source["model"], table, int(result["seed"]),
                     config_digest(cfg, source.get("model_config", {})),
                     result["result_path"], created.isoformat(), experiment,
                     data_digest, code_digest)
        results[table] = result
    return results
