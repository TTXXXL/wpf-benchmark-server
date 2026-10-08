"""Numerical, paired day-block comparisons of saved benchmark runs.

Reads existing evaluation JSON/NPZ files. Requires only NumPy and the standard
library; it does not train models or import the benchmark package.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


WAVE1_EXPECTED = {"persistence": 1, "gbdt": 1, "agcrn": 1,
                  "barest": 5, "ours_no_wake": 5}
WAVE1_COMPARISONS = (
    ("P1", "ours_no_wake", "barest"),
    ("P2", "barest", "agcrn"),
    ("P3", "agcrn", "persistence"),
    ("P4", "barest", "persistence"),
    ("P5", "ours_no_wake", "persistence"),
    ("P6", "gbdt", "persistence"),
    ("P7", "barest", "gbdt"),
    ("P8", "ours_no_wake", "gbdt"),
)
# Keep these names for callers of the original wave-one script.
EXPECTED = WAVE1_EXPECTED
COMPARISONS = WAVE1_COMPARISONS
PHYSICS_KEYS = ("violation_rate_pct", "truth_violation_rate_pct",
                "curve_coverage_pct", "neg_rate_pct", "over_rated_rate_pct",
                "direct_consist_error_kW")
POWER_BINS = ("low", "mid", "high", "missing")
NS_PER_DAY = 86_400_000_000_000


def _created(value: str) -> datetime:
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return date if date.tzinfo else date.replace(tzinfo=timezone.utc)


def select_runs(eval_dir: Path, experiment: str,
                models: Optional[List[str]] = None,
                expected_seeds: Optional[Dict[str, int]] = None
                ) -> Tuple[List[dict], List[str]]:
    index = eval_dir / "index.jsonl"
    if not index.is_file():
        raise ValueError("Missing index: {}".format(index))
    latest: Dict[Tuple[str, int], dict] = {}
    warnings: List[str] = []
    for line_no, line in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row["experiment"] != experiment or row["table"] != "main":
                continue
            model = str(row["model"])
            if models is not None and model not in models:
                continue
            seed = int(row["seed"])
            _created(str(row["created"]))
            row["run_id"] = str(row["run_id"])
            row["json_path"] = str(row["json_path"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("index.jsonl line {}: {}".format(line_no, exc)) from exc
        key = (model, seed)
        old = latest.get(key)
        if old is None:
            latest[key] = row
        elif _created(row["created"]) >= _created(old["created"]):
            warnings.append("duplicate {} seed {}: discarded {}".format(
                model, seed, old["run_id"]))
            latest[key] = row
        else:
            warnings.append("duplicate {} seed {}: discarded {}".format(
                model, seed, row["run_id"]))
    for model in models or []:
        if not any(name == model for name, _ in latest):
            warnings.append("{}: no main runs for experiment {}".format(model, experiment))
    for model, expected_count in (expected_seeds or {}).items():
        seeds = sorted(seed for name, seed in latest if name == model)
        if len(seeds) != expected_count:
            warnings.append("{}: expected {} seeds, found {} ({})".format(
                model, expected_count, len(seeds), seeds))
        if expected_count > 1 and seeds != list(range(expected_count)):
            warnings.append("{}: expected seeds {}, found {}".format(
                model, list(range(expected_count)), seeds))
    order = {model: i for i, model in enumerate(models or [])}
    runs = [latest[key] for key in sorted(latest,
            key=lambda key: (order.get(key[0], len(order)), key[0], key[1]))]
    return runs, warnings


def _require(condition: bool, run_id: str, check: str) -> None:
    if not condition:
        raise ValueError("{}: {} failed".format(run_id, check))


def _read_run(eval_dir: Path, row: dict, reference: Optional[dict]
              ) -> Tuple[dict, dict, dict]:
    run_id = row["run_id"]
    name = Path(row["json_path"].replace("\\", "/")).name
    json_path = eval_dir / name
    array_path = eval_dir / (run_id + "_arrays.npz")
    _require(json_path.is_file(), run_id, "result JSON exists: " + name)
    _require(array_path.is_file(), run_id, "prediction arrays exist: " + array_path.name)
    result = json.loads(json_path.read_text(encoding="utf-8"))
    for key, expected in (("model", row["model"]), ("seed", row["seed"]),
                          ("run_id", run_id), ("table", "main")):
        _require(result.get(key) == expected, run_id, "JSON " + key)
    try:
        grid = result["grid"]
        shape = (int(grid["T_eff"]), int(grid["N"]), int(grid["H"]))
        horizon = int(result["config"]["horizon"])
        rated = float(result["config"]["rated_power_kw"])
        json_mae = float(result["A_turbine"]["MAE_kW"])
        json_rmse = float(result["A_turbine"]["RMSE_kW"])
        physics = result["D_physics"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("{}: JSON metrics/config/grid failed: {}".format(run_id, exc)) from exc
    _require(len(shape) == 3 and min(shape) > 0 and shape[2] == horizon,
             run_id, "JSON grid/horizon")
    _require(math.isfinite(rated) and rated > 0, run_id, "rated_power_kw")
    _require(isinstance(physics, dict), run_id, "D_physics")
    with np.load(array_path, allow_pickle=False) as archive:
        required = ("forecasts", "truth", "valid", "times", "turbine_ids", "last_power")
        for key in required:
            _require(key in archive, run_id, "array " + key)
        arrays = {key: archive[key] for key in required}
    for key in ("forecasts", "truth", "valid"):
        _require(arrays[key].shape == shape, run_id, key + " shape/grid")
    _require(arrays["times"].shape == (shape[0] + shape[2],), run_id, "times shape")
    _require(arrays["turbine_ids"].shape == (shape[1],), run_id, "turbine_ids shape")
    _require(arrays["last_power"].shape == shape[:2], run_id, "last_power shape")
    _require(arrays["valid"].dtype == np.dtype("bool"), run_id, "valid dtype")
    _require(np.isfinite(arrays["forecasts"]).all(), run_id, "forecasts finite")
    valid = arrays["valid"]
    _require(bool(valid.any()) and np.isfinite(arrays["truth"][valid]).all(),
             run_id, "truth finite on valid cells")
    if reference is not None:
        _require(shape == reference["shape"], run_id, "grid alignment")
        _require(rated == reference["rated"], run_id, "rated_power_kw alignment")
        for key in ("times", "turbine_ids", "valid"):
            _require(np.array_equal(arrays[key], reference[key]), run_id, key + " alignment")
        _require(np.array_equal(arrays["truth"][valid], reference["truth_valid"]),
                 run_id, "truth[valid] alignment")
        _require(np.array_equal(arrays["last_power"], reference["last_power"],
                                equal_nan=True), run_id, "last_power alignment")
    err = (arrays["forecasts"] - arrays["truth"])[valid]
    mae = float(np.abs(err).mean(dtype=np.float64))
    rmse = float(np.sqrt(np.square(err).mean(dtype=np.float64)))
    _require(abs(mae - json_mae) < 1e-3, run_id, "MAE recomputation")
    _require(abs(rmse - json_rmse) < 1e-3, run_id, "RMSE recomputation")
    info = {"model": row["model"], "seed": row["seed"], "run_id": run_id,
            "json_file": name, "arrays_file": array_path.name,
            "MAE_kW": mae, "RMSE_kW": rmse,
            "json_MAE_kW": json_mae, "json_RMSE_kW": json_rmse,
            "physics": {key: (physics.get(key) if key != "direct_consist_error_kW"
                              else result.get(key)) for key in PHYSICS_KEYS}}
    checks = {key: True for key in ("shape_grid", "times", "turbine_ids", "valid",
                                   "truth_valid", "last_power", "forecasts_finite",
                                   "MAE", "RMSE")}
    return arrays, info, checks


def strata(horizon: int) -> List[dict]:
    return ([{"level": "all", "stratum": "all"}] +
            [{"level": "h", "stratum": str(step)} for step in range(1, horizon + 1)] +
            [{"level": "issue_power", "stratum": label} for label in POWER_BINS])


def daily_stats(arrays: dict, days: np.ndarray, rated: float) -> np.ndarray:
    """Return D x S x (n, sum absolute error, sum squared error)."""
    forecasts, truth, valid = (arrays[key] for key in ("forecasts", "truth", "valid"))
    h = forecasts.shape[2]
    issue_days = np.floor_divide(arrays["times"][:forecasts.shape[0]], NS_PER_DAY) + 1
    out = np.zeros((len(days), 1 + h + len(POWER_BINS), 3), dtype=np.float64)
    for d, day in enumerate(days):
        pick = issue_days == day
        v = valid[pick]
        err = forecasts[pick] - truth[pick]
        absolute = np.abs(err)
        squared = np.square(err)

        def put(index: int, mask: np.ndarray) -> None:
            values = absolute[mask]
            out[d, index, 0] = values.size
            out[d, index, 1] = values.sum(dtype=np.float64)
            out[d, index, 2] = squared[mask].sum(dtype=np.float64)

        put(0, v)
        for step in range(h):
            mask = v[:, :, step]
            out[d, 1 + step, 0] = mask.sum()
            out[d, 1 + step, 1] = absolute[:, :, step][mask].sum(dtype=np.float64)
            out[d, 1 + step, 2] = squared[:, :, step][mask].sum(dtype=np.float64)
        last = arrays["last_power"][pick]
        ratio = last / rated
        bins = (np.where(np.isnan(ratio), 3,
                np.where(ratio < 0.05, 0, np.where(ratio < 0.90, 1, 2))))
        for bucket in range(len(POWER_BINS)):
            put(1 + h + bucket, v & (bins[:, :, None] == bucket))
    return out


def _scores(totals: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        mae = totals[..., 1] / totals[..., 0]
        rmse = np.sqrt(totals[..., 2] / totals[..., 0])
    return np.stack((mae, rmse), axis=-1)


def _interval(values: np.ndarray) -> Tuple[List[Optional[float]], int]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return [None, None], 0
    return [float(x) for x in np.quantile(finite, (0.025, 0.975))], int(finite.size)


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator / denominator) if denominator != 0 else None


def compare(runs: List[dict], stats: np.ndarray, labels: List[dict],
            n_boot: int, boot_seed: int, warnings: List[str],
            comparison_defs: Optional[List[Tuple[str, str, str]]] = None) -> List[dict]:
    comparison_defs = comparison_defs if comparison_defs is not None else list(WAVE1_COMPARISONS)
    if not runs:
        for cid, a, b in comparison_defs:
            warnings.append("{}: missing {} and {}".format(cid, a, b))
        return []
    group_names = list(dict.fromkeys(model for _, a, b in comparison_defs
                                     for model in (a, b)))
    groups = {model: [i for i, row in enumerate(runs) if row["model"] == model]
              for model in group_names}
    if not comparison_defs:
        return []
    observed = _scores(stats.sum(axis=1))
    d = stats.shape[1]
    rng = np.random.default_rng(boot_seed)
    counts = rng.multinomial(d, np.full(d, 1.0 / d), size=n_boot)
    weighted = counts @ stats.transpose(1, 0, 2, 3).reshape(d, -1)
    weighted = weighted.reshape((n_boot, len(runs), len(labels), 3))
    boot_scores = _scores(weighted)
    data: Dict[str, np.ndarray] = {}
    data_seed: Dict[str, np.ndarray] = {}
    for model, indices in groups.items():
        if not indices:
            continue
        group = boot_scores[:, indices, :, :]
        data[model] = group.mean(axis=1)
        if len(indices) == 1:
            data_seed[model] = data[model]
        else:
            chosen = rng.integers(0, len(indices), size=(n_boot, len(indices)))
            data_seed[model] = group[np.arange(n_boot)[:, None], chosen].mean(axis=1)
    comparisons: List[dict] = []
    for cid, a, b in comparison_defs:
        if not groups[a] or not groups[b]:
            warnings.append("{}: missing {}".format(
                cid, ", ".join(model for model in (a, b) if not groups[model])))
            continue
        for s, label in enumerate(labels):
            n_cells = int(stats[groups[a][0], :, s, 0].sum())
            if n_cells == 0:
                warnings.append("{} {}:{}: no valid cells".format(
                    cid, label["level"], label["stratum"]))
                continue
            for m, metric in enumerate(("MAE_kW", "RMSE_kW")):
                ma = float(observed[groups[a], s, m].mean())
                mb = float(observed[groups[b], s, m].mean())
                delta = ma - mb
                data_delta = data[a][:, s, m] - data[b][:, s, m]
                both_delta = data_seed[a][:, s, m] - data_seed[b][:, s, m]
                ci_data, nd = _interval(data_delta)
                ci_both, nb = _interval(both_delta)
                if nd < n_boot or nb < n_boot:
                    warnings.append("{} {}:{} {}: valid bootstrap draws data={}, data+seed={}/{}".format(
                        cid, label["level"], label["stratum"], metric, nd, nb, n_boot))
                finite = both_delta[np.isfinite(both_delta)]
                se = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                mde = 2.80 * se
                comparisons.append({"id": cid, "A": a, "B": b,
                    "stratum": label["stratum"], "level": label["level"],
                    "metric": metric, "n_cells": n_cells, "M_A": ma, "M_B": mb,
                    "delta": delta, "rel": _safe_ratio(delta, mb),
                    "ci_data": ci_data, "ci_data_seed": ci_both,
                    "se": se, "mde_abs": mde, "mde_rel": _safe_ratio(mde, mb),
                    "frac_boot_A_worse": (float(np.mean(finite > 0)) if finite.size else None),
                    "n_boot_valid_data": nd, "n_boot_valid_data_seed": nb})
    return comparisons


def seed_level(runs: List[dict], stats: np.ndarray) -> dict:
    groups = {model: {run["seed"]: i for i, run in enumerate(runs)
                      if run["model"] == model}
              for model in ("barest", "ours_no_wake")}
    scores = _scores(stats.sum(axis=1)) if len(runs) else np.empty((0, 0, 2))
    rows = []
    for seed in sorted(set(groups["barest"]) & set(groups["ours_no_wake"])):
        a = scores[groups["ours_no_wake"][seed], 0]
        b = scores[groups["barest"][seed], 0]
        rows.append({"seed": seed,
                     "ours_no_wake": {"MAE_kW": float(a[0]), "RMSE_kW": float(a[1])},
                     "barest": {"MAE_kW": float(b[0]), "RMSE_kW": float(b[1])},
                     "delta_MAE_kW": float(a[0] - b[0]),
                     "delta_RMSE_kW": float(a[1] - b[1])})
    summary = {}
    for metric in ("MAE", "RMSE"):
        values = np.array([row["delta_" + metric + "_kW"] for row in rows])
        summary[metric + "_kW"] = {
            "mean_delta": float(values.mean()) if values.size else None,
            "std_delta": float(values.std(ddof=1)) if values.size > 1 else None,
            "n_negative": int((values < 0).sum()), "n_pairs": int(values.size)}
    within = {}
    for model, mapping in groups.items():
        values = np.array([scores[index, 0, 0] for index in mapping.values()])
        within[model] = {"MAE_seed_std_kW":
                         float(values.std(ddof=1)) if values.size > 1 else None,
                         "n_seeds": int(values.size)}
    return {"paired": rows, "summary": summary, "within_model": within,
            "initialization_pairing": "inferred, not verified bitwise"}


def paired_seed_levels(runs: List[dict], stats: np.ndarray,
                       comparison_defs: List[Tuple[str, str, str]],
                       warnings: List[str]) -> dict:
    """Report shared seed labels only when both groups have multiple seeds."""
    if not runs:
        return {}
    scores = _scores(stats.sum(axis=1))
    groups = {model: {run["seed"]: i for i, run in enumerate(runs)
                      if run["model"] == model}
              for _, a, b in comparison_defs for model in (a, b)}
    output = {}
    for cid, a, b in comparison_defs:
        if len(groups[a]) < 2 or len(groups[b]) < 2:
            continue
        common = sorted(set(groups[a]) & set(groups[b]))
        if not common:
            warnings.append("{}: no shared seed labels for {} and {}".format(cid, a, b))
            continue
        if len(common) < min(len(groups[a]), len(groups[b])):
            warnings.append("{}: {} shared seed labels for {} and {}".format(
                cid, len(common), a, b))
        rows = []
        for seed in common:
            va = scores[groups[a][seed], 0]
            vb = scores[groups[b][seed], 0]
            rows.append({"seed": seed,
                         "A": {"MAE_kW": float(va[0]), "RMSE_kW": float(va[1])},
                         "B": {"MAE_kW": float(vb[0]), "RMSE_kW": float(vb[1])},
                         "delta_MAE_kW": float(va[0] - vb[0]),
                         "delta_RMSE_kW": float(va[1] - vb[1])})
        summary = {}
        for metric in ("MAE", "RMSE"):
            values = np.array([row["delta_" + metric + "_kW"] for row in rows])
            summary[metric + "_kW"] = {
                "mean_delta": float(values.mean()),
                "std_delta": float(values.std(ddof=1)) if len(values) > 1 else None,
                "n_negative": int((values < 0).sum()), "n_pairs": len(values)}
        within = {}
        for model in (a, b):
            values = np.array([scores[index, 0, 0] for index in groups[model].values()])
            within[model] = {"MAE_seed_std_kW": float(values.std(ddof=1)),
                             "n_seeds": len(values)}
        output[cid] = {"A": a, "B": b, "paired": rows,
                       "summary": summary, "within_model": within,
                       "seed_label_pairing": "same numeric seed; initialization equivalence not assumed"}
    return output


def physics_summary(runs: List[dict]) -> dict:
    output = {}
    for model in dict.fromkeys(row["model"] for row in runs):
        model_runs = [row for row in runs if row["model"] == model]
        if not model_runs:
            continue
        group = {}
        for key in PHYSICS_KEYS:
            values = []
            for row in model_runs:
                value = row["physics"].get(key)
                if value is not None and math.isfinite(float(value)):
                    values.append(float(value))
            if values:
                group[key] = {"mean": float(np.mean(values)),
                              "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                              "min": min(values), "max": max(values), "n": len(values)}
        output[model] = group
    return output


def analyze(eval_dir: Path, experiment: str = "wave1", n_boot: int = 10000,
            boot_seed: int = 20260929, preset: str = "wave1",
            models: Optional[List[str]] = None,
            pairs: Optional[List[Tuple[str, str]]] = None,
            expected_seeds: Optional[Dict[str, int]] = None) -> dict:
    if n_boot <= 0:
        raise ValueError("--n-boot must be positive")
    if preset not in ("auto", "wave1"):
        raise ValueError("Unknown preset: {}".format(preset))
    if preset == "wave1":
        if models is not None or pairs is not None or expected_seeds is not None:
            raise ValueError("--preset wave1 cannot be combined with custom models, pairs or seeds")
        model_order = list(WAVE1_EXPECTED)
        seed_expectations = WAVE1_EXPECTED
        comparison_defs = list(WAVE1_COMPARISONS)
    else:
        if models is not None and (not models or len(models) != len(set(models))):
            raise ValueError("--models must contain distinct model names")
        if pairs is not None and any(not a or not b or a == b for a, b in pairs):
            raise ValueError("Each --pair must name two different models")
        seed_expectations = expected_seeds or {}
        if any(count <= 0 for count in seed_expectations.values()):
            raise ValueError("--expect-seeds counts must be positive")
        if models is not None:
            model_order = list(models)
        elif pairs is not None:
            model_order = list(dict.fromkeys(model for pair in pairs for model in pair))
        else:
            model_order = None
        if model_order is not None and any(model not in model_order
                                           for model in seed_expectations):
            raise ValueError("--expect-seeds names a model outside --models/--pair")
        if pairs is not None and model_order is not None and any(
                model not in model_order for pair in pairs for model in pair):
            raise ValueError("--pair names a model outside --models")
    selected, warnings = select_runs(eval_dir, experiment, model_order, seed_expectations)
    if preset == "auto":
        if model_order is None:
            model_order = list(dict.fromkeys(row["model"] for row in selected))
        if pairs is None:
            pairs = list(itertools.combinations(model_order, 2))
        comparison_defs = [("C{}".format(i), a, b)
                           for i, (a, b) in enumerate(pairs, 1)]
        if not selected:
            warnings.append("No main runs for experiment {}".format(experiment))
    runs: List[dict] = []
    stats_list = []
    reference = None
    days = np.array([], dtype=np.int64)
    labels: List[dict] = []
    alignment = {}
    for row in selected:
        arrays, info, checks = _read_run(eval_dir, row, reference)
        if reference is None:
            shape = arrays["forecasts"].shape
            rated = float(json.loads((eval_dir / info["json_file"]).read_text(
                encoding="utf-8"))["config"]["rated_power_kw"])
            reference = {"shape": shape, "rated": rated,
                         "times": arrays["times"], "turbine_ids": arrays["turbine_ids"],
                         "valid": arrays["valid"],
                         "truth_valid": arrays["truth"][arrays["valid"]],
                         "last_power": arrays["last_power"]}
            issue_days = np.floor_divide(arrays["times"][:shape[0]], NS_PER_DAY) + 1
            days = np.unique(issue_days)
            labels = strata(shape[2])
        stats_list.append(daily_stats(arrays, days, reference["rated"]))
        runs.append(info)
        alignment[info["run_id"]] = checks
    stats = (np.stack(stats_list) if stats_list else
             np.empty((0, 0, 0, 3), dtype=np.float64))
    comparisons = compare(runs, stats, labels, n_boot, boot_seed, warnings,
                          comparison_defs)
    paired = seed_level(runs, stats) if preset == "wave1" else None
    seed_pairs = (paired_seed_levels(runs, stats, comparison_defs, warnings)
                  if preset == "auto" else {})
    if preset == "wave1" and runs and not paired["paired"]:
        warnings.append("P1 seed-level: no shared seeds between barest and ours_no_wake")
    return {"meta": {
        "preset": preset,
        "experiment": experiment,
        "models": model_order,
        "comparison_plan": [{"id": cid, "A": a, "B": b}
                            for cid, a, b in comparison_defs],
        "expected_seeds": seed_expectations,
        "runs": [{key: row[key] for key in ("model", "seed", "run_id", "json_file", "arrays_file")}
                 for row in runs],
        "warnings": warnings,
        "alignment_checks": alignment,
        "reference_run_id": runs[0]["run_id"] if runs else None,
        "n_boot": n_boot, "boot_seed": boot_seed,
        "strata": {"h": "1..H", "issue_power": {
            "ratio": "last_power / rated_power_kw", "low": "<0.05",
            "mid": "[0.05,0.90)", "high": ">=0.90", "missing": "NaN"}},
        "issue_days": [int(day) for day in days],
        "seed_std_ddof": 1, "mde_multiplier": 2.80,
        "single_seed_uncertainty": "not included for groups with one seed"},
        "per_run": [{key: value for key, value in row.items() if key != "physics"}
                    for row in runs],
        "comparisons": comparisons,
        "seed_level": paired,
        "seed_levels": seed_pairs,
        "physics": physics_summary(runs)}


def _number(value: Optional[float], percent: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return "{:.2f}{}".format(value * 100 if percent else value,
                             "%" if percent else "")


def _ci(values: List[Optional[float]]) -> str:
    return "[{}, {}]".format(_number(values[0]), _number(values[1]))


def make_markdown(summary: dict) -> str:
    meta = summary["meta"]
    title = ("Wave 1 paired bootstrap" if meta["preset"] == "wave1"
             else "Paired bootstrap: {}".format(meta["experiment"]))
    lines = ["# " + title, "",
             "n_boot: {}; boot_seed: {}; issue_days: {}".format(
                 meta["n_boot"], meta["boot_seed"], meta["issue_days"]), "",
             "## Runs and warnings", "",
             "| model | seed | run_id | JSON | arrays |",
             "| --- | ---: | --- | --- | --- |"]
    for run in meta["runs"]:
        lines.append("| {model} | {seed} | {run_id} | {json_file} | {arrays_file} |".format(**run))
    if not meta["runs"]:
        lines.append("| — | — | — | — | — |")
    lines.extend(("", "Warnings:", ""))
    lines.extend("- {}".format(warning) for warning in meta["warnings"])
    if not meta["warnings"]:
        lines.append("- None")
    by_key = {(c["id"], c["level"], c["stratum"], c["metric"]): c
              for c in summary["comparisons"]}

    def table(title: str, rows: List[dict], include_n: bool = False) -> None:
        lines.extend(("", "## " + title, "",
            "| id | A | B | stratum |{} M_A (kW) | M_B (kW) | Δ (kW) | rel | CI(b) (kW) | CI(a) (kW) | MDE (kW) |".format(
                " n_cells |" if include_n else ""),
            "| --- | --- | --- | --- |{} ---: | ---: | ---: | ---: | --- | --- | ---: |".format(
                " ---: |" if include_n else "")))
        for c in rows:
            lines.append("| {id} | {A} | {B} | {stratum} |{n} {ma} | {mb} | {delta} | {rel} | {both} | {data} | {mde} |".format(
                id=c["id"], A=c["A"], B=c["B"], stratum=c["stratum"],
                n=" {:,} |".format(c["n_cells"]) if include_n else "",
                ma=_number(c["M_A"]), mb=_number(c["M_B"]),
                delta=_number(c["delta"]), rel=_number(c["rel"], True),
                both=_ci(c["ci_data_seed"]), data=_ci(c["ci_data"]),
                mde=_number(c["mde_abs"])))
        if not rows:
            lines.append("| — | — | — | — |{} — | — | — | — | — | — | — |".format(
                " — |" if include_n else ""))

    lines.extend(("", "## Overall (MAE / RMSE)", "",
        "| id | A | B | M_A (kW) | M_B (kW) | Δ (kW) | rel | CI(b) (kW) | CI(a) (kW) | MDE (kW) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |"))
    overall_count = 0
    for plan in meta["comparison_plan"]:
        cid, a, b = plan["id"], plan["A"], plan["B"]
        keys = [(cid, "all", "all", metric) for metric in ("MAE_kW", "RMSE_kW")]
        if not all(key in by_key for key in keys):
            continue
        mae, rmse = (by_key[key] for key in keys)
        pair = (mae, rmse)
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            cid, a, b,
            " / ".join(_number(row["M_A"]) for row in pair),
            " / ".join(_number(row["M_B"]) for row in pair),
            " / ".join(_number(row["delta"]) for row in pair),
            " / ".join(_number(row["rel"], True) for row in pair),
            " / ".join(_ci(row["ci_data_seed"]) for row in pair),
            " / ".join(_ci(row["ci_data"]) for row in pair),
            " / ".join(_number(row["mde_abs"]) for row in pair)))
        overall_count += 1
    if not overall_count:
        lines.append("| — | — | — | — | — | — | — | — | — | — |")
    horizon_ids = ({"P1", "P4", "P6"} if meta["preset"] == "wave1"
                   else {plan["id"] for plan in meta["comparison_plan"]})
    horizon_rows = [c for c in summary["comparisons"] if c["id"] in horizon_ids
                    and c["level"] == "h" and c["metric"] == "MAE_kW"]
    table("Horizon MAE", horizon_rows)
    power_rows = [c for c in summary["comparisons"] if c["level"] == "issue_power"
                  and c["metric"] == "MAE_kW"]
    table("Issue-time power MAE", power_rows, include_n=True)
    lines.extend(("", "CI(a): day blocks; CI(b): day blocks and seed resampling.",
                  "One-seed groups have no estimated seed uncertainty. MDE = 2.80 × SE (normal approximation)."))
    if meta["preset"] == "wave1":
        lines.extend(("", "## P1 seed level", "",
                      "| seed | ours MAE | barest MAE | Δ MAE | ours RMSE | barest RMSE | Δ RMSE |",
                      "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"))
        for row in summary["seed_level"]["paired"]:
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                row["seed"], _number(row["ours_no_wake"]["MAE_kW"]),
                _number(row["barest"]["MAE_kW"]), _number(row["delta_MAE_kW"]),
                _number(row["ours_no_wake"]["RMSE_kW"]),
                _number(row["barest"]["RMSE_kW"]), _number(row["delta_RMSE_kW"])))
        if not summary["seed_level"]["paired"]:
            lines.append("| — | — | — | — | — | — | — |")
        lines.extend(("", "| metric | mean Δ | std Δ | Δ < 0 | pairs |",
                      "| --- | ---: | ---: | ---: | ---: |"))
        for metric, row in summary["seed_level"]["summary"].items():
            lines.append("| {} | {} | {} | {} | {} |".format(
                metric, _number(row["mean_delta"]), _number(row["std_delta"]),
                row["n_negative"], row["n_pairs"]))
        lines.extend(("", "| model | MAE seed std (kW) | n_seeds |",
                      "| --- | ---: | ---: |"))
        for model, row in summary["seed_level"]["within_model"].items():
            lines.append("| {} | {} | {} |".format(
                model, _number(row["MAE_seed_std_kW"]), row["n_seeds"]))
    else:
        for cid, group in summary["seed_levels"].items():
            a, b = group["A"], group["B"]
            lines.extend(("", "## {} shared seed labels: {} − {}".format(cid, a, b), "",
                          "| seed | A MAE | B MAE | Δ MAE | A RMSE | B RMSE | Δ RMSE |",
                          "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"))
            for row in group["paired"]:
                lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                    row["seed"], _number(row["A"]["MAE_kW"]),
                    _number(row["B"]["MAE_kW"]), _number(row["delta_MAE_kW"]),
                    _number(row["A"]["RMSE_kW"]), _number(row["B"]["RMSE_kW"]),
                    _number(row["delta_RMSE_kW"])))
            lines.extend(("", "| metric | mean Δ | std Δ | Δ < 0 | pairs |",
                          "| --- | ---: | ---: | ---: | ---: |"))
            for metric, row in group["summary"].items():
                lines.append("| {} | {} | {} | {} | {} |".format(
                    metric, _number(row["mean_delta"]), _number(row["std_delta"]),
                    row["n_negative"], row["n_pairs"]))
            lines.extend(("", "| model | MAE seed std (kW) | n_seeds |",
                          "| --- | ---: | ---: |"))
            for model, row in group["within_model"].items():
                lines.append("| {} | {} | {} |".format(
                    model, _number(row["MAE_seed_std_kW"]), row["n_seeds"]))
        if summary["seed_levels"]:
            lines.extend(("", "Shared numeric seed labels are paired; matching model initialization is not assumed."))
    lines.extend(("", "## Physics metrics", "",
                  "| model | metric | mean | std | min | max | n |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"))
    for model, group in summary["physics"].items():
        for key, row in group.items():
            suffix = "%" if key.endswith("_pct") else ""
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                model, key, _number(row["mean"]) + suffix,
                _number(row["std"]) + suffix, _number(row["min"]) + suffix,
                _number(row["max"]) + suffix, row["n"]))
    lines.extend(("", "D_physics uses observed wind speed at the target time with a curve fitted on training data. "
                  "direct_consist_error_kW uses the PIN wind-head output and is directly optimized in training.", ""))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None, default_preset: str = "wave1") -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=Path("reports/eval"))
    parser.add_argument("--experiment", help="Index experiment label (default: wave1 for preset, standard for auto)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--boot-seed", type=int, default=20260929)
    parser.add_argument("--preset", choices=("auto", "wave1"),
                        help="auto discovers models; wave1 reproduces the original fixed plan")
    parser.add_argument("--models", help="Comma-separated model names, in A/B order")
    parser.add_argument("--pair", action="append", metavar="A:B",
                        help="Directed comparison; repeat to select several pairs")
    parser.add_argument("--expect-seeds", action="append", metavar="MODEL=N",
                        help="Expected number of seeds for a model; repeat as needed")
    parser.add_argument("--output-prefix", help="Output filename stem (without extension)")
    args = parser.parse_args(argv)
    try:
        models = None
        if args.models is not None:
            models = [model.strip() for model in args.models.split(",")]
            if any(not model for model in models):
                raise ValueError("--models contains an empty model name")
        pairs = None
        if args.pair is not None:
            pairs = []
            for value in args.pair:
                parts = [part.strip() for part in value.split(":")]
                if len(parts) != 2 or not all(parts):
                    raise ValueError("--pair must have the form A:B")
                pairs.append((parts[0], parts[1]))
            if len(pairs) != len(set(pairs)):
                raise ValueError("Duplicate --pair")
        expected = None
        if args.expect_seeds is not None:
            expected = {}
            for value in args.expect_seeds:
                if "=" not in value:
                    raise ValueError("--expect-seeds must have the form MODEL=N")
                model, count = [part.strip() for part in value.split("=", 1)]
                if not model or model in expected:
                    raise ValueError("--expect-seeds has an empty or duplicate model")
                expected[model] = int(count)
        preset = args.preset or ("auto" if any(value is not None for value in
                                    (models, pairs, expected)) else default_preset)
        experiment = args.experiment or ("wave1" if preset == "wave1" else "standard")
        summary = analyze(args.eval_dir, experiment, args.n_boot, args.boot_seed,
                          preset=preset, models=models, pairs=pairs,
                          expected_seeds=expected)
        prefix = args.output_prefix or ("wave1_paired_summary" if preset == "wave1"
                                        else "paired_summary")
        if prefix in ("", ".", "..") or "/" in prefix or "\\" in prefix or ":" in prefix:
            raise ValueError("--output-prefix must be a filename stem")
        args.out.mkdir(parents=True, exist_ok=True)
        json_path = args.out / (prefix + ".json")
        md_path = args.out / (prefix + ".md")
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8")
        md_path.write_text(make_markdown(summary), encoding="utf-8")
        print(json_path)
        print(md_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(default_preset="auto"))
