# -*- coding: utf-8 -*-
"""Ablate SDWPF target-time exclusion rules on saved Wave 1 forecasts.

Reads evaluation NPZ/JSON, the raw CSV and the existing clean Parquet. It never
trains a model or writes either input data file. Counts are forecast cells
(issue time x turbine x horizon), not distinct raw observations.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import official_mask_rescore as om
import paired_bootstrap as wb


FLAG_COLUMNS = ("m_missing", "m_imputed", "m_outlier",
                "f_fault", "f_curtail", "f_farm")
STAGE_NAMES = ("M0", "only_zero", "only_pitch", "only_direction", "M1", "M2")
RULE_NAMES = ("zero", "pitch", "direction")


def load_clean_grid(path: Path, times: np.ndarray, turbines: np.ndarray
                    ) -> Dict[str, np.ndarray]:
    """Read-only alignment of clean target labels and six repository flags."""
    day = np.floor_divide(times.astype(np.int64), om.NS_PER_DAY) + 1
    columns = ["ts", "Day", "TurbID", "Patv"] + list(FLAG_COLUMNS)
    frame = pd.read_parquet(path, columns=columns,
                            filters=[("Day", ">=", int(day.min())),
                                     ("Day", "<=", int(day.max()))])
    if frame.duplicated(["ts", "TurbID"]).any():
        raise ValueError("Clean Parquet has duplicate (ts, TurbID) rows")
    ts = frame["ts"].to_numpy(dtype="timedelta64[ns]").astype(np.int64)
    ti = pd.Index(times.astype(np.int64)).get_indexer(ts)
    ni = pd.Index(turbines.astype(np.int64)).get_indexer(
        frame["TurbID"].to_numpy(dtype=np.int64))
    keep = (ti >= 0) & (ni >= 0)
    shape = (len(times), len(turbines))
    covered = np.zeros(shape, dtype=bool)
    covered[ti[keep], ni[keep]] = True
    if not covered.all():
        raise ValueError("Clean Parquet does not cover {:,} test-grid rows".format(
            int((~covered).sum())))
    result: Dict[str, np.ndarray] = {}
    clean_power = np.full(shape, np.nan, dtype=np.float64)
    clean_power[ti[keep], ni[keep]] = frame["Patv"].to_numpy(dtype=np.float64)[keep]
    result["Patv"] = clean_power
    for name in FLAG_COLUMNS:
        values = frame[name].to_numpy()
        if pd.isna(values).any():
            raise ValueError("Clean Parquet flag {} contains null".format(name))
        grid = np.zeros(shape, dtype=bool)
        grid[ti[keep], ni[keep]] = values[keep].astype(bool)
        result[name] = grid
    return result


def build_groups(m0: np.ndarray, official: Dict[str, np.ndarray],
                 clean: Dict[str, np.ndarray], raw_patv: np.ndarray
                 ) -> Tuple[Dict[str, np.ndarray], dict]:
    """Create score stages and disjoint diagnostic groups on target cells."""
    t_eff, _, horizon = m0.shape
    rules = {"zero": om.target_stack(official["zero_windy"], t_eff, horizon),
             "pitch": om.target_stack(official["pitch"], t_eff, horizon),
             "direction": om.target_stack(official["abnormal"], t_eff, horizon)}
    bad = rules["zero"] | rules["pitch"] | rules["direction"]
    finite = om.target_stack(official["finite"], t_eff, horizon)
    groups = {
        "M0": m0,
        "only_zero": m0 & ~rules["zero"],
        "only_pitch": m0 & ~rules["pitch"],
        "only_direction": m0 & ~rules["direction"],
        "M1": m0 & ~bad,
        "M2": finite & ~bad,
    }
    extra = groups["M2"] & ~groups["M1"]
    groups["M2_extra"] = extra
    for name in RULE_NAMES:
        groups["hit_" + name] = m0 & rules[name]
    groups["hit_official_union"] = m0 & bad

    # Exact combinations form disjoint partitions. Marginals below deliberately
    # overlap; neither their counts nor their percentages should be summed.
    rule_bits = sum((rules[name].astype(np.uint8) << i)
                    for i, name in enumerate(RULE_NAMES))
    rule_combos = {}
    for bits in range(1, 8):
        count = int((m0 & (rule_bits == bits)).sum())
        if count:
            label = "rule_combo_" + "+".join(
                name for i, name in enumerate(RULE_NAMES) if bits & (1 << i))
            groups[label] = m0 & (rule_bits == bits)
            rule_combos[label] = count

    clean_flags = {name: om.target_stack(clean[name], t_eff, horizon)
                   for name in FLAG_COLUMNS}
    reconstructed = om.target_stack(np.isfinite(clean["Patv"]), t_eff, horizon)
    for flag in clean_flags.values():
        reconstructed &= ~flag
    if not np.array_equal(reconstructed, m0):
        raise ValueError("Existing Parquet flags do not reconstruct NPZ M0 valid mask")

    flag_bits = sum((clean_flags[name].astype(np.uint8) << i)
                    for i, name in enumerate(FLAG_COLUMNS))
    flag_combos = {}
    for bits in range(64):
        count = int((extra & (flag_bits == bits)).sum())
        if count:
            label = ("extra_combo_none" if bits == 0 else "extra_combo_" + "+".join(
                name for i, name in enumerate(FLAG_COLUMNS) if bits & (1 << i)))
            groups[label] = extra & (flag_bits == bits)
            flag_combos[label] = count
    clean_missing = int((extra & ~om.target_stack(
        np.isfinite(clean["Patv"]), t_eff, horizon)).sum())
    raw_negative = int((extra & (om.target_stack(raw_patv, t_eff, horizon) < 0)).sum())

    counts = {name: int(mask.sum()) for name, mask in groups.items()}
    checks = {
        "group_counts": counts,
        "sequential_removed": {
            "zero": counts["M0"] - counts["only_zero"],
            "pitch_after_zero": counts["only_zero"] - int((m0 & ~rules["zero"] & ~rules["pitch"]).sum()),
            "direction_after_zero_pitch": int((m0 & ~rules["zero"] & ~rules["pitch"]).sum()) - counts["M1"],
        },
        "rule_marginals": {name: counts["hit_" + name] for name in RULE_NAMES},
        "rule_exact_combinations": rule_combos,
        "extra_flag_marginals": {name: int((extra & clean_flags[name]).sum())
                                 for name in FLAG_COLUMNS},
        "extra_flag_exact_combinations": flag_combos,
        "extra_clean_patv_nonfinite": clean_missing,
        "extra_raw_patv_negative_clipped": raw_negative,
    }
    if sum(rule_combos.values()) != counts["M0"] - counts["M1"]:
        raise ValueError("Rule combinations do not partition M0-M1")
    if sum(flag_combos.values()) != counts["M2_extra"]:
        raise ValueError("Flag combinations do not partition M2 extra cells")
    if counts["M2"] - counts["M1"] != counts["M2_extra"]:
        raise ValueError("M2/M1 set relationship failed")
    return groups, checks


def _aggregate(mask: np.ndarray, absolute: np.ndarray, squared: np.ndarray,
               truth: np.ndarray,
               index: Optional[np.ndarray] = None,
               size: Optional[int] = None) -> np.ndarray:
    selected = mask.ravel()
    y = truth[selected]
    if index is None:
        return np.array([[int(selected.sum()),
                          float(absolute[selected].sum(dtype=np.float64)),
                          float(squared[selected].sum(dtype=np.float64)),
                          float(y.sum(dtype=np.float64)),
                          float(np.square(y).sum(dtype=np.float64))]])
    return np.stack((np.bincount(index[selected], minlength=size),
                     np.bincount(index[selected], weights=absolute[selected], minlength=size),
                     np.bincount(index[selected], weights=squared[selected], minlength=size),
                     np.bincount(index[selected], weights=y, minlength=size),
                     np.bincount(index[selected], weights=np.square(y), minlength=size)), axis=1)


def _score_row(n: int, abs_sum: float, sq_sum: float,
               truth_sum: float, truth_sq_sum: float, rated_kw: float) -> dict:
    mae = abs_sum / n if n else np.nan
    centered_truth_sq_sum = truth_sq_sum - truth_sum * truth_sum / n if n else np.nan
    return {"n_cells": n,
            "MAE_kW": mae,
            "RMSE_kW": np.sqrt(sq_sum / n) if n else np.nan,
            "NMAE_pct": 100.0 * mae / rated_kw if n else np.nan,
            "R2": 1.0 - sq_sum / centered_truth_sq_sum
            if n and centered_truth_sq_sum > 0 else np.nan}


def _summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["scope", "unit", "group", "model"]
    out = frame.groupby(keys, as_index=False, sort=False).agg(
        n_cells=("n_cells", "first"), MAE_kW=("MAE_kW", "mean"),
        RMSE_kW=("RMSE_kW", "mean"), NMAE_pct=("NMAE_pct", "mean"),
        R2=("R2", "mean"), n_seeds=("seed", "nunique"))
    return out


def _model_deltas(scores: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    lookup = scores.set_index(["scope", "unit", "group", "model"])
    rows = []
    for a, b in itertools.combinations(models, 2):
        left = lookup.xs(a, level="model")
        right = lookup.xs(b, level="model")
        matched = left[["MAE_kW", "RMSE_kW", "NMAE_pct", "R2", "n_cells"]].join(
            right[["MAE_kW", "RMSE_kW", "NMAE_pct", "R2", "n_cells"]],
            lsuffix="_A", rsuffix="_B", how="inner")
        for (scope, unit, group), row in matched.iterrows():
            if int(row["n_cells_A"]) != int(row["n_cells_B"]):
                raise ValueError("Model score masks differ for {}".format(group))
            if int(row["n_cells_A"]) == 0:
                continue
            rows.append({"scope": scope, "unit": unit, "group": group,
                         "A": a, "B": b, "n_cells": int(row["n_cells_A"]),
                         "delta_MAE_kW": row["MAE_kW_A"] - row["MAE_kW_B"],
                         "delta_RMSE_kW": row["RMSE_kW_A"] - row["RMSE_kW_B"],
                         "delta_NMAE_pct": row["NMAE_pct_A"] - row["NMAE_pct_B"],
                         "delta_R2": row["R2_A"] - row["R2_B"]})
    return pd.DataFrame(rows)


def _coverage(scores: pd.DataFrame, model: str, counts: dict,
              candidate_counts: dict) -> pd.DataFrame:
    frame = scores[scores.model == model][["scope", "unit", "group", "n_cells"]].copy()
    frame["candidate_cells"] = [candidate_counts[(scope, unit)]
                                for scope, unit in zip(frame.scope, frame.unit)]
    frame["pct_of_scope_candidates"] = 100 * frame.n_cells / frame.candidate_cells
    frame["pct_of_group"] = [100 * n / counts[group] if counts[group] else 0.0
                             for n, group in zip(frame.n_cells, frame.group)]
    return frame


def analyze(eval_dir: Path, raw_path: Path, clean_path: Path,
            experiment: str, n_boot: int, boot_seed: int) -> dict:
    selected, warnings = wb.select_runs(eval_dir, experiment, expected_seeds=wb.WAVE1_EXPECTED)
    if not selected:
        raise ValueError("No saved main runs for experiment {}".format(experiment))
    models = list(dict.fromkeys(row["model"] for row in selected))
    models = [name for name in wb.WAVE1_EXPECTED if name in models] + [
        name for name in models if name not in wb.WAVE1_EXPECTED]
    reference = None
    run_rows: List[dict] = []
    issue_stats = []
    run_info = []
    for row in selected:
        arrays, info, _ = wb._read_run(eval_dir, row, reference)
        if reference is None:
            shape = arrays["forecasts"].shape
            reference = {"shape": shape, "rated": float(json.loads(
                (eval_dir / info["json_file"]).read_text(encoding="utf-8"))
                ["config"]["rated_power_kw"]),
                "times": arrays["times"], "turbine_ids": arrays["turbine_ids"],
                "valid": arrays["valid"], "truth_valid": arrays["truth"][arrays["valid"]],
                "last_power": arrays["last_power"]}
            t_eff, n_turbines, horizon = shape
            raw = om.load_raw_grid(raw_path, np.asarray(arrays["times"], dtype=np.int64),
                                   np.asarray(arrays["turbine_ids"]))
            official = om.official_flags(raw)
            clean = load_clean_grid(clean_path, arrays["times"], arrays["turbine_ids"])
            groups, checks = build_groups(arrays["valid"], official, clean, raw["Patv"])
            truth_raw = om.target_stack(np.maximum(raw["Patv"], 0.0), t_eff, horizon)
            truth_diff = np.abs(arrays["truth"][arrays["valid"]].astype(np.float64) -
                                truth_raw[arrays["valid"]])
            if np.any(truth_diff > om.TRUTH_TOLERANCE_KW):
                raise ValueError("NPZ and raw truth mismatch on M0 cells")
            checks["truth_max_abs_diff_kW"] = float(truth_diff.max())
            truth_npz_flat = arrays["truth"].astype(np.float64).ravel()
            truth_raw_flat = truth_raw.ravel()
            # Diagnostics use target day; bootstrap uses issue day.
            target_times = np.stack([arrays["times"][h:t_eff + h]
                                     for h in range(1, horizon + 1)], axis=1)
            target_day = np.broadcast_to(np.floor_divide(target_times[:, None, :],
                                                        om.NS_PER_DAY) + 1, shape).ravel()
            days, day_idx = np.unique(target_day, return_inverse=True)
            turbine_idx = np.broadcast_to(np.arange(n_turbines)[None, :, None], shape).ravel()
            issue_day = np.floor_divide(arrays["times"][:t_eff], om.NS_PER_DAY) + 1
            issue_days, issue_idx_small = np.unique(issue_day, return_inverse=True)
            issue_idx = np.broadcast_to(issue_idx_small[:, None, None], shape).ravel()
            labels = [{"level": name, "stratum": "all"} for name in STAGE_NAMES]
        forecast = arrays["forecasts"].astype(np.float64)
        error_npz = forecast - arrays["truth"]
        error_raw = forecast - truth_raw
        abs_npz, sq_npz = np.abs(error_npz).ravel(), np.square(error_npz).ravel()
        abs_raw, sq_raw = np.abs(error_raw).ravel(), np.square(error_raw).ravel()
        one_issue = np.zeros((len(issue_days), len(labels), 3), dtype=np.float64)
        for name, mask in groups.items():
            absolute, squared, truth = (
                (abs_raw, sq_raw, truth_raw_flat) if name in ("M2", "M2_extra") or
                name.startswith("extra_combo_") else (abs_npz, sq_npz, truth_npz_flat))
            total = _aggregate(mask, absolute, squared, truth)[0]
            run_rows.append(dict(scope="overall", unit="all", group=name,
                                 model=info["model"], seed=info["seed"], run_id=info["run_id"],
                                 **_score_row(int(total[0]), *total[1:], reference["rated"])))
            for scope, units, index in (("target_day", days, day_idx),
                                        ("turbine", arrays["turbine_ids"], turbine_idx)):
                values = _aggregate(mask, absolute, squared, truth, index, len(units))
                for j, unit in enumerate(units):
                    n, a, s, y, yy = values[j]
                    if n:
                        run_rows.append(dict(scope=scope, unit=str(int(unit)), group=name,
                                             model=info["model"], seed=info["seed"],
                                             run_id=info["run_id"],
                                             **_score_row(int(n), a, s, y, yy,
                                                          reference["rated"])))
            if name in STAGE_NAMES:
                one_issue[:, STAGE_NAMES.index(name), :] = _aggregate(
                    mask, absolute, squared, truth, issue_idx, len(issue_days))[:, :3]
        # The original M0 score must be recreated before any comparison.
        m0 = next(item for item in run_rows[::-1] if item["scope"] == "overall" and
                  item["group"] == "M0" and item["model"] == info["model"] and
                  item["seed"] == info["seed"])
        if abs(m0["MAE_kW"] - info["json_MAE_kW"]) > 1e-3 or abs(
                m0["RMSE_kW"] - info["json_RMSE_kW"]) > 1e-3:
            raise ValueError("M0 score does not reproduce JSON for {}".format(info["run_id"]))
        issue_stats.append(one_issue)
        run_info.append(info)

    per_run = pd.DataFrame(run_rows)
    scores = _summarize_runs(per_run)
    deltas = _model_deltas(scores, models)
    candidate_counts = {("overall", "all"): int(np.prod(shape))}
    candidate_counts.update({("target_day", str(int(day))): int(n)
                             for day, n in zip(days, np.bincount(day_idx))})
    candidate_counts.update({("turbine", str(int(turbine))): int(t_eff * horizon)
                             for turbine in arrays["turbine_ids"]})
    coverage = _coverage(scores, models[0], checks["group_counts"], candidate_counts)
    defs = [("C{}".format(i), a, b)
            for i, (a, b) in enumerate(itertools.combinations(models, 2), 1)]
    comparisons = wb.compare(run_info, np.stack(issue_stats), labels,
                             n_boot, boot_seed, warnings, defs)
    for comp in comparisons:
        comp["group"] = comp.pop("level")
    metadata = {"experiment": experiment, "eval_dir": str(eval_dir),
                "raw_csv": str(raw_path), "clean_parquet": str(clean_path),
                "n_boot": n_boot, "boot_seed": boot_seed,
                "runs": [{key: info[key] for key in ("run_id", "model", "seed", "arrays_file")}
                         for info in run_info],
                "issue_days": issue_days.tolist(), "target_days": days.tolist(),
                "warnings": warnings}
    return {"checks": checks, "models": models, "metadata": metadata,
            "warnings": warnings,
            "issue_days": issue_days.tolist(), "target_days": days.tolist(),
            "per_run": per_run, "scores": scores, "deltas": deltas,
            "coverage": coverage,
            "bootstrap": pd.DataFrame(comparisons)}


def write_outputs(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("per_run", "scores", "deltas", "bootstrap", "coverage"):
        result[name].to_csv(out / (name + ".csv"), index=False, encoding="utf-8-sig")
    checks = result["checks"]
    (out / "counts.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(result["metadata"], ensure_ascii=False,
                                                indent=2) + "\n", encoding="utf-8")
    scores = result["scores"]
    totals = scores[(scores.scope == "overall") & scores.group.isin(STAGE_NAMES)]
    lines = ["# SDWPF 规则消融（已有预测，不重训）", "",
             "实验：{}；预测运行 {} 份；bootstrap {} 次。".format(
                 result["metadata"]["experiment"], len(result["metadata"]["runs"]),
                 result["metadata"]["n_boot"]),
             "计数单位：发布时刻 × 风机 × 预测步长。按天明细使用目标日；配对区间按发布日抽样。",
             "MAE/RMSE/NMAE/R² 是逐机预测单元指标，不是 KDD Cup 原始计分公式。",
             "NMAE = MAE / 1550 kW × 100%；R² 在每个名单内以真实功率的总体均值为基准；多种子先逐运行计算再取均值。", "",
             "## 样本数量", "",
             "| 评分名单 | 单元数 | 从 M0 剔除 | 额外纳入 |", "| --- | ---: | ---: | ---: |"]
    for name in STAGE_NAMES:
        n = checks["group_counts"][name]
        removed = (checks["group_counts"]["M0"] - checks["group_counts"]["M1"]
                   if name == "M2" else checks["group_counts"]["M0"] - n)
        added = checks["group_counts"]["M2_extra"] if name == "M2" else 0
        lines.append("| {} | {:,} | {:,} | {:,} |".format(name, n, removed, added))
    lines += ["", "顺序加入零功率、叶片角度、方向异常时，新增剔除：" +
              "、".join("{} {:,}".format(k, v) for k, v in checks["sequential_removed"].items()) + "。",
              "", "## 规则重叠（M0 内，互斥组合）", ""]
    for name, n in checks["rule_exact_combinations"].items():
        lines.append("- {}：{:,}".format(name, n))
    lines += ["", "## M2 比 M1 多出的单元", "",
              "总数：{:,}。下列单项命中可重叠。".format(checks["group_counts"]["M2_extra"]), ""]
    for name, n in checks["extra_flag_marginals"].items():
        lines.append("- {}：{:,}".format(name, n))
    lines += ["- 清洗后 Patv 非有限值：{:,}".format(checks["extra_clean_patv_nonfinite"]),
              "", "互斥标记组合见 counts.json；其格数之和等于 M2_extra。", "",
              "## 总体指标（种子均值）", "",
              "| 名单 | 模型 | n | MAE (kW) | RMSE (kW) | NMAE (%) | R² |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in STAGE_NAMES:
        for model in result["models"]:
            row = totals[(totals.group == name) & (totals.model == model)].iloc[0]
            lines.append("| {} | {} | {:,} | {:.2f} | {:.2f} | {:.2f} | {:.4f} |".format(
                name, model, int(row.n_cells), row.MAE_kW, row.RMSE_kW,
                row.NMAE_pct, row.R2))
    lines += ["", "## 同名单模型差值（A − B，kW）", "",
              "负值表示 A 在该名单上的误差较小。", "",
              "| 名单 | A − B | ΔMAE | ΔRMSE |", "| --- | --- | ---: | ---: |"]
    deltas = result["deltas"]
    for name in STAGE_NAMES:
        part = deltas[(deltas.scope == "overall") & (deltas.group == name)]
        for row in part.itertuples():
            lines.append("| {} | {} − {} | {:+.2f} | {:+.2f} |".format(
                name, row.A, row.B, row.delta_MAE_kW, row.delta_RMSE_kW))
    lines += ["", "发布日块与种子重抽区间见 bootstrap.csv。",
              "按目标日和风机的各组成绩见 scores.csv；各组占比见 coverage.csv；逐种子结果见 per_run.csv。",
              "不同名单包含的样本不同，跨名单 MAE 的升降不能当作模型性能变化。", ""]
    lines += ["## 集中出现的目标日与风机", "",
              "以下按组内单元数排序；百分比是该日/风机的候选预测单元中属于该组的比例。", ""]
    coverage = result["coverage"]
    for group in ("hit_zero", "hit_pitch", "hit_official_union", "M2_extra"):
        lines.append("### {}（{:,} 格）".format(group, checks["group_counts"][group]))
        lines.append("")
        for scope, label in (("target_day", "目标日"), ("turbine", "风机")):
            part = coverage[(coverage.group == group) & (coverage.scope == scope)]
            part = part.sort_values("n_cells", ascending=False).head(5)
            text = "；".join("{}: {:,} ({:.1f}%)".format(
                row.unit, int(row.n_cells), row.pct_of_scope_candidates)
                for row in part.itertuples())
            lines.append("- {}：{}".format(label, text or "无"))
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=Path("reports/eval"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/sdwpf/sdwpf_245days_v1.csv"))
    parser.add_argument("--clean", type=Path, default=Path("data/processed/sdwpf_clean.parquet"))
    parser.add_argument("--experiment", default="wave1")
    parser.add_argument("--out", type=Path, default=Path("reports/mask_rule_ablation"))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--boot-seed", type=int, default=20260929)
    args = parser.parse_args(argv)
    if args.n_boot <= 0:
        parser.error("--n-boot must be positive")
    for path in (args.raw, args.clean):
        if not path.is_file():
            parser.error("Input file not found: {}".format(path))
    result = analyze(args.eval_dir, args.raw, args.clean,
                     args.experiment, args.n_boot, args.boot_seed)
    write_outputs(result, args.out)
    print(args.out / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
