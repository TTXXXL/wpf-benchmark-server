# -*- coding: utf-8 -*-
"""A 方案：按 SDWPF 官方 unknown/abnormal 目标掩码重评已保存的预测（不训练）。

输入（均为已有文件，不改动）：
  reports/eval/index.jsonl、<run>_main_*.json、<run_id>_arrays.npz（wave1 已产出）
  data/raw/sdwpf/sdwpf_245days_v1.csv（取官方规则所需的原始列）

官方规则（Zhou 等 2024, Sci. Data 11:649，"Unknown values"/"Abnormal values"）：
  unknown : Patv<=0 且 Wspd>2.5；或 Pab1/Pab2/Pab3 任一 >89°
  abnormal: |Ndir|>720° 或 |Wdir|>180°
  这些目标时刻不参与评分。

三种评分掩码（都作用于目标时刻 t+h）：
  M0_repo           仓库主表掩码（npz 的 valid），真值 = npz truth；用于复现 JSON 数字
  M1_repo_official  M0 ∧ 非官方 unknown/abnormal；真值 = npz truth
  M2_official       原始 Patv、Wspd 有限 ∧ 非官方 unknown/abnormal，不用仓库 flag；
                    真值 = max(原始 Patv, 0)（与清洗 P1 规则、模型输出 >=0 一致）
分层只用发布时刻 t 的信息：步长 h、发布时刻运行状态（原始数据）、发布时刻功率档。
区间：按发布日重抽的配对 bootstrap，多种子模型再对种子重抽（复用
scripts/wave1_paired_bootstrap.py 的 select_runs/_read_run/compare）。

用法（在项目根目录）：
  python scripts/official_mask_rescore.py --experiment wave1 --out reports/official_mask
依赖：numpy、pandas。Python 3.8 可运行。

ClawsGO Science Agent · 2026-09-29
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# wave1_paired_bootstrap.py 可与本脚本同目录，或位于当前目录的 scripts/ 下
for _candidate in (Path(__file__).resolve().parent, Path.cwd() / "scripts"):
    if (_candidate / "wave1_paired_bootstrap.py").is_file():
        sys.path.insert(0, str(_candidate))
        break
import wave1_paired_bootstrap as wb  # noqa: E402

NS_PER_DAY = 86_400_000_000_000
MASK_NAMES = ("M0_repo", "M1_repo_official", "M2_official")
ISSUE_STATES = ("normal", "pitch_gt89", "zero_power_windy", "missing")
POWER_BINS = ("low", "mid", "high", "missing")
RAW_COLUMNS = ["TurbID", "Day", "Tmstamp", "Wspd", "Wdir", "Ndir",
               "Pab1", "Pab2", "Pab3", "Patv"]
TRUTH_TOLERANCE_KW = 0.01      # npz truth 为 float32，与原始值的舍入差远小于此
MAX_TRUTH_MISMATCH_FRAC = 0.01  # 超过即判定为网格未对齐，终止


# ───────────────────────────── 原始数据与官方规则 ─────────────────────────────

def load_raw_grid(raw_path: Path, times_ns: np.ndarray, turbine_ids: np.ndarray
                  ) -> Dict[str, np.ndarray]:
    """把原始 CSV 对齐到 npz 的 (时间, 机组) 网格，返回各列 (T, N) float64 数组。"""
    first_day = int(times_ns.min() // NS_PER_DAY) + 1
    last_day = int(times_ns.max() // NS_PER_DAY) + 1
    raw = pd.read_csv(raw_path, usecols=RAW_COLUMNS)
    raw = raw[(raw["Day"] >= first_day) & (raw["Day"] <= last_day)].copy()
    # 与 preprocessing/sdwpf.py:53 相同的时间戳构造
    ts = (pd.to_timedelta(raw["Day"] - 1, unit="D") +
          pd.to_timedelta(raw["Tmstamp"] + ":00"))
    raw["ts_ns"] = ts.to_numpy(dtype="timedelta64[ns]").astype(np.int64)
    if raw.duplicated(["ts_ns", "TurbID"]).any():
        raise ValueError("原始数据存在重复的 (时间, 机组) 行")
    t_idx = pd.Index(times_ns).get_indexer(raw["ts_ns"].to_numpy())
    n_idx = pd.Index(turbine_ids.astype(np.int64)).get_indexer(
        raw["TurbID"].to_numpy(dtype=np.int64))
    keep = (t_idx >= 0) & (n_idx >= 0)
    T, N = len(times_ns), len(turbine_ids)
    filled = np.zeros((T, N), dtype=bool)
    filled[t_idx[keep], n_idx[keep]] = True
    if not filled.all():
        raise ValueError("原始数据未覆盖 npz 网格：缺 {:,} 个 (时间, 机组) 格".format(
            int((~filled).sum())))
    grid = {}
    for col in ("Wspd", "Wdir", "Ndir", "Pab1", "Pab2", "Pab3", "Patv"):
        out = np.full((T, N), np.nan, dtype=np.float64)
        out[t_idx[keep], n_idx[keep]] = raw[col].to_numpy(dtype=np.float64)[keep]
        grid[col] = out
    return grid


def official_flags(grid: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """逐格官方判据；NaN 比较为 False，缺失另由 finite 掩码处理。"""
    with np.errstate(invalid="ignore"):
        pitch = ((grid["Pab1"] > 89) | (grid["Pab2"] > 89) | (grid["Pab3"] > 89))
        zero_windy = (grid["Patv"] <= 0) & (grid["Wspd"] > 2.5)
        abnormal = (np.abs(grid["Ndir"]) > 720) | (np.abs(grid["Wdir"]) > 180)
    finite = np.isfinite(grid["Patv"]) & np.isfinite(grid["Wspd"])
    return {"pitch": pitch, "zero_windy": zero_windy, "abnormal": abnormal,
            "bad": pitch | zero_windy | abnormal, "finite": finite}


def target_stack(base: np.ndarray, t_eff: int, horizon: int) -> np.ndarray:
    """(T, N) → (T_eff, N, H)，第 h 步取 t+h；与 evaluator._valid_stack 相同。"""
    return np.stack([base[h:t_eff + h] for h in range(1, horizon + 1)], axis=2)


# ───────────────────────────── 掩码与分层 ─────────────────────────────

def build_masks(reference: dict, flags: Dict[str, np.ndarray], raw_patv: np.ndarray,
                truth_raw: np.ndarray, warnings: List[str]) -> Tuple[dict, dict, dict]:
    t_eff, n, horizon = reference["shape"]
    valid = reference["valid"]
    bad_t = target_stack(flags["bad"], t_eff, horizon)
    finite_t = target_stack(flags["finite"], t_eff, horizon)
    masks = {"M0_repo": valid,
             "M1_repo_official": valid & ~bad_t,
             "M2_official": finite_t & ~bad_t}

    # 一致性检查：M0 有效格上 npz 真值应等于 max(原始 Patv, 0)
    truth_npz = reference["truth"]
    diff = np.abs(truth_npz[valid].astype(np.float64) - truth_raw[valid])
    mismatch = int((~(diff <= TRUTH_TOLERANCE_KW)).sum())
    frac = mismatch / max(int(valid.sum()), 1)
    checks = {"M0_cells": int(valid.sum()),
              "truth_mismatch_cells": mismatch,
              "truth_mismatch_frac": frac,
              "truth_max_abs_diff_kW": float(np.nanmax(diff)) if diff.size else 0.0}
    if frac > MAX_TRUTH_MISMATCH_FRAC:
        raise ValueError("M0 格上 npz 真值与原始 Patv 不一致 {:,} 格（{:.2%}），"
                         "网格可能未对齐，终止".format(mismatch, frac))
    if mismatch:
        warnings.append("M0 格上 npz 真值与 max(原始 Patv,0) 差 >{} kW 的有 {:,} 格（{:.4%}）".format(
            TRUTH_TOLERANCE_KW, mismatch, frac))

    # 发布时刻（t）运行状态：只用 t 时刻原始观测
    state = np.full((t_eff, n), 0, dtype=np.int8)             # normal
    zero_windy_only = flags["zero_windy"] & ~flags["pitch"]
    state[zero_windy_only[:t_eff]] = 2
    state[flags["pitch"][:t_eff]] = 1
    state[~flags["finite"][:t_eff]] = 3                       # missing 优先
    ratio = reference["last_power"] / reference["rated"]
    power_bin = np.where(np.isnan(ratio), 3,
                         np.where(ratio < 0.05, 0, np.where(ratio < 0.90, 1, 2))).astype(np.int8)

    raw_neg = masks["M2_official"] & (target_stack(np.nan_to_num(
        raw_patv, nan=0.0), t_eff, horizon) < 0)
    composition = {}
    for name, m in masks.items():
        total = int(m.sum())
        composition[name] = {
            "n_cells": total,
            "frac_of_M0": total / max(int(valid.sum()), 1),
            "issue_state": {s: int((m & (state[:, :, None] == i)).sum())
                            for i, s in enumerate(ISSUE_STATES)},
            "issue_power": {s: int((m & (power_bin[:, :, None] == i)).sum())
                            for i, s in enumerate(POWER_BINS)}}
    composition["M2_official"]["raw_negative_truth_cells_clipped_to_0"] = int(raw_neg.sum())
    removed = valid & bad_t
    checks["M0_cells_removed_by_official"] = {
        "total": int(removed.sum()),
        "pitch_gt89": int((valid & target_stack(flags["pitch"], t_eff, horizon)).sum()),
        "zero_power_windy": int((valid & target_stack(flags["zero_windy"], t_eff, horizon)).sum()),
        "abnormal_dir": int((valid & target_stack(flags["abnormal"], t_eff, horizon)).sum())}
    return masks, {"state": state, "power_bin": power_bin}, dict(checks, composition=composition)


def labels_for(horizon: int) -> List[dict]:
    labels = []
    for mask in MASK_NAMES:
        labels.append({"mask": mask, "level": mask + "|all", "stratum": "all"})
        for h in range(1, horizon + 1):
            labels.append({"mask": mask, "level": mask + "|h", "stratum": str(h)})
        for s in ISSUE_STATES:
            labels.append({"mask": mask, "level": mask + "|issue_state", "stratum": s})
        for s in POWER_BINS:
            labels.append({"mask": mask, "level": mask + "|issue_power", "stratum": s})
    return labels


def daily_stats(forecasts: np.ndarray, truth_npz: np.ndarray, truth_raw: np.ndarray,
                masks: dict, strata: dict, day_index: np.ndarray, n_days: int,
                labels: List[dict]) -> np.ndarray:
    """返回 D × S × (n, 绝对误差和, 平方误差和)。"""
    t_eff, n, horizon = forecasts.shape
    out = np.zeros((n_days, len(labels), 3), dtype=np.float64)
    day_cells = np.broadcast_to(day_index[:, None, None], forecasts.shape)
    errors = {}
    for truth_key, truth in (("npz", truth_npz), ("raw", truth_raw)):
        e = forecasts.astype(np.float64) - truth
        errors[truth_key] = (np.abs(e), np.square(e))
    h_index = np.broadcast_to(np.arange(1, horizon + 1)[None, None, :], forecasts.shape)
    state = np.broadcast_to(strata["state"][:, :, None], forecasts.shape)
    power_bin = np.broadcast_to(strata["power_bin"][:, :, None], forecasts.shape)
    for s, label in enumerate(labels):
        m = masks[label["mask"]]
        kind = label["level"].split("|", 1)[1]
        if kind == "h":
            sel = m & (h_index == int(label["stratum"]))
        elif kind == "issue_state":
            sel = m & (state == ISSUE_STATES.index(label["stratum"]))
        elif kind == "issue_power":
            sel = m & (power_bin == POWER_BINS.index(label["stratum"]))
        else:
            sel = m
        absolute, squared = errors["raw" if label["mask"] == "M2_official" else "npz"]
        d = day_cells[sel]
        out[:, s, 0] = np.bincount(d, minlength=n_days)
        out[:, s, 1] = np.bincount(d, weights=absolute[sel], minlength=n_days)
        out[:, s, 2] = np.bincount(d, weights=squared[sel], minlength=n_days)
    return out


# ───────────────────────────── 汇总 ─────────────────────────────

def per_model_scores(runs: List[dict], stats: np.ndarray, labels: List[dict]) -> dict:
    scores = wb._scores(stats.sum(axis=1))          # R × S × 2
    output: Dict[str, dict] = {}
    for model in dict.fromkeys(run["model"] for run in runs):
        idx = [i for i, run in enumerate(runs) if run["model"] == model]
        rows = {}
        for s, label in enumerate(labels):
            mae = scores[idx, s, 0]
            rmse = scores[idx, s, 1]
            rows[label["level"] + ":" + label["stratum"]] = {
                "MAE_kW": float(np.mean(mae)) if np.isfinite(mae).all() else None,
                "RMSE_kW": float(np.mean(rmse)) if np.isfinite(rmse).all() else None,
                "MAE_seed_std_kW": float(np.std(mae, ddof=1)) if len(idx) > 1 and np.isfinite(mae).all() else None,
                "n_cells": int(stats[idx[0], :, s, 0].sum()),
                "n_seeds": len(idx)}
        output[model] = rows
    return output


def _num(value: Optional[float], digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return "{:.{}f}".format(value, digits)


def make_markdown(summary: dict) -> str:
    meta, scores = summary["meta"], summary["scores"]
    comps = {(c["id"], c["level"], c["stratum"], c["metric"]): c
             for c in summary["comparisons"]}
    models = meta["models"]
    horizon = meta["horizon"]
    checks = meta["checks"]
    L = ["# 官方 unknown/abnormal 掩码重评（A 方案）", "",
         "生成：ClawsGO Science Agent · official_mask_rescore.py", "",
         "experiment: {}；n_boot: {}；boot_seed: {}；发布日 {} 天（Day {}–{}）".format(
             meta["experiment"], meta["n_boot"], meta["boot_seed"], len(meta["issue_days"]),
             meta["issue_days"][0], meta["issue_days"][-1]), "",
         "## 掩码定义", "",
         "- M0_repo：仓库主表掩码，真值取 npz（应复现 JSON 的 MAE/RMSE）",
         "- M1_repo_official：M0 ∧ 目标时刻非官方 unknown/abnormal，真值取 npz",
         "- M2_official：目标时刻原始 Patv、Wspd 有限 ∧ 非官方 unknown/abnormal，不用仓库 flag，真值 = max(原始 Patv, 0)",
         "", "## 自检", "",
         "- M0 格数 {:,}；npz 真值与 max(原始 Patv,0) 不一致 {:,} 格（{:.4%}），最大差 {} kW".format(
             checks["M0_cells"], checks["truth_mismatch_cells"], checks["truth_mismatch_frac"],
             _num(checks["truth_max_abs_diff_kW"], 4)),
         "- M0 中被官方规则剔除：合计 {total:,}（Pab>89° {pitch_gt89:,}；Patv≤0 且 Wspd>2.5 {zero_power_windy:,}；方向异常 {abnormal_dir:,}；三类可重叠）".format(
             **checks["M0_cells_removed_by_official"])]
    pers = scores.get("persistence", {})
    for mask in ("M0_repo", "M1_repo_official"):
        row = pers.get(mask + "|all:all", {})
        L.append("- persistence {}：MAE {} / RMSE {}".format(
            mask, _num(row.get("MAE_kW")), _num(row.get("RMSE_kW"))))
    L += ["", "警告：", ""] + ["- " + w for w in meta["warnings"]] + (
        ["- 无"] if not meta["warnings"] else [])

    L += ["", "## 各掩码的样本构成（按发布时刻状态）", "",
          "| 掩码 | 评分格 | 占 M0 | normal | pitch_gt89 | zero_power_windy | missing | 发布功率 low | mid | high |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mask in MASK_NAMES:
        c = checks["composition"][mask]
        L.append("| {} | {:,} | {:.1%} | {} | {} |".format(
            mask, c["n_cells"], c["frac_of_M0"],
            " | ".join("{:,}".format(c["issue_state"][s]) for s in ISSUE_STATES),
            " | ".join("{:,}".format(c["issue_power"][s]) for s in ("low", "mid", "high"))))
    L.append("")
    L.append("M2 中原始 Patv<0 被截为 0 的目标格：{:,}".format(
        checks["composition"]["M2_official"]["raw_negative_truth_cells_clipped_to_0"]))

    L += ["", "## 总体 MAE / RMSE（多种子取均值）", "",
          "| 模型 | 种子 | " + " | ".join("{} MAE / RMSE".format(m) for m in MASK_NAMES) + " |",
          "| --- | ---: |" + " ---: |" * len(MASK_NAMES)]
    for model in models:
        cells = []
        for mask in MASK_NAMES:
            row = scores[model][mask + "|all:all"]
            cells.append("{} / {}".format(_num(row["MAE_kW"]), _num(row["RMSE_kW"])))
        L.append("| {} | {} | {} |".format(model, scores[model][MASK_NAMES[0] + "|all:all"]["n_seeds"],
                                           " | ".join(cells)))

    L += ["", "## 与持续性的配对差（Δ = 模型 − persistence，MAE，kW）", "",
          "CI(b)：发布日块 + 种子重抽；CI(a)：仅发布日块。单种子模型两者相同。", "",
          "| 比较 | 掩码 | Δ MAE | 相对 | CI(b) | CI(a) | Δ RMSE | CI(b) RMSE |",
          "| --- | --- | ---: | ---: | --- | --- | ---: | --- |"]
    for plan in meta["comparison_plan"]:
        for mask in MASK_NAMES:
            key = (plan["id"], mask + "|all", "all")
            mae, rmse = comps.get(key + ("MAE_kW",)), comps.get(key + ("RMSE_kW",))
            if mae is None:
                continue
            L.append("| {} {}−{} | {} | {} | {} | {} | {} | {} | {} |".format(
                plan["id"], plan["A"], plan["B"], mask, _num(mae["delta"]),
                "{:.1%}".format(mae["rel"]) if mae["rel"] is not None else "—",
                wb._ci(mae["ci_data_seed"]), wb._ci(mae["ci_data"]),
                _num(rmse["delta"]), wb._ci(rmse["ci_data_seed"])))

    for mask in MASK_NAMES:
        L += ["", "## 逐步长 MAE：{}".format(mask), "",
              "| h | " + " | ".join(models) + " |", "| ---: |" + " ---: |" * len(models)]
        for h in range(1, horizon + 1):
            L.append("| {} | {} |".format(h, " | ".join(
                _num(scores[m][mask + "|h:" + str(h)]["MAE_kW"]) for m in models)))

    for mask in MASK_NAMES:
        L += ["", "## 按发布时刻运行状态的 MAE：{}".format(mask), "",
              "| 状态 | n | " + " | ".join(models) + " |",
              "| --- | ---: |" + " ---: |" * len(models)]
        for s in ISSUE_STATES:
            n = scores[models[0]][mask + "|issue_state:" + s]["n_cells"]
            L.append("| {} | {:,} | {} |".format(s, n, " | ".join(
                _num(scores[m][mask + "|issue_state:" + s]["MAE_kW"]) for m in models)))
        L += ["", "| 发布功率档 | n | " + " | ".join(models) + " |",
              "| --- | ---: |" + " ---: |" * len(models)]
        for s in POWER_BINS:
            n = scores[models[0]][mask + "|issue_power:" + s]["n_cells"]
            L.append("| {} | {:,} | {} |".format(s, n, " | ".join(
                _num(scores[m][mask + "|issue_power:" + s]["MAE_kW"]) for m in models)))

    L += ["", "注：这里重算的是本仓库的逐机 MAE/RMSE，并非 KDD Cup 赛事评分；M2 还要求原始 Patv、Wspd 均有限。",
          "分层只用发布时刻信息；分层与逐步长结果未做多重比较校正，仅作诊断。",
          "运行状态由原始数据判定：pitch_gt89 = 任一桨距角 >89°；zero_power_windy = 非 pitch_gt89 且 Patv≤0、Wspd>2.5；missing = 原始 Patv 或 Wspd 缺失。", ""]
    return "\n".join(L)


# ───────────────────────────── 主流程 ─────────────────────────────

def analyze(eval_dir: Path, raw_path: Path, experiment: str, n_boot: int,
            boot_seed: int) -> dict:
    selected, warnings = wb.select_runs(eval_dir, experiment)
    if not selected:
        raise ValueError("index.jsonl 中没有 experiment={} 的 main 运行".format(experiment))
    models = list(dict.fromkeys(row["model"] for row in selected))
    if "persistence" not in models:
        raise ValueError("缺少 persistence 运行，无法做与持续性的配对比较")
    models = ["persistence"] + [m for m in models if m != "persistence"]

    reference = None
    runs: List[dict] = []
    stats_list = []
    for row in selected:
        source_path = eval_dir / Path(row["json_path"].replace("\\", "/")).name
        source_json = json.loads(source_path.read_text(encoding="utf-8"))
        if source_json.get("target_mask") or source_json.get("config", {}).get("target_mask"):
            raise ValueError("This historical diagnostic requires legacy M0 arrays; "
                             "new M1/M2 runs already store their selected mask")
        arrays, info, _ = wb._read_run(eval_dir, row, reference)   # 含 MAE/RMSE 复算与网格对齐检查
        if reference is None:
            shape = arrays["forecasts"].shape
            rated = float(json.loads((eval_dir / info["json_file"]).read_text(
                encoding="utf-8"))["config"]["rated_power_kw"])
            reference = {"shape": shape, "rated": rated,
                         "times": arrays["times"], "turbine_ids": arrays["turbine_ids"],
                         "valid": arrays["valid"], "truth": arrays["truth"],
                         "truth_valid": arrays["truth"][arrays["valid"]],
                         "last_power": arrays["last_power"]}
            t_eff, _, horizon = shape
            grid = load_raw_grid(raw_path, np.asarray(arrays["times"], dtype=np.int64),
                                 np.asarray(arrays["turbine_ids"]))
            flags = official_flags(grid)
            truth_raw = target_stack(np.maximum(grid["Patv"], 0.0), t_eff, horizon)
            masks, strata, checks = build_masks(reference, flags, grid["Patv"],
                                                truth_raw, warnings)
            issue_days = np.floor_divide(np.asarray(arrays["times"][:t_eff], dtype=np.int64),
                                         NS_PER_DAY) + 1
            days, day_index = np.unique(issue_days, return_inverse=True)
            labels = labels_for(horizon)
        stats_list.append(daily_stats(arrays["forecasts"], reference["truth"], truth_raw,
                                      masks, strata, day_index, len(days), labels))
        runs.append(info)
    stats = np.stack(stats_list)

    # M0 复现检查：按掩码 M0 重算的总体 MAE 必须与 JSON 一致
    all0 = labels.index(next(l for l in labels if l["level"] == "M0_repo|all"))
    for i, info in enumerate(runs):
        mae = stats[i, :, all0, 1].sum() / stats[i, :, all0, 0].sum()
        if abs(mae - info["json_MAE_kW"]) > 1e-3:
            raise ValueError("{}: M0 重算 MAE {:.4f} ≠ JSON {:.4f}".format(
                info["run_id"], mae, info["json_MAE_kW"]))

    comparison_defs = [("R{}".format(i), m, "persistence")
                       for i, m in enumerate(models[1:], 1)]
    if "ours_no_wake" in models and "barest" in models:
        comparison_defs.append(("P1", "ours_no_wake", "barest"))
    comparisons = wb.compare(runs, stats, labels, n_boot, boot_seed, warnings,
                             comparison_defs)
    return {"meta": {"experiment": experiment, "n_boot": n_boot, "boot_seed": boot_seed,
                     "models": models, "horizon": int(reference["shape"][2]),
                     "comparison_plan": [{"id": c, "A": a, "B": b} for c, a, b in comparison_defs],
                     "runs": [{k: r[k] for k in ("model", "seed", "run_id", "arrays_file")} for r in runs],
                     "issue_days": [int(d) for d in days],
                     "raw_csv": str(raw_path), "warnings": warnings, "checks": checks,
                     "official_rule": "unknown: Patv<=0 & Wspd>2.5, or any Pab>89; "
                                      "abnormal: |Ndir|>720 or |Wdir|>180 (target time)"},
            "scores": per_model_scores(runs, stats, labels),
            "comparisons": comparisons}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", type=Path, default=Path("reports/eval"))
    parser.add_argument("--raw", type=Path,
                        default=Path("data/raw/sdwpf/sdwpf_245days_v1.csv"))
    parser.add_argument("--experiment", default="wave1")
    parser.add_argument("--out", type=Path, default=Path("reports/official_mask"))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--boot-seed", type=int, default=20260929)
    parser.add_argument("--output-prefix", default="official_mask_rescore")
    args = parser.parse_args(argv)
    try:
        if args.n_boot <= 0:
            raise ValueError("--n-boot 必须为正")
        if not args.raw.is_file():
            raise ValueError("找不到原始 CSV：{}".format(args.raw))
        summary = analyze(args.eval_dir, args.raw, args.experiment, args.n_boot, args.boot_seed)
        args.out.mkdir(parents=True, exist_ok=True)
        json_path = args.out / (args.output_prefix + ".json")
        md_path = args.out / (args.output_prefix + ".md")
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
    raise SystemExit(main())
