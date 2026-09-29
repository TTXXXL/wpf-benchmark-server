"""Strict JSON and Markdown reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..paths import ProjectPaths


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_result(R: Dict[str, Any], tag: str, out_dir: Optional[Path] = None,
                timestamp: Optional[str] = None) -> Path:
    """Write strict JSON, including the full reproducibility configuration."""
    out_dir = Path(out_dir) if out_dir is not None else ProjectPaths.resolve().evaluation
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
    path = out_dir / "{}_{}.json".format(
        safe_tag, timestamp or pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    path.write_text(json.dumps(_json_safe(R), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")
    return path


def _fmt(value: Any, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return "{:.{}f}".format(value, decimals)


def result_to_markdown(R: Dict[str, Any]) -> str:
    """Render all deterministic layers in a compact human-readable report."""
    lines = ["# 评估结果：{}（{} 表，{:,} 样本）".format(
        R["model"], R["table"], R["n_samples"]), "", "## A 精度主表", "",
        "| 层级 | MAE (kW) | RMSE (kW) | NMAE (%) | SS vs 持续性 (%) | 样本数 |",
        "|---|---:|---:|---:|---:|---:|"]
    for label, key in (("单机", "A_turbine"), ("场站", "A_farm")):
        row = R[key]
        lines.append("| {} | {} | {} | {} | {} | {:,} |".format(
            label, _fmt(row["MAE_kW"]), _fmt(row["RMSE_kW"]),
            _fmt(row["NMAE_pct"]), _fmt(row.get("SS_vs_persistence_pct")),
            row.get("n", R["n_samples"])))
    lines += ["", "## B 误差结构", "",
              "| 预测步 h | MAE (kW) | RMSE (kW) | n |",
              "|---:|---:|---:|---:|"]
    for h, row in R["B_per_horizon"].items():
        lines.append("| {} | {} | {} | {:,} |".format(
            h, _fmt(row["MAE_kW"]), _fmt(row["RMSE_kW"]), row["n"]))
    lines += ["", "| 实测风速 (m/s) | MAE (kW) | RMSE (kW) | n |",
              "|---|---:|---:|---:|"]
    for label, row in R["B_wind_bins"].items():
        lines.append("| {} | {} | {} | {:,} |".format(
            label, _fmt(row["MAE_kW"]), _fmt(row["RMSE_kW"]), row["n"]))
    bias = R["B_bias"]
    lines += ["", "偏差：ME {} kW；高估 {}%；低估 {}%。".format(
        _fmt(bias["ME_kW"]), _fmt(100 * bias["over_frac"]),
        _fmt(100 * bias["under_frac"])), "", "| 测试日分块 | MAE (kW) | n |",
        "|---|---:|---:|"]
    for row in R["B_stability"]:
        lines.append("| {} | {} | {:,} |".format(row["days"], _fmt(row["MAE_kW"]), row["n"]))
    ramp, energy, high = R["C_ramp"], R["C_daily_energy"], R["C_high_wind"]
    lines += ["", "## C 工业工况", "",
              "- 爬坡逐格判别（发布时刻×机组×时距；阈值 {} kW）：真实阳性 {:,} 格；预测阳性 {:,} 格；recall {}；precision {}；F1 {}；真实阳性格 MAE {} kW。".format(
                  _fmt(ramp["threshold_kW"]), ramp["n_positive_cells"],
                  ramp["n_pred_positive_cells"],
                  _fmt(ramp["recall"], 3), _fmt(ramp["precision"], 3),
                  _fmt(ramp["f1"], 3), _fmt(ramp["positive_cell_MAE_kW"])),
              "- 日电量（h=1，各目标时刻只计一次）：{} 天，MAE {} MWh，MAPE {}%。".format(
                  energy["n_days"], _fmt(energy["MAE_MWh"]), _fmt(energy["MAPE_pct"])),
              "- 大风段（实测 ≥{} m/s）：{:,} 样本。".format(
                  R["config"]["high_wind_speed"], high["n"])]
    if high["n"] > 100:
        lines.append("  MAE {} kW；RMSE {} kW。".format(
            _fmt(high["MAE_kW"]), _fmt(high["RMSE_kW"])))
    else:
        lines.append("  样本不足，未报告误差。")
    if energy["per_day"]:
        first, last = energy["per_day"][0], energy["per_day"][-1]
        steps = R["config"]["steps_per_day"]
        lines.append("  边界日只统计有预测的时段：首日 {}/{} 步，末日 {}/{} 步。".format(
            first["n_steps"], steps, last["n_steps"], steps))
    physics = R["D_physics"]
    lines += ["", "## D 物理合理性", "",
              "- 曲线违背率：预测 {}%；真实值参照 {}%；曲线覆盖率 {}%。".format(
                  _fmt(physics["violation_rate_pct"]),
                  _fmt(physics["truth_violation_rate_pct"]),
                  _fmt(physics["curve_coverage_pct"])),
              "- 负功率率 {}%；超额定率 {}%。".format(
                  _fmt(physics["neg_rate_pct"]), _fmt(physics["over_rated_rate_pct"])),
              "", "## E 概率指标", "", "留待论文②实现。", ""]
    return "\n".join(lines)
