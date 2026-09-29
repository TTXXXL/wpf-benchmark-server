"""把 persistence 的 main/all 评估 JSON 转成易读的中文 Markdown。

默认读取 reports/eval 中最新的一对文件；也可以用 --main 和 --all 指定文件。
仅使用 Python 标准库，可在 Python 3.8 及以上运行。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = ROOT / "reports" / "eval"


def latest_pair(directory: Path) -> Tuple[Path, Path]:
    """按文件修改时间选择同一次运行的一对文件，兼容本地和 UTC 文件名。"""
    def dated_files(table: str) -> List[Tuple[float, Path]]:
        return [(path.stat().st_mtime, path)
                for path in directory.glob("persistence_{}_*.json".format(table))]

    mains, alls = dated_files("main"), dated_files("all")
    if not mains or not alls:
        raise ValueError("未找到 persistence_main_*.json 和 persistence_all_*.json；请先运行基线，或同时传入 --main、--all。")
    for main_time, main_path in sorted(mains, key=lambda item: item[0], reverse=True):
        all_time, all_path = min(alls, key=lambda item: abs(item[0] - main_time))
        if abs(all_time - main_time) <= 120:
            return main_path, all_path
    raise ValueError("找到了 JSON，但没有时间相近的一对 main/all；请同时传入 --main、--all。")


def read_result(path: Path, table: str) -> Dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("无法读取 {}：{}".format(path, exc)) from exc
    if not isinstance(result, dict) or result.get("model") != "persistence" or result.get("table") != table:
        raise ValueError("{} 不是 persistence 的 {} 表结果".format(path, table))
    return result


def fmt(value: Any, digits: int = 2) -> str:
    return "—" if value is None else "{:,.{}f}".format(value, digits)


def percent(value: Any, digits: int = 2) -> str:
    return "—" if value is None else "{}%".format(fmt(value, digits))


def make_report(main: Dict[str, Any], all_rows: Dict[str, Any],
                main_path: Path, all_path: Path) -> str:
    if main.get("config") != all_rows.get("config"):
        raise ValueError("main 和 all 的 config 不同，不能当成同一次评估来比较")
    cfg = main["config"]
    extra = all_rows["n_samples"] - main["n_samples"]
    if extra < 0:
        raise ValueError("all 的样本数小于 main，请检查文件是否配对")
    main_mae = main["A_turbine"]["MAE_kW"]
    all_mae = all_rows["A_turbine"]["MAE_kW"]
    horizon = main["B_per_horizon"]
    first_step, last_step = horizon["1"], horizon[str(cfg["horizon"])]
    worst_wind = max(main["B_wind_bins"].items(),
                     key=lambda item: item[1]["MAE_kW"] if item[1]["n"] else float("-inf"))
    worst_days = max(main["B_stability"], key=lambda row: row["MAE_kW"])
    over = main["B_bias"]["over_frac"] * 100
    under = main["B_bias"]["under_frac"] * 100
    tie = 100 - over - under
    worst_energy_day = max(main["C_daily_energy"]["per_day"],
                           key=lambda row: row["APE_pct"] if row["APE_pct"] is not None else -1)
    lines = [
        "# 持续性基线评估结果解读",
        "",
        "数据来源：`{}`（主表）和 `{}`（附表）。".format(main_path.name, all_path.name),
        "本报告由 `scripts/analyze_persistence.py` 从这两个 JSON 生成。",
        "",
        "## 先看结论",
        "",
        "- 主表单机平均绝对误差（MAE）为 **{} kW**；附表为 **{} kW**。这是后续模型最直接的比较起点。".format(
            fmt(main_mae), fmt(all_mae)),
        "- 预测越远，误差越大：第 1 步（10 分钟后）为 {} kW，第 {} 步（{} 分钟后）为 {} kW。".format(
            fmt(first_step["MAE_kW"]), cfg["horizon"], cfg["horizon"] * 10, fmt(last_step["MAE_kW"])),
        "- 主表误差最高的风速段是 {} m/s，MAE 为 {} kW；测试末段 Day {} 的 MAE 为 {} kW。".format(
            worst_wind[0], fmt(worst_wind[1]["MAE_kW"]), worst_days["days"], fmt(worst_days["MAE_kW"])),
        "- 这个模型把当前功率原样复制到未来，因此不会预测出功率突变；真实突变事件的召回率为 {}。".format(
            percent(main["C_ramp"]["recall"] * 100)),
        "",
        "## 这两个 JSON 分别是什么",
        "",
        "模型使用最近观测的功率，重复作为未来 {} 个 10 分钟时间点的预测。".format(cfg["horizon"]),
        "`main` 和 `all` 使用同一批预测，区别是哪些真实值参与打分：",
        "",
        "- **主表（main）**：排除缺失、补值、异常、故障、限电和全场停机等标记。后续比较模型时优先看这张表。",
        "- **附表（all）**：只要求真实功率有数值，因此也会纳入部分带标记的数据。它用于观察数据范围扩大后的结果。",
        "",
        "| 指标 | 主表 main | 附表 all | 怎么理解 |",
        "| --- | ---: | ---: | --- |",
        "| 参与计分的样本 | {:,} | {:,} | 每个机组、每个预测时间点各算一个样本 |".format(
            main["n_samples"], all_rows["n_samples"]),
        "| 单机 MAE | {} kW | {} kW | 平均差多少千瓦；越小越好 |".format(fmt(main_mae), fmt(all_mae)),
        "| 单机 RMSE | {} kW | {} kW | 对大误差更敏感；越小越好 |".format(
            fmt(main["A_turbine"]["RMSE_kW"]), fmt(all_rows["A_turbine"]["RMSE_kW"])),
        "| 单机 NMAE | {} | {} | MAE 除以单机额定功率 {} kW |".format(
            percent(main["A_turbine"]["NMAE_pct"]), percent(all_rows["A_turbine"]["NMAE_pct"]),
            fmt(cfg["rated_power_kw"], 0)),
        "| 场站 MAE | {} kW | {} kW | 先把同一时刻各机组的误差相加，再取绝对值和平均 |".format(
            fmt(main["A_farm"]["MAE_kW"]), fmt(all_rows["A_farm"]["MAE_kW"])),
        "",
        "附表多计入 **{:,}** 个样本（比主表多 {}）；其单机 MAE 高 {} kW。".format(
            extra, percent(100 * extra / main["n_samples"]), fmt(all_mae - main_mae)),
        "这是样本范围变化后的结果，不代表换了模型或训练方法。",
        "",
        "**注意场站 NMAE：**当前评估代码仍用单机额定功率 {} kW 作分母，".format(fmt(cfg["rated_power_kw"], 0))
        + "但场站 MAE 是多台机组误差之和。所以 JSON 中场站 NMAE 高于 100% 不表示场站预测误差超过总装机容量；"
        + "这里建议直接看场站 MAE（kW），不要把该 NMAE 当作通常意义的场站百分比误差。",
        "",
        "`SS_vs_persistence_pct` 为 0% 是正常的：这里评估的模型本身就是持续性基线，它和自己比较。",
        "",
        "## 误差主要出现在哪里",
        "",
        "### 预测时间越远，误差越大",
        "",
        "| 预测步 | 距当前 | 主表 MAE (kW) | 附表 MAE (kW) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for step in range(1, cfg["horizon"] + 1):
        key = str(step)
        lines.append("| {} | {} 分钟 | {} | {} |".format(
            step, step * 10, fmt(main["B_per_horizon"][key]["MAE_kW"]),
            fmt(all_rows["B_per_horizon"][key]["MAE_kW"])))
    lines += [
        "",
        "### 按实际风速看",
        "",
        "| 风速范围 (m/s) | 主表样本 | 主表 MAE (kW) | 附表 MAE (kW) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, row in main["B_wind_bins"].items():
        lines.append("| {} | {:,} | {} | {} |".format(
            label, row["n"], fmt(row["MAE_kW"]),
            fmt(all_rows["B_wind_bins"][label]["MAE_kW"])))
    lines += [
        "",
        "风速段的样本数量差别很大。样本很少的区间不能只凭低 MAE 判断模型表现好。",
        "",
        "### 按测试日期看",
        "",
        "| 测试日 | 主表 MAE (kW) | 附表 MAE (kW) |",
        "| --- | ---: | ---: |",
    ]
    all_blocks = {row["days"]: row for row in all_rows["B_stability"]}
    for row in main["B_stability"]:
        lines.append("| Day {} | {} | {} |".format(
            row["days"], fmt(row["MAE_kW"]), fmt(all_blocks[row["days"]]["MAE_kW"])))
    lines += [
        "",
        "`B_bias.ME_kW` 是有正负号的平均误差；主表为 {} kW。".format(fmt(main["B_bias"]["ME_kW"])),
        "其中高估占 {}、低估占 {}、恰好相等约占 {}。".format(
            percent(over), percent(under), percent(tie)),
        "正负误差会互相抵消，所以 ME 接近零不代表 MAE 也接近零。",
        "",
        "## 其他指标怎么读",
        "",
        "- **爬坡逐格判别（`C_ramp`）**：每个发布时刻×机组×时距格的预测和当前功率相差至少 {} kW，记为预测阳性。".format(
            fmt(main["C_ramp"]["threshold_kW"])),
        "  主表真实阳性 {:,} 格，附表 {:,} 格；模型预测阳性 {:,} 格，主表召回率 {}。".format(
            main["C_ramp"]["n_positive_cells"], all_rows["C_ramp"]["n_positive_cells"],
            main["C_ramp"]["n_pred_positive_cells"],
            percent(main["C_ramp"]["recall"] * 100)),
        "  `precision: null` 表示模型未报阳性格，精确率无法计算，不是 JSON 损坏。",
        "- **日电量（`C_daily_energy`）**：主表平均每天差 {} MWh，逐日百分比误差平均 {}；".format(
            fmt(main["C_daily_energy"]["MAE_MWh"]),
            percent(main["C_daily_energy"]["MAPE_pct"]))
        + "附表分别为 {} MWh 和 {}。".format(
            fmt(all_rows["C_daily_energy"]["MAE_MWh"]),
            percent(all_rows["C_daily_energy"]["MAPE_pct"])),
        "  百分比误差会被真实电量很小的日子放大：主表 Day {} 的真实电量约 {} MWh，".format(
            worst_energy_day["day"], fmt(worst_energy_day["actual_MWh"]))
        + "该日百分比误差为 {}。".format(percent(worst_energy_day["APE_pct"])),
        "- **高风速（`C_high_wind`）**：门槛为 {} m/s，主表可评估样本 {:,} 个，附表 {:,} 个；没有样本时不能判断高风速表现。".format(
            fmt(cfg["high_wind_speed"]), main["C_high_wind"]["n"],
            all_rows["C_high_wind"]["n"]),
        "- **功率曲线（`D_physics`）**：主表预测值偏离训练段功率曲线容许范围的比例为 {}，真实值自身为 {}；".format(
            percent(main["D_physics"]["violation_rate_pct"]),
            percent(main["D_physics"]["truth_violation_rate_pct"]))
        + "附表分别为 {} 和 {}。".format(
            percent(all_rows["D_physics"]["violation_rate_pct"]),
            percent(all_rows["D_physics"]["truth_violation_rate_pct"])),
        "  这是与经验曲线不一致的比例，不等于物理定律被违反的比例。主表曲线覆盖率 {}，附表 {}；两表的曲线比例要结合覆盖率一起看。".format(
            percent(main["D_physics"]["curve_coverage_pct"]),
            percent(all_rows["D_physics"]["curve_coverage_pct"])),
        "- **概率指标（`E_probabilistic`）**：值为 `null`，表示当前基线没有输出概率预测，此项尚未评估。",
        "",
        "## 后续模型先比什么",
        "",
        "在**相同主表规则**下，先看单机 MAE 是否低于 {} kW，再看第 {} 步、".format(
            fmt(main_mae), cfg["horizon"])
        + "误差较高的风速段和 Day {} 等测试时段是否改善。".format(worst_days["days"]),
        "突变事件与功率曲线指标可作为补充，避免只改善平均误差。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_EVAL_DIR,
                        help="自动寻找最新 JSON 的目录，默认是项目 reports/eval")
    parser.add_argument("--main", type=Path, help="指定 persistence_main_*.json")
    parser.add_argument("--all", type=Path, help="指定 persistence_all_*.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_EVAL_DIR / "persistence_analysis.md",
                        help="输出 Markdown 路径")
    args = parser.parse_args()
    try:
        if (args.main is None) != (args.all is None):
            raise ValueError("--main 和 --all 必须一起提供")
        main_path, all_path = ((args.main, args.all) if args.main is not None
                               else latest_pair(args.input_dir))
        report = make_report(read_result(main_path, "main"), read_result(all_path, "all"),
                             main_path, all_path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, "分析失败：{}\n".format(exc))
    print("分析完成：{}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
