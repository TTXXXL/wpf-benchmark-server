"""Train-only preregistered calibration of SDWPF wind/wake conventions."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data.io import load_clean
from ..evaluation.config import ProtocolConfig
from ..paths import ProjectPaths
from ..models.wake import (CONVENTIONS, WakeConfig, WakeConvention,
                           circular_mean_degrees, direction_bins,
                           load_locations, wake_matrices, wind_from_degrees)


@dataclass(frozen=True)
class CalibrationConfig:
    selection_fraction: float = 0.6
    min_reference_turbines: int = 5
    min_bin_days: int = 20
    min_stratum_each: int = 50
    min_inside: int = 5000
    min_inside_days: int = 30
    speed_bin_width: float = 1.0
    wind_anchor_ms: float = 0.3
    power_anchor_pct: float = 2.0
    wind_tie_ms: float = 0.05
    power_tie_pct: float = 0.5
    phi_step_degrees: int = 5
    phi_tolerance_degrees: float = 15.0
    bootstrap_reps: int = 200
    permutation_reps: int = 1000
    random_seed: int = 1729


@dataclass
class CalibrationData:
    day: np.ndarray
    speed: np.ndarray
    power: np.ndarray
    ndir: np.ndarray
    wdir: np.ndarray
    valid_direction: np.ndarray
    valid_wind: np.ndarray
    valid_power: np.ndarray

    def subset(self, selected: np.ndarray) -> "CalibrationData":
        return CalibrationData(*[getattr(self, key)[selected]
                                 for key in self.__dataclass_fields__])


@dataclass
class CandidateState:
    convention: WakeConvention
    bins: np.ndarray
    exposed: np.ndarray
    vref: np.ndarray
    reference_valid: np.ndarray
    deficit_v: np.ndarray
    deficit_p: Optional[np.ndarray] = None
    curve: Optional[np.ndarray] = None

    def subset(self, selected: np.ndarray) -> "CandidateState":
        return CandidateState(self.convention, self.bins[selected],
                              self.exposed[selected], self.vref[selected],
                              self.reference_valid[selected],
                              self.deficit_v[selected],
                              self.deficit_p[selected] if self.deficit_p is not None else None,
                              self.curve)


def _read_data(paths: ProjectPaths, protocol: ProtocolConfig,
               turbine_ids: np.ndarray) -> CalibrationData:
    flags = list(protocol.exclude_flags_main)
    columns = list(dict.fromkeys(["ts", "Day", "TurbID", "Wspd", "Patv",
                                  "Ndir", "Wdir", "m_missing_wspd",
                                  "m_imputed_wspd", "f_stuck"] + flags))
    frame = load_clean(paths, columns=columns,
                       filters=[("Day", "<=", protocol.train_days)])
    frame = frame.sort_values(["ts", "TurbID"])
    n = len(turbine_ids)
    if not np.array_equal(np.sort(frame["TurbID"].unique()), turbine_ids) or (
            len(frame) % n):
        raise ValueError("Training data and location turbine grids differ")
    times = len(frame) // n
    def array(name, dtype=None):
        return frame[name].to_numpy(dtype=dtype).reshape(times, n)
    day = array("Day", np.int32)[:, 0]
    direction_good = (np.isfinite(array("Ndir", np.float32)) &
                      np.isfinite(array("Wdir", np.float32)) &
                      ~array("m_missing", bool) & ~array("m_imputed", bool))
    wind_good = (np.isfinite(array("Wspd", np.float32)) &
                 ~array("m_missing_wspd", bool) &
                 ~array("m_imputed_wspd", bool) & ~array("f_stuck", bool))
    power_good = np.isfinite(array("Patv", np.float32))
    for flag in flags:
        power_good &= ~array(flag, bool)
    return CalibrationData(day, array("Wspd", np.float32),
                           array("Patv", np.float32), array("Ndir", np.float32),
                           array("Wdir", np.float32), direction_good,
                           wind_good, power_good)


def _global_bins(data: CalibrationData, convention: WakeConvention) -> np.ndarray:
    heading = wind_from_degrees(data.ndir, data.wdir, convention)
    heading[~data.valid_direction] = np.nan
    return direction_bins(circular_mean_degrees(heading))


def _reference_state(data: CalibrationData, convention: WakeConvention,
                     bins: np.ndarray, matrices: np.ndarray,
                     config: CalibrationConfig) -> CandidateState:
    t, n = data.speed.shape
    exposed_by_bin = matrices.sum(axis=2) > 0
    exposed = np.zeros((t, n), dtype=bool)
    valid_bin = bins >= 0
    exposed[valid_bin] = exposed_by_bin[bins[valid_bin]]
    free = ~exposed & data.valid_wind & valid_bin[:, None]
    count = free.sum(axis=1)
    reference_valid = valid_bin & (count >= config.min_reference_turbines)
    vref = np.full(t, np.nan, dtype=np.float32)
    if reference_valid.any():
        rows = np.flatnonzero(reference_valid)
        ordered = np.sort(np.where(free[rows], data.speed[rows], np.inf), axis=1)
        left = (count[rows] - 1) // 2
        right = count[rows] // 2
        vref[rows] = (ordered[np.arange(len(rows)), left] +
                      ordered[np.arange(len(rows)), right]) / 2.0
    deficit_v = vref[:, None] - data.speed
    return CandidateState(convention, bins, exposed, vref,
                          reference_valid, deficit_v)


def _fit_reference_curve(data: CalibrationData, state: CandidateState,
                         config: CalibrationConfig) -> Optional[np.ndarray]:
    """Fit only on selection days; the confirmation curve is frozen."""
    n_bins = int(np.ceil(40.0 / config.speed_bin_width))
    curve = np.full(n_bins, np.nan, dtype=np.float32)
    free_power = (~state.exposed & data.valid_power & data.valid_wind &
                  state.reference_valid[:, None])
    bin_index = np.clip(np.floor(state.vref / config.speed_bin_width),
                        0, n_bins - 1)
    for index in range(n_bins):
        times = state.reference_valid & (bin_index == index)
        values = data.power[times][free_power[times]]
        if len(values) >= config.min_stratum_each:
            curve[index] = np.median(values)
    good = np.flatnonzero(np.isfinite(curve))
    if len(good) < 2:
        return None
    curve[:] = np.interp(np.arange(n_bins), good, curve[good])
    return curve


def _attach_power_deficit(data: CalibrationData, state: CandidateState,
                          curve: Optional[np.ndarray],
                          config: CalibrationConfig,
                          rated_power_kw: float) -> None:
    state.curve = curve
    if curve is None:
        state.deficit_p = np.full_like(state.deficit_v, np.nan)
        return
    index = np.clip(np.floor(state.vref / config.speed_bin_width),
                    0, len(curve) - 1)
    reference = curve[np.where(np.isfinite(index), index, 0).astype(np.int32)]
    state.deficit_p = (reference[:, None] - data.power) * (100.0 / rated_power_kw)
    state.deficit_p[~state.reference_valid] = np.nan


def _common_support(states: List[CandidateState], days: np.ndarray,
                    config: CalibrationConfig) -> Tuple[np.ndarray, List[int]]:
    common = np.logical_and.reduce([state.reference_valid for state in states])
    # Filtering one candidate's bins can remove another candidate's day
    # coverage; iterate until the eligible common set is stable.
    while True:
        common_bins = set(range(8))
        for state in states:
            eligible = {k for k in range(8) if len(np.unique(days[
                common & (state.bins == k)])) >= config.min_bin_days}
            common_bins &= eligible
        ordered = sorted(common_bins)
        if not ordered:
            return np.zeros_like(common), []
        revised = common.copy()
        for state in states:
            revised &= np.isin(state.bins, ordered)
        if np.array_equal(revised, common):
            return common, ordered
        common = revised


def _strata(data: CalibrationData, state: CandidateState,
            common: np.ndarray, config: CalibrationConfig):
    summaries = {}
    excluded = {"inside_below_min": 0, "outside_below_min": 0,
                "both_below_min": 0}
    speed_bins = np.floor(state.vref / config.speed_bin_width)
    for index in np.unique(speed_bins[common & np.isfinite(speed_bins)]):
        ticks = np.flatnonzero(common & (speed_bins == index))
        if not len(ticks):
            continue
        for turbine in range(data.speed.shape[1]):
            good = (data.valid_wind[ticks, turbine] &
                    data.valid_power[ticks, turbine] &
                    np.isfinite(state.deficit_p[ticks, turbine]))
            inside = good & state.exposed[ticks, turbine]
            outside = good & ~state.exposed[ticks, turbine]
            n_in, n_out = int(inside.sum()), int(outside.sum())
            if min(n_in, n_out) < config.min_stratum_each:
                if n_in < config.min_stratum_each and n_out < config.min_stratum_each:
                    excluded["both_below_min"] += 1
                elif n_in < config.min_stratum_each:
                    excluded["inside_below_min"] += 1
                else:
                    excluded["outside_below_min"] += 1
                continue
            v = state.deficit_v[ticks, turbine]
            p = state.deficit_p[ticks, turbine]
            summaries[(int(index), turbine)] = (
                float(np.median(v[inside]) - np.median(v[outside])),
                float(np.median(p[inside]) - np.median(p[outside])),
                n_in + n_out)
    return summaries, excluded


def _estimate(data: CalibrationData, states: List[CandidateState],
              config: CalibrationConfig) -> dict:
    common, bins = _common_support(states, data.day, config)
    if not bins or not common.any():
        return {"status": "undetermined", "reason": "No common wind bins",
                "common_times": 0, "common_bins": bins}
    counted = [_strata(data, state, common, config) for state in states]
    summaries = [entry[0] for entry in counted]
    exclusions = [entry[1] for entry in counted]
    all_strata = set().union(*(set(entry) for entry in summaries))
    shared = set(summaries[0])
    for entry in summaries[1:]:
        shared &= set(entry)
    if not shared:
        return {"status": "undetermined", "reason": "No common eligible strata",
                "common_times": int(common.sum()), "common_bins": bins}
    keys = sorted(shared)
    pooled = np.array([sum(item[key][2] for item in summaries)
                       for key in keys], dtype=np.float64)
    weight = pooled / pooled.sum()
    candidates = []
    for state, summary, excluded in zip(states, summaries, exclusions):
        inside = (state.exposed & data.valid_wind & data.valid_power &
                  common[:, None])
        n_in = int(inside.sum())
        n_out = int((~state.exposed & data.valid_wind & data.valid_power &
                     common[:, None]).sum())
        inside_days = int(len(np.unique(data.day[inside.any(axis=1)])))
        estimates = np.array([summary[key][:2] for key in keys])
        candidates.append({
            "name": state.convention.name,
            "delta_v_ms": float(np.dot(weight, estimates[:, 0])),
            "delta_p_pct": float(np.dot(weight, estimates[:, 1])),
            "n_in": n_in, "n_out": n_out, "inside_days": inside_days,
            "eligible_strata": len(summary), "shared_strata": len(keys),
            "excluded_from_common_strata": len(all_strata - set(summary)),
            "stratum_exclusions": dict(excluded,
                                       eligible_not_shared=len(summary) - len(keys)),
            "reference_coverage": float(state.reference_valid.mean()),
            "judgeable": n_in >= config.min_inside and
                         inside_days >= config.min_inside_days})
    return {"status": "ok", "common_times": int(common.sum()),
            "common_bins": bins, "common_strata": len(keys),
            "candidates": candidates, "_common": common}


def _rotation(convention: WakeConvention) -> float:
    degrees = (0 if convention.x_axis == "north" else 90) + (
        0 if convention.direction == "from" else 180)
    return ((degrees + 180) % 360) - 180


def _choose(candidates: List[dict], config: CalibrationConfig) -> Optional[str]:
    available = [row for row in candidates if row["judgeable"]]
    if not available:
        return None
    available.sort(key=lambda row: row["delta_v_ms"], reverse=True)
    best = available[0]
    if len(available) > 1 and (
            best["delta_v_ms"] - available[1]["delta_v_ms"] < config.wind_tie_ms):
        second = available[1]
        if abs(best["delta_p_pct"] - second["delta_p_pct"]) >= config.power_tie_pct:
            best = max((best, second), key=lambda row: row["delta_p_pct"])
        else:
            order = {candidate.name: i for i, candidate in enumerate(CONVENTIONS)}
            best = min((best, second), key=lambda row: (
                abs(_rotation(WakeConvention.parse(row["name"]))), order[row["name"]]))
    return best["name"]


def _classify(delta: float, lower: float, anchor: float) -> str:
    if np.isfinite(delta) and delta > 0 and lower > 0 and delta >= anchor:
        return "significant"
    if np.isfinite(delta) and delta > 0:
        return "weak"
    return "none"


def _bootstrap(data: CalibrationData, states: List[CandidateState],
               selected: str, config: CalibrationConfig) -> dict:
    days = np.unique(data.day)
    rng = np.random.RandomState(config.random_seed)
    values = {state.convention.name: [] for state in states}
    for _ in range(config.bootstrap_reps):
        drawn = rng.choice(days, size=len(days), replace=True)
        blocks = [np.flatnonzero(data.day == day) for day in drawn]
        indices = np.concatenate(blocks)
        sample = data.subset(indices)
        # Each draw is one bootstrap block, even when the same original day
        # appears more than once. Coverage thresholds count drawn blocks.
        sample.day = np.concatenate([np.full(len(block), i, dtype=data.day.dtype)
                                     for i, block in enumerate(blocks)])
        estimate = _estimate(sample,
                             [state.subset(indices) for state in states], config)
        if estimate["status"] != "ok":
            continue
        for row in estimate["candidates"]:
            values[row["name"]].append((row["delta_v_ms"], row["delta_p_pct"]))
    count = len(values[selected])
    if count < max(20, config.bootstrap_reps // 2):
        return {"status": "undetermined", "valid_reps": count}
    intervals = {}
    for name, observations in values.items():
        array = np.asarray(observations)
        intervals[name] = {
            "wind_ci95": np.percentile(array[:, 0], [2.5, 97.5]).tolist(),
            "power_ci95": np.percentile(array[:, 1], [2.5, 97.5]).tolist(),
            "valid_reps": len(observations)}
    return {"status": "ok", "valid_reps": count,
            "wind_ci95": intervals[selected]["wind_ci95"],
            "power_ci95": intervals[selected]["power_ci95"],
            "per_candidate": intervals}


def _phi_scan(data: CalibrationData, xy_km: np.ndarray,
              geometry: WakeConfig, config: CalibrationConfig) -> Dict[str, dict]:
    result = {}
    for sign in (1, -1):
        convention = WakeConvention(sign, "north", "from")
        bins = _global_bins(data, convention)
        scores = []
        rotations = list(range(0, 360, config.phi_step_degrees))
        for phi in rotations:
            graph, _ = wake_matrices(xy_km, convention, geometry, phi)
            state = _reference_state(data, convention, bins, graph, config)
            valid = (state.reference_valid[:, None] & data.valid_wind &
                     data.valid_power & np.isfinite(state.deficit_v))
            inner = valid & state.exposed
            outer = valid & ~state.exposed
            scores.append(float(np.median(state.deficit_v[inner]) -
                                np.median(state.deficit_v[outer])) if (
                                    inner.any() and outer.any()) else float("nan"))
        finite = np.isfinite(scores)
        peak = int(np.nanargmax(scores)) if finite.any() else None
        result["plus" if sign == 1 else "minus"] = {
            "rotations": rotations, "delta_v_ms": scores,
            "peak_degrees": rotations[peak] if peak is not None else None,
            "peak_score": scores[peak] if peak is not None else None}
    return result


def _day_permutation_p(data: CalibrationData, state: CandidateState,
                       common: np.ndarray, config: CalibrationConfig) -> Optional[float]:
    daily = []
    valid = (data.valid_wind & data.valid_power & common[:, None] &
             np.isfinite(state.deficit_v))
    for day in np.unique(data.day[common]):
        take = data.day == day
        inside = valid[take] & state.exposed[take]
        outside = valid[take] & ~state.exposed[take]
        if inside.any() and outside.any():
            daily.append(float(np.median(state.deficit_v[take][inside]) -
                               np.median(state.deficit_v[take][outside])))
    if len(daily) < 2:
        return None
    effects = np.asarray(daily)
    observed = effects.mean()
    rng = np.random.RandomState(config.random_seed + 1)
    signs = rng.choice((-1.0, 1.0),
                       size=(config.permutation_reps, len(effects)))
    null = (signs * effects[None, :]).mean(axis=1)
    return float((1 + (null >= observed).sum()) / (1 + len(null)))


def _render_report(result: dict) -> str:
    lines = ["# W0 尾流方向标定", "", "## 预注册协议", "",
             "本报告只使用训练段。前 60% 日期选择候选，后 40% 日期确认一次。",
             "风速亏损与功率亏损均采用正值代表下游亏损；所有阈值和候选顺序在代码中固定。",
             "候选比较使用共同参考时间、共同风向 bin、共同可纳入分层和固定分层权重。",
             "确认集按天重采样估计 95% CI；按天符号置换的 p 值仅作报告。",
             "", "- 参考机组不足 5 台的时间步剔除；每个候选风向 bin 至少覆盖 20 天。",
             "- 分层为参考风速分箱 × 机组；各候选层内扇区内/外均至少 50 条，"
             "取 8 候选可纳入层的交集，层权重由 8 候选合并样本数归一化。",
             "- 扇区内至少 5,000 条且覆盖 30 天才可判定。选择集按风速亏损降序；"
             "前两名差小于 0.05 m/s 时看功率亏损，差小于 0.5% 时看折回的旋转角绝对值，"
             "再并列按预注册清单顺序。",
             "- 确认集逐通道：δ>0、CI 下界>0 且达到锚点为显著；其余 δ>0 为弱证据；"
             "δ≤0 为无信号。两通道取较弱一档；仅功率显著时标记方法学可疑。",
             "- 只有两通道均显著且 φ 峰离最近 90° 不超过 15°，尾流进入主方法。"
             "弱证据、无信号、无法判定或方位未确认均使用无尾流两变体。",
             "", "```json", json.dumps(result["config"], ensure_ascii=False, indent=2),
             "```", "", "## 结论", "",
             "- 选择约定：{}".format(result.get("selected_convention") or "无法判定"),
             "- 结论：{}".format(result["decision"]),
             "- 原因：{}".format(result.get("reason") or "按预注册规则判定"),
             "", "## 候选比较（选择集）", "",
             "| 候选 | δ风速 (m/s) | δ功率 (%额定) | 扇区内 n | 扇区外 n | 日期数 | 共享/自身层数 | 层剔除：内/外/双侧/非共享 | 自身参考覆盖率 | 可判定 |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for row in result.get("selection_candidates", []):
        reason = row.get("stratum_exclusions", {})
        counts = "/".join(str(reason.get(key, 0)) for key in (
            "inside_below_min", "outside_below_min", "both_below_min",
            "eligible_not_shared"))
        lines.append("| {} | {:.4f} | {:.4f} | {} | {} | {} | {}/{} | {} | {:.1%} | {} |".format(
            row["name"], row["delta_v_ms"], row["delta_p_pct"],
            row["n_in"], row["n_out"], row["inside_days"],
            row["shared_strata"], row["eligible_strata"],
            counts,
            row["reference_coverage"],
            "是" if row["judgeable"] else "否"))
    lines += ["", "### 各风向 bin 的自由流参考覆盖率（选择集）", "",
              "| 候选 | bin 0 | bin 1 | bin 2 | bin 3 | bin 4 | bin 5 | bin 6 | bin 7 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, coverage in result.get("selection_reference_coverage_by_bin", {}).items():
        lines.append("| {} | {} |".format(name, " | ".join(
            "{:.1%}".format(value) if value is not None else "—"
            for value in coverage)))
    confirm = result.get("confirmation", {})
    lines += ["", "## 确认集", ""]
    intervals = confirm.get("bootstrap", {}).get("per_candidate", {})
    if intervals:
        lines += ["| 候选 | 风速 95% CI (m/s) | 功率 95% CI (%额定) | 重采样次数 |",
                  "|---|---|---|---:|"]
        for name, row in intervals.items():
            v, p = row["wind_ci95"], row["power_ci95"]
            lines.append("| {} | [{:.3f}, {:.3f}] | [{:.3f}, {:.3f}] | {} |".format(
                name, v[0], v[1], p[0], p[1], row["valid_reps"]))
    lines += ["",
              "```json", json.dumps(confirm, ensure_ascii=False, indent=2), "```",
              "", "## φ 诊断（仅选择集）", "",
              "```json", json.dumps(result.get("phi", {}), ensure_ascii=False, indent=2),
              "```", ""]
    return "\n".join(lines)


def _coverage_by_bin(states: List[CandidateState]) -> Dict[str, List[Optional[float]]]:
    coverage = {}
    for state in states:
        entries = []
        for k in range(8):
            rows = state.bins == k
            entries.append(float(state.reference_valid[rows].mean())
                           if rows.any() else None)
        coverage[state.convention.name] = entries
    return coverage


def calibrate_wake(paths: ProjectPaths, protocol: Optional[ProtocolConfig] = None,
                   geometry: WakeConfig = WakeConfig(),
                   config: CalibrationConfig = CalibrationConfig()) -> dict:
    """Select on one chronological training block; confirm once on another."""
    protocol = protocol or ProtocolConfig()
    turbine_ids, xy_km = load_locations(paths)
    all_data = _read_data(paths, protocol, turbine_ids)
    days = np.unique(all_data.day)
    cut = int(len(days) * config.selection_fraction)
    if cut <= 0 or cut >= len(days):
        raise ValueError("Not enough training days to split wake calibration")
    selection = all_data.subset(np.isin(all_data.day, days[:cut]))
    confirmation = all_data.subset(np.isin(all_data.day, days[cut:]))
    output = paths.reports / "wake_calibration.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    prereg = {"calibration": asdict(config), "geometry": asdict(geometry),
              "train_days": protocol.train_days,
              "candidate_order": [candidate.name for candidate in CONVENTIONS]}
    protocol_path = paths.reports / "wake_protocol.json"
    if protocol_path.is_file():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != prereg:
            raise ValueError("W0 protocol is already registered with different settings")
    else:
        protocol_path.write_text(json.dumps(prereg, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    output.write_text("# W0 预注册协议\n\n```json\n" +
                      json.dumps(prereg, ensure_ascii=False, indent=2) +
                      "\n```\n\n运行中，结果尚未写入。\n", encoding="utf-8")
    selection_states = []
    curves = {}
    for candidate in CONVENTIONS:
        graph, _ = wake_matrices(xy_km, candidate, geometry)
        state = _reference_state(selection, candidate,
                                 _global_bins(selection, candidate), graph, config)
        curve = _fit_reference_curve(selection, state, config)
        _attach_power_deficit(selection, state, curve, config,
                              protocol.rated_power_kw)
        curves[candidate.name] = curve
        selection_states.append(state)
    estimate = _estimate(selection, selection_states, config)
    selected = (_choose(estimate["candidates"], config)
                if estimate["status"] == "ok" else None)
    phi = _phi_scan(selection, xy_km, geometry, config)
    result = {"config": prereg, "selected_convention": selected,
              "decision": "undetermined", "reason": estimate.get("reason"),
              "selection_candidates": estimate.get("candidates", []),
              "selection_common_bins": estimate.get("common_bins", []),
              "selection_common_times": estimate.get("common_times", 0),
              "selection_reference_coverage_by_bin": _coverage_by_bin(selection_states),
              "phi": phi}
    if selected is not None:
        family = "plus" if WakeConvention.parse(selected).sign == 1 else "minus"
        peak = phi[family]["peak_degrees"]
        offset = min(abs(((peak - cardinal + 180) % 360) - 180)
                     for cardinal in (0, 90, 180, 270)) if peak is not None else 180
        result["phi_offset_degrees"] = offset
        confirm_states = []
        for candidate in CONVENTIONS:
            graph, _ = wake_matrices(xy_km, candidate, geometry)
            state = _reference_state(confirmation, candidate,
                                     _global_bins(confirmation, candidate),
                                     graph, config)
            _attach_power_deficit(confirmation, state, curves[candidate.name],
                                  config, protocol.rated_power_kw)
            confirm_states.append(state)
        confirmed = _estimate(confirmation, confirm_states, config)
        if confirmed["status"] == "ok":
            chosen = next(row for row in confirmed["candidates"]
                          if row["name"] == selected)
            boot = _bootstrap(confirmation, confirm_states, selected, config)
            common = confirmed.pop("_common")
            selected_state = next(state for state in confirm_states
                                  if state.convention.name == selected)
            p_value = _day_permutation_p(confirmation, selected_state,
                                         common, config)
            result["confirmation"] = {"candidate": chosen, "bootstrap": boot,
                                      "day_permutation_p": p_value,
                                      "reference_coverage_by_bin": _coverage_by_bin(confirm_states),
                                      "common_bins": confirmed["common_bins"],
                                      "common_times": confirmed["common_times"]}
            if chosen["judgeable"] and boot["status"] == "ok":
                wind = _classify(chosen["delta_v_ms"], boot["wind_ci95"][0],
                                 config.wind_anchor_ms)
                power = _classify(chosen["delta_p_pct"], boot["power_ci95"][0],
                                  config.power_anchor_pct)
                result["channel_decisions"] = {"wind": wind, "power": power}
                rank = {"none": 0, "weak": 1, "significant": 2}
                result["decision"] = min((wind, power), key=lambda x: rank[x])
                result["reason"] = "Confirmation channel decisions"
                if power == "significant" and wind != "significant":
                    result["methodology_suspect"] = True
                    result["decision"] = wind
                    result["reason"] = ("Power-only signal; check curtailment "
                                        "and fault filtering")
            else:
                result["reason"] = "Confirmation coverage or bootstrap insufficient"
        else:
            result["reason"] = "Confirmation has no common eligible support"
        if offset > config.phi_tolerance_degrees:
            result["decision"] = "orientation_unconfirmed"
            result["reason"] = "Selection-set phi peak is not near a cardinal rotation"
    # Strict JSON is used by prepare-wake-prior; Markdown is the human report.
    from ..evaluation.report import _json_safe
    safe = _json_safe(result)
    (paths.reports / "wake_calibration.json").write_text(
        json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    output.write_text(_render_report(safe), encoding="utf-8")
    return safe
