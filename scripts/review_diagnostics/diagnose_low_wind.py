# -*- coding: utf-8 -*-
"""P0：低风速段系统性高估的成因判别（纯离线，不训练、不改模型）。

输入：已有 M1 预测数组 NPZ，数组内含目标时刻风速、发布时刻功率。
输出：若干 CSV + console.json，供报告引用。

七模型对照（全部 SDWPF M1、seed 0、同一掩码与同一真值，已逐项核对）：
  命名 = 骨干 _ 当前功率锚点 _ 损失
  agcrn_noskip / agcrn_skip       原版 AGCRN，MAE
  lite_noskip_mse / lite_skip_mse / lite_skip_mae   轻量图模型
  B / C                           低功率双分支（分类权重 0 / 0.05），锚点开启

注：lite_noskip_mae 一格在已有产物中不存在，故 Lite 的锚点效应只在 MSE 下可分离。

不依赖 parquet / matplotlib。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .common import daily_errors, load_manifest, new_output, source_summary, write_csv, write_json
except ImportError:
    from common import daily_errors, load_manifest, new_output, source_summary, write_csv, write_json

REPORTS = None
OUT = Path(__file__).resolve().parent

RUNS = {
    "agcrn_noskip": "m1_graph_agcrn_s0_20260930_074645_167677_arrays.npz",
    "agcrn_skip": "m1_graph_ablation_agcrn_mae_skip_s0_20260930_100350_841512_arrays.npz",
    "lite_noskip_mse": "m1_graph_agcrn_lite_s0_20260930_083533_670011_arrays.npz",
    "lite_skip_mse": "m1_graph_ablation_agcrn_lite_mse_skip_s0_20260930_111254_443627_arrays.npz",
    "lite_skip_mae": "m1_graph_ablation_agcrn_lite_mae_skip_s0_20260930_103325_346649_arrays.npz",
    "B": "lite_mixture_no_aux_20261002_092023_227415_arrays.npz",
    "C": "lite_low_power_20261002_061816_059203_arrays.npz",
}
REF = "agcrn_noskip"

WIND_BINS = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 40)]
SUB_BINS = [(0, 1), (1, 2), (2, 2.5), (2.5, 3)]
LOW_POWER_EDGES = [0, 10, 50, 300, 1e9]
LOW_POWER_LABELS = ["<10", "10-50", "50-300", ">=300"]

console: dict = {}


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    if pred.size == 0:
        return {"n": 0, "MAE_kW": None, "RMSE_kW": None, "ME_kW": None, "over_frac": None}
    err = pred - truth
    return {"n": int(pred.size),
            "MAE_kW": float(np.mean(np.abs(err))),
            "RMSE_kW": float(np.sqrt(np.mean(err ** 2))),
            "ME_kW": float(np.mean(err)),
            "over_frac": float(np.mean(err > 0))}


def statistic(values, kind):
    if not values.size:
        return None
    if kind == 'p95':
        return float(np.percentile(values, 95))
    return float(getattr(values, kind)())


def correlation(first, second):
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def nearzero_summary(preds, truth, selection, last_b, wind):
    return {'n': int(selection.sum()),
            'truth_min': statistic(truth[selection], 'min'),
            'truth_max': statistic(truth[selection], 'max'),
            'truth_mean': statistic(truth[selection], 'mean'),
            'truth_n_unique': int(np.unique(truth[selection]).size),
            'last_max': statistic(last_b[selection], 'max'),
            'ws_max': statistic(wind[selection], 'max'),
            'pred_mean_kW': {name: statistic(pred[selection], 'mean') for name, pred in preds.items()},
            'pred_p95_kW': {name: statistic(pred[selection], 'p95') for name, pred in preds.items()},
            'MAE_kW_by_horizon': {
                name: [metrics(pred[..., h][selection[..., h]], truth[..., h][selection[..., h]])['MAE_kW']
                       for h in range(truth.shape[2])] for name, pred in preds.items()}}


def group_table(groups: dict, preds: dict, truth, valid) -> list:
    rows, total = [], int(valid.sum())
    for label, sel in groups.items():
        sel = sel & valid
        row = {"group": label, "n": int(sel.sum()),
               "share_pct": 100.0 * sel.sum() / total if total else 0.0}
        for name, pred in preds.items():
            row.update({f"{name}_{k}": v for k, v in metrics(pred[sel], truth[sel]).items()
                        if k != "n"})
        rows.append(row)
    return rows


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', required=True)
    parser.add_argument('--runs', required=True, help='JSON mapping the seven model labels to run_ids')
    parser.add_argument('--out', required=True, help='New or empty output directory')
    args = parser.parse_args()
    runs = load_manifest(args.reports, args.runs)
    if set(runs) != set(RUNS):
        raise ValueError('P0 requires exactly these labels: {}'.format(sorted(RUNS)))
    console.clear()
    d = runs[REF]['arrays']
    truth = np.asarray(d['truth_m1'], dtype=np.float64)
    valid = np.asarray(d['valid_m1'], dtype=bool)
    ws = np.asarray(d['wind_speed'], dtype=np.float64)
    last = np.asarray(d['last_power'], dtype=np.float64)
    times = d['times']
    if not valid.any():
        raise ValueError('P0 requires at least one valid M1 forecast cell')
    OUT = new_output(args.out)
    console['sources'] = source_summary(runs)
    console['inference_scope'] = 'Exploratory realized-wind slices; forecast cells are not independent samples.'
    preds = {"persistence": np.broadcast_to(last[:, :, None], truth.shape).copy()}
    for name in RUNS:
        preds[name] = np.asarray(runs[name]['arrays']['forecasts'], dtype=np.float64)

    console["n_valid_M1"] = int(valid.sum())
    console["shape"] = list(truth.shape)

    # ---------- H 整体口径 + 锚点/损失分解 ----------
    overall = {n: metrics(p[valid], truth[valid])["MAE_kW"] for n, p in preds.items()}
    console["H_overall_MAE_kW"] = overall
    dec = {}
    if "lite_skip_mse" in overall:
        dec["lite_anchor_effect_mse"] = overall["lite_skip_mse"] - overall["lite_noskip_mse"]
        dec["lite_loss_effect_skip"] = overall["lite_skip_mae"] - overall["lite_skip_mse"]
    dec["agcrn_anchor_effect_mae"] = overall["agcrn_skip"] - overall["agcrn_noskip"]
    dec["C_minus_A"] = overall["C"] - overall["lite_skip_mae"]
    console["H_decomposition_kW"] = dec

    # ---------- A 主风速分箱 ----------
    main_groups = {f"[{a},{b})": (ws >= a) & (ws < b) for a, b in WIND_BINS}
    rows_a = group_table(main_groups, preds, truth, valid)
    _write_csv("A_主风速分箱.csv", rows_a)
    console["A"] = rows_a

    # ---------- B [0,3) 内部细分 ----------
    low = (ws >= 0) & (ws < 3)
    sub_groups = {f"[{a},{b})": low & (ws >= a) & (ws < b) for a, b in SUB_BINS}
    rows_b = group_table(sub_groups, preds, truth, valid)
    last_b = np.broadcast_to(last[:, :, None], truth.shape)
    for r, (a, b) in zip(rows_b, SUB_BINS):
        sel = sub_groups[r["group"]] & valid
        r["truth_mean_kW"] = statistic(truth[sel], 'mean')
        r["truth_p95_kW"] = statistic(truth[sel], 'p95')
        r["truth_max_kW"] = statistic(truth[sel], 'max')
        r["truth_below_1kw_frac"] = statistic((truth[sel] < 1.0), 'mean')
        r["truth_zero_frac"] = statistic((truth[sel] == 0.0), 'mean')
        r["last_mean_kW"] = statistic(last_b[sel], 'mean')
    _write_csv("B_低风速内部细分.csv", rows_b)
    console["B"] = rows_b

    # ---------- C [0,3) 内按发布功率分层 ----------
    idx = np.digitize(last_b, LOW_POWER_EDGES[1:-1])
    rows_c = []
    for k, lab in enumerate(LOW_POWER_LABELS):
        rows_c += group_table({lab: low & (idx == k)}, preds, truth, valid)
    _write_csv("C_低风速内按发布功率分层.csv", rows_c)
    console["C_table"] = rows_c

    # ---------- D 经验可用功率包络 + 疑似降出力子集 ----------
    step = 0.5
    edges = np.arange(0, 20.0 + step, step)
    centers, envelope = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = valid & (ws >= lo) & (ws < hi)
        if sel.sum() < 500:
            continue
        centers.append(0.5 * (lo + hi))
        envelope.append(float(np.percentile(truth[sel], 95)))
    centers, envelope = np.asarray(centers), np.asarray(envelope)
    env_at = (np.interp(ws, centers, envelope, left=envelope[0], right=envelope[-1])
              if len(envelope) else np.full_like(ws, np.nan))
    _write_csv("D_经验可用功率包络.csv",
               [{"ws_center": float(c), "available_p95_kW": float(e)}
                for c, e in zip(centers, envelope)])

    suspected = (env_at > 100.0) & (truth < 0.5 * env_at)
    envelope_groups = ({"疑似降出力": suspected, "其余": ~suspected} if len(envelope)
                       else {"包络样本不足": np.ones_like(valid)})
    rows_d = group_table(envelope_groups, preds, truth, valid)
    for r in rows_d:
        sel = envelope_groups[r['group']] & valid
        for n in ("lite_noskip_mse", "lite_skip_mae", "C"):
            total_sse = np.sum((preds[n][valid] - truth[valid]) ** 2)
            r[f"{n}_share_of_sqerr_pct"] = (100.0 * np.sum(
                (preds[n][sel] - truth[sel]) ** 2) / total_sse if total_sse else None)
    _write_csv("D2_疑似降出力误差分解.csv", rows_d)
    console["D2"] = rows_d

    # ---------- E 误差的时间聚集度（h=1，低风速段） ----------
    low_h1 = low[..., 0] & valid[..., 0]
    console["E"] = {}
    daily_rows = []
    for n in ("lite_noskip_mse", "lite_skip_mae", "C"):
        result = daily_errors(preds[n][..., 0], truth[..., 0], low_h1, times)
        console['E'][n] = result
        daily_rows.extend(dict(model=n, **row) for row in result['daily'])
    _write_csv('E_逐起报日低风速段误差.csv', daily_rows)

    # ---------- F 条件离散度（不可预测性代理） ----------
    lp_edges = [0, 10, 50, 150, 400, 1e9]
    lp_lab = ["<10", "10-50", "50-150", "150-400", ">=400"]
    rows_f = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        for k, lab in enumerate(lp_lab):
            sel = (valid & (ws >= lo) & (ws < hi)
                   & (last_b >= lp_edges[k]) & (last_b < lp_edges[k + 1]))
            if sel.sum() < 200:
                continue
            rows_f.append({
                "ws_center": float(0.5 * (lo + hi)), "last_power_band": lab,
                "n": int(sel.sum()),
                "truth_mean_kW": float(truth[sel].mean()),
                "truth_std_kW": float(truth[sel].std()),
                "truth_below_1kw_frac": float(np.mean(truth[sel] < 1.0)),
                "truth_zero_frac": float(np.mean(truth[sel] == 0.0)),
                **{f"{n}_ME_kW": float((preds[n][sel] - truth[sel]).mean())
                   for n in ("persistence", "agcrn_noskip", "lite_noskip_mse",
                             "lite_skip_mae", "C")},
            })
    _write_csv("F_条件离散度.csv", rows_f)
    f_low = [r for r in rows_f if r["ws_center"] < 3.0]
    if len(f_low) >= 4:
        console["F"] = {
            "n_cells_low_wind": len(f_low),
            "corr_truth_std_vs_liteME": correlation(
                [r["truth_std_kW"] for r in f_low],
                [r["lite_noskip_mse_ME_kW"] for r in f_low]),
            "corr_truth_zero_frac_vs_liteME": correlation(
                [r["truth_zero_frac"] for r in f_low],
                [r["lite_noskip_mse_ME_kW"] for r in f_low])}

    # ---------- G [0,1) 段核验：标签是否真的确定 ----------
    g = valid & (ws >= 0) & (ws < 1)
    console['G'] = nearzero_summary(preds, truth, g, last_b, ws)

    write_json(OUT / 'console.json', console)
    print(json.dumps({'n_valid_M1': console['n_valid_M1'],
                      'overall_MAE_kW': console['H_overall_MAE_kW'],
                      'daily': {name: {key: value for key, value in result.items() if key != 'daily'}
                                for name, result in console['E'].items()}},
                     ensure_ascii=False, indent=2))


def _write_csv(name: str, rows: list) -> None:
    write_csv(OUT / name, rows)


if __name__ == "__main__":
    main()
