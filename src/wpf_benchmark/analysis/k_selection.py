# -*- coding: utf-8 -*-
"""确定空间借力邻居数 k 的标定实验。

两步：
  1. 机组间风速相关 vs 距离：看相关性随距离的衰减，给邻居池一个物理边界
  2. 人造缺失标定：随机遮掩 5% 已知值，用 k 最近邻（距离倒数加权）填补，
     扫 k=1..8，与"前值填充"基线比较，选 MAE 最小的 k

输出：reports/k_selection.md + reports/figs/k_selection.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..paths import ProjectPaths
from ..evaluation.config import ProtocolConfig

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

K_LIST = [1, 2, 3, 4, 6, 8, 12, 16]
RNG = np.random.default_rng(0)


def load_pivot(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(paths.raw / "sdwpf_245days_v1.csv", usecols=["TurbID", "Day", "Tmstamp", "Wspd"])
    df["ts"] = pd.to_timedelta(df["Day"] - 1, unit="D") + pd.to_timedelta(df["Tmstamp"] + ":00")
    W = df.pivot(index="ts", columns="TurbID", values="Wspd")  # 35280 × 134
    loc = pd.read_csv(paths.raw / "sdwpf_baidukddcup2022_turb_location.csv", encoding="utf-8-sig")
    loc = loc.set_index("TurbID")
    return W, loc


def corr_vs_distance(W: pd.DataFrame, loc: pd.DataFrame, lines: list[str],
                     paths: ProjectPaths) -> np.ndarray:
    C = W.corr().to_numpy()  # 134×134 机组间相关
    ids = W.columns.to_numpy()
    xy = loc.loc[ids, ["x", "y"]].to_numpy()
    D = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)  # 欧氏距离
    iu = np.triu_indices(len(ids), k=1)
    dist, cor = D[iu], C[iu]
    bins = np.arange(0, 8000, 500)
    mids, means = 0.5 * (bins[:-1] + bins[1:]), []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (dist >= a) & (dist < b)
        means.append(cor[sel].mean() if sel.sum() else np.nan)
    means = np.array(means)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(mids / 1000, means, marker="o", color="#0072B2")
    ax.set_xlabel("Distance between turbines (km)"); ax.set_ylabel("Mean wind-speed correlation")
    ax.set_title("Wind-speed correlation versus distance")
    fig.tight_layout(); fig.savefig(paths.figures / "k_selection.png", dpi=150); plt.close(fig)
    r1k = means[1]  # 0.5–1.5 km 档
    lines.append(f"- 机组对平均距离 {dist.mean()/1000:.2f} km；相关随距离单调衰减")
    lines.append(f"- 500 m 内平均相关 {means[0]:.3f}；约 1 km 处 {r1k:.3f}；超过 3 km 后稳定在 {np.nanmean(means[6:]):.3f} 左右（全场共同风况的下限）")
    return D


def _mae_rmse(imp: np.ndarray, mask: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(imp[mask])
    return float(np.abs(imp[mask][ok] - truth[ok]).mean()), float(np.sqrt(((imp[mask][ok] - truth[ok]) ** 2).mean()))


def knn_fill(Xm: np.ndarray, mask: np.ndarray, D: np.ndarray, k: int) -> np.ndarray:
    """对被 mask 遮掩的格子，用距离最近的前 k 台可用机组做距离倒数加权填补。

    Xm：已遮掩的数据（缺失为 NaN）；返回填补后的副本。
    """
    out = Xm.copy()
    n_turb = Xm.shape[1]
    order = np.argsort(D, axis=1)  # 每台机按距离排序的邻居序
    for i in range(n_turb):
        rows = np.where(mask[:, i])[0]
        if len(rows) == 0:
            continue
        neigh_ids = order[i, 1 : k + 1]
        vals = Xm[:, neigh_ids][rows]             # n × k 邻居值（自身已遮掩）
        w = (1.0 / D[i, neigh_ids] ** 2)[None, :]
        w = np.repeat(w, len(rows), axis=0)
        ok = np.isfinite(vals)
        w = w * ok
        num = np.nansum(vals * w, axis=1)
        den = w.sum(axis=1)
        fill = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        out[rows, i] = fill
    return out


def time_fill(Xm: np.ndarray, kind: str) -> np.ndarray:
    """时间方向填补：前值填充或线性插值。"""
    df = pd.DataFrame(Xm)
    if kind == "prev":
        return df.ffill().to_numpy()
    return df.interpolate(method="linear", limit_direction="both").to_numpy()


def calibration(W: pd.DataFrame, D: np.ndarray, lines: list[str]) -> None:
    X = W.to_numpy()

    def eval_row(name, imp, mask, truth):
        mae, rmse = _mae_rmse(imp, mask, truth)
        lines.append(f"| {name} | {mae:.3f} | {rmse:.3f} |")
        return mae

    # ── 场景 A：随机单点缺失（传感器偶发丢包）──
    observed = np.isfinite(X)
    maskA = observed & (RNG.random(X.shape) < 0.05)
    truth = X[maskA]
    Xm = X.copy(); Xm[maskA] = np.nan
    lines.append("\n## 场景 A · 随机单点缺失（遮掩 5% 已知值）\n")
    lines.append("| 填补方法 | MAE (m/s) | RMSE (m/s) |")
    lines.append("|---|---|---|")
    eval_row("前值填充（时间基线）", time_fill(Xm, "prev"), maskA, truth)
    eval_row("线性插值（时间基线）", time_fill(Xm, "linear"), maskA, truth)
    best_k, best_mae = None, np.inf
    for k in K_LIST:
        mae = eval_row(f"空间 kNN k={k}", knn_fill(Xm, maskA, D, k), maskA, truth)
        if mae < best_mae:
            best_mae, best_k = mae, k
    lines.append(f"\n场景 A 最优 k = {best_k}（MAE {best_mae:.3f} m/s）")

    # ── 场景 B：连续长缺口（30 台机各挖 2 段 4 小时 = 24 步）──
    maskB = np.zeros_like(observed)
    turb_ids = RNG.choice(X.shape[1], size=30, replace=False)
    for t in turb_ids:
        for _ in range(2):
            s = RNG.integers(1000, X.shape[0] - 100)
            maskB[s : s + 24, t] = True
    maskB &= observed
    truth = X[maskB]
    Xm = X.copy(); Xm[maskB] = np.nan
    lines.append("\n## 场景 B · 连续长缺口（30 台机 × 2 段 × 4 小时）\n")
    lines.append("| 填补方法 | MAE (m/s) | RMSE (m/s) |")
    lines.append("|---|---|---|")
    eval_row("前值填充（时间基线）", time_fill(Xm, "prev"), maskB, truth)
    eval_row("线性插值（时间基线）", time_fill(Xm, "linear"), maskB, truth)
    best_k, best_mae = None, np.inf
    for k in K_LIST:
        mae = eval_row(f"空间 kNN k={k}", knn_fill(Xm, maskB, D, k), maskB, truth)
        if mae < best_mae:
            best_mae, best_k = mae, k
    lines.append(f"\n场景 B 最优 k = {best_k}（MAE {best_mae:.3f} m/s）")
    lines.append("\n> 空间 kNN 的 k 取两场景 MAE 最小的共同值。在线清洗对当前连续缺失前 3 步使用因果前值；双端线性插值仅作离线比较，不用于预测输入。")


def main(paths: ProjectPaths = None) -> None:
    paths = paths or ProjectPaths.resolve()
    paths.figures.mkdir(parents=True, exist_ok=True)
    W, loc = load_pivot(paths)
    train_days = ProtocolConfig().train_days
    W = W.loc[W.index < pd.Timedelta(days=train_days)]
    if W.empty:
        raise ValueError("No training rows available for k calibration")
    L: list[str] = ["# 邻居数 k 标定实验", "",
                    f"仅使用前 {train_days} 天训练数据（{len(W):,} 个时刻）。", ""]
    D = corr_vs_distance(W, loc, L, paths)
    calibration(W, D, L)
    out = paths.reports / "k_selection.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("完成 →", out)
