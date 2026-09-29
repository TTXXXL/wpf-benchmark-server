# -*- coding: utf-8 -*-
"""SDWPF 数据验收与 EDA（阶段 0）。

对 sdwpf_245days_v1.csv 做全表体检：
  1. 结构核对（行数 / 风机数 / 天数 / 时间步）
  2. 各列缺失率与异常值统计
  3. 限电 / 停机段识别（基于 Prtv 与 Patv 的关系）
  4. 风速-功率散点密度图 + 分风机的功率曲线（含滞回检查素材）
  5. 尾流预研素材：相邻机组功率滞后相关（按风向分组）

输出：reports/eda_sdwpf.md + reports/figs/*.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..paths import ProjectPaths

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

COLS = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
STEPS_PER_DAY = 144  # 10 min × 24 h


def load(paths: ProjectPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.raw / "sdwpf_245days_v1.csv")
    # 统一时间索引：Day + Tmstamp → 绝对时刻
    df["ts"] = pd.to_timedelta(df["Day"] - 1, unit="D") + pd.to_timedelta(df["Tmstamp"] + ":00")
    df["TurbID"] = df["TurbID"].astype(int)
    df["Day"] = df["Day"].astype(int)
    return df


def structure_check(df: pd.DataFrame, lines: list[str]) -> None:
    n_turb = df["TurbID"].nunique()
    n_day = df["Day"].nunique()
    lines.append(f"- 行数 {len(df):,}（期望 134×245×144 = {134*245*STEPS_PER_DAY:,}）")
    lines.append(f"- 风机数 {n_turb}，天数 {n_day}，每日时间步 {STEPS_PER_DAY}")
    dup = df.duplicated(subset=["TurbID", "Day", "Tmstamp"]).sum()
    lines.append(f"- (TurbID, Day, Tmstamp) 重复行 {dup}")
    per_turb = df.groupby("TurbID").size()
    lines.append(f"- 每台机行数 min/max：{per_turb.min()}/{per_turb.max()}")


def missing_check(df: pd.DataFrame, lines: list[str]) -> None:
    lines.append("\n## 各列缺失率\n")
    lines.append("| 列 | 缺失数 | 缺失率 |")
    lines.append("|---|---|---|")
    for c in COLS:
        n = df[c].isna().sum()
        lines.append(f"| {c} | {n:,} | {n/len(df):.2%} |")
    # 空行（首个 00:00 全空）：确认是每日起始占位还是随机
    empty_rows = df[COLS].isna().all(axis=1).sum()
    at_day_start = (df.loc[df[COLS].isna().all(axis=1), "Tmstamp"] == "00:00").mean()
    lines.append(f"\n- 整行全空的行数 {empty_rows:,}，其中位于每日 00:00 的比例 {at_day_start:.1%}")


def range_check(df: pd.DataFrame, lines: list[str]) -> None:
    lines.append("\n## 数值范围\n")
    lines.append("| 列 | min | max | 均值 | 说明 |")
    lines.append("|---|---|---|---|---|")
    notes = {
        "Wspd": "风速 m/s，物理上限约 40",
        "Wdir": "风向（单位需确认，KDD Cup 官方说明为 °）",
        "Etmp": "环境温度 ℃，风电场常见范围 -30~50",
        "Itmp": "机舱/机内温度 ℃",
        "Ndir": "机舱方位角",
        "Pab1": "桨距角 1（°）",
        "Pab2": "桨距角 2（°）",
        "Pab3": "桨距角 3（°）",
        "Prtv": "有功功率指令值 kW（限电时 < 实际可用功率）",
        "Patv": "实际有功功率 kW，额定约 2000",
    }
    for c in COLS:
        s = df[c].dropna()
        lines.append(f"| {c} | {s.min():.2f} | {s.max():.2f} | {s.mean():.2f} | {notes[c]} |")


def curtail_idle(df: pd.DataFrame, lines: list[str], paths: ProjectPaths) -> None:
    """限电/停机/异常段统计——清洗策略的依据。"""
    m = df.dropna(subset=["Patv", "Wspd"])
    idle = (m["Patv"] <= 0) & (m["Wspd"] >= 3)          # 风够却零功率：停机或故障
    curtail = (m["Patv"] > 0) & (m["Prtv"].notna()) & (m["Patv"] < m["Prtv"] - 10)  # 指令高于实际≈不限电；实际明显低于指令视情况
    # 更稳的限电判据：Patv ≈ Prtv 且两者都低于风速对应功率——先给原始统计
    above_rated = m["Wspd"] >= 11.5
    neg = m["Patv"] < 0
    lines.append("\n## 限电 / 停机 / 异常模式\n")
    lines.append(f"- 停机/故障嫌疑（Patv≤0 且 Wspd≥3 m/s）：{idle.sum():,} 行（{idle.mean():.2%}）")
    lines.append(f"- 负功率行（Patv<0）：{neg.sum():,} 行（{neg.mean():.2%}）")
    lines.append(f"- 超额定风速行（Wspd≥11.5）：{above_rated.sum():,} 行；其中 Patv<1500 kW 的 {((m['Wspd']>=11.5)&(m['Patv']<1500)).sum():,} 行（限电主要藏在这里）")
    # Patv==Prtv 的比例：说明限电期 Patv 被钳制在指令值
    both = m[m["Prtv"].notna()]
    clamp = (both["Patv"] - both["Prtv"]).abs() < 1
    lines.append(f"- |Patv−Prtv|<1 kW 的行占（Prtv 非空行）{clamp.mean():.1%}；Prtv 非空且 <Patv 的比例 {((both['Prtv']<both['Patv'])).mean():.1%}")
    # 按天统计停机行，画时间热力图素材
    daily = m.assign(idle=idle).groupby("Day")["idle"].mean()
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.imshow(daily.values[None, :], aspect="auto", cmap="Reds", extent=[1, 245, 0, 1])
    ax.set_yticks([]); ax.set_xlabel("Day"); ax.set_title("Daily share of suspected idle or fault rows")
    fig.tight_layout(); fig.savefig(paths.figures / "idle_daily.png", dpi=150); plt.close(fig)


def power_curve(df: pd.DataFrame, lines: list[str], paths: ProjectPaths) -> None:
    """风速-功率密度散点 + 单机功率曲线（滞回素材）。"""
    m = df.dropna(subset=["Patv", "Wspd"]).sample(n=min(500_000, len(df)), random_state=0)
    m = m[(m["Patv"] >= 0) & (m["Wspd"] < 30)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.hexbin(m["Wspd"], m["Patv"], gridsize=90, cmap="viridis", bins="log", mincnt=1)
    ax.set_xlabel("Wind speed Wspd (m/s)"); ax.set_ylabel("Power Patv (kW)")
    ax.set_title("Farm wind-power density (log scale)")
    # 单机：1 号机，区分升/降风速段看滞回
    ax = axes[1]
    t1 = df[df["TurbID"] == 1].dropna(subset=["Patv", "Wspd"])
    t1 = t1[(t1["Patv"] >= 0) & (t1["Wspd"] < 30)].copy()
    dv = t1["Wspd"].diff()  # 相邻 10min 风速变化
    up = t1[dv > 0.05]; down = t1[dv < -0.05]
    for seg, lbl, c in [(up, "Increasing wind speed", "#0072B2"), (down, "Decreasing wind speed", "#D55E00")]:
        ax.scatter(seg["Wspd"], seg["Patv"], s=2, alpha=0.15, color=c, label=lbl)
    ax.legend(markerscale=8); ax.set_xlabel("Wind speed (m/s)"); ax.set_ylabel("Power (kW)")
    ax.set_title("Turbine 1: wind-power hysteresis")
    fig.tight_layout(); fig.savefig(paths.figures / "power_curve.png", dpi=150); plt.close(fig)
    lines.append("\n## 风速–功率\n")
    lines.append("- 见 `figs/power_curve.png`：左为全场密度散点（功率曲线形态），右为 1 号机升/降风速段分层（滞回证据初查）")


def wake_lag(df: pd.DataFrame, lines: list[str], paths: ProjectPaths) -> None:
    """尾流预研：上下游机组功率的滞后相关（H2 素材）。

    简化版：取主导风向（全场 Wdir 中位数）下几何下游 1 跳的机组对，
    比较下游机组功率相对上游机组的滞后互相关。
    """
    loc = pd.read_csv(paths.raw / "sdwpf_baidukddcup2022_turb_location.csv", encoding="utf-8-sig")
    loc["x"] = loc["x"].astype(float); loc["y"] = loc["y"].astype(float)
    m = df.dropna(subset=["Patv", "Wdir"])
    dom = m["Wdir"].median()
    lines.append(f"\n## 尾流滞后相关（H2 预研）\n- 主导风向（Wdir 中位数）：{dom:.1f}")
    # Wdir 定义需在精读时对照官方说明；此处按"风向角 β 下，沿 (sinβ, cosβ) 方向为下游"的常用约定做初查
    rad = np.deg2rad(dom)
    ux, uy = np.sin(rad), np.cos(rad)
    P = loc.set_index("TurbID")[["x", "y"]]
    pairs = []
    ids = P.index.to_numpy()
    for i in ids:
        d = P.loc[ids] - P.loc[i]
        proj = d["x"].to_numpy() * ux + d["y"].to_numpy() * uy
        perp = np.abs(-d["x"].to_numpy() * uy + d["y"].to_numpy() * ux)
        cand = (proj > 300) & (proj < 1200) & (perp < 150)  # 下游 300–1200 m、横向 ±150 m ≈ 1 跳尾流带
        for j in ids[cand]:
            pairs.append((int(i), int(j)))
    lines.append(f"- 主导风向下识别出相邻上下游机组对 {len(pairs)} 对")
    if not pairs:
        return
    # 宽表按时间对齐，算滞后互相关（子采样加速）
    wide = m.pivot_table(index="ts", columns="TurbID", values="Patv")
    wide = wide.iloc[::6]  # 1h 分辨率，245 天足够估相关
    lags = range(0, 7)  # 0–6 h
    rs = np.zeros((len(pairs), len(lags)))
    for k, (i, j) in enumerate(pairs):
        a = wide[i].to_numpy(); b = wide[j].shift(-lags[0] * 0)  # 占位
        for li, lag in enumerate(lags):
            bj = wide[j].shift(-lag).to_numpy()  # j（下游）超前/滞后 lag 小时
            ok = np.isfinite(a) & np.isfinite(bj)
            if ok.sum() > 500:
                rs[k, li] = np.corrcoef(a[ok], bj[ok])[0, 1]
    mean_r = rs.mean(axis=0)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(lags, mean_r, marker="o", color="#0072B2")
    ax.set_xlabel("Downstream lag relative to upstream (h)"); ax.set_ylabel("Mean cross-correlation")
    ax.set_title(f"Downstream power lag ({len(pairs)} pairs)")
    fig.tight_layout(); fig.savefig(paths.figures / "wake_lag.png", dpi=150); plt.close(fig)
    best = lags[int(np.argmax(mean_r))]
    lines.append(f"- 平均互相关在滞后 {best} h 最大（{mean_r.max():.3f}）；0 滞后相关 {mean_r[0]:.3f}")
    lines.append("- 注意：此处风向单位/朝向为初查约定，正式预研须对照官方文档确认后再下结论")


def main(paths: ProjectPaths = None) -> None:
    paths = paths or ProjectPaths.resolve()
    paths.figures.mkdir(parents=True, exist_ok=True)
    df = load(paths)
    L: list[str] = ["# SDWPF 数据 EDA 报告", "", f"生成：ClawsGO Science Agent · 2026-09-24", ""]
    L.append("## 结构核对\n")
    structure_check(df, L)
    missing_check(df, L)
    range_check(df, L)
    curtail_idle(df, L, paths)
    power_curve(df, L, paths)
    wake_lag(df, L, paths)
    out = paths.reports / "eda_sdwpf.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("EDA 完成 →", out)
