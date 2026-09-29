# -*- coding: utf-8 -*-
"""SDWPF 清洗管道（阶段 0）。

方案（已标定/评审通过，见 reports/eda_sdwpf.md 与 k_selection.md）：
  缺失值三级递补：
    L0 结构空行保留在时间网格内，按缺失处理
    L1 当前连续缺失 ≤3 步 → 因果前值填补
    L2 当前连续缺失 >3 步 → 同时刻空间 kNN（k=12，距离倒数平方加权）
    所有填补位置打 m_imputed mask（训练损失屏蔽，防循环论证）
  异常值三层过滤：
    P1 物理范围硬规则：温度超 [−40,60]℃ → NaN 走缺失流程；Wdir 回绕；Patv<0 截 0
    P2 物理关系异常：逐机功率曲线（分风速箱中位数，稳健）→ 残差
       |残差| > 3×1.4826×MAD（分箱稳健 σ）→ m_outlier，Patv 置 NaN 重补
    P3 工况异常：f_fault（Patv≤0 且 Wspd≥3）、f_curtail（残差大幅负偏 + Wspd≥5）、
       f_stuck（传感器卡死：连续 6 步同值）、f_farm（全场停机日）
  归一化不在此处做：本管道输出物理单位 + 全套 mask；训练脚本用 train 段统计量在线归一化
    （meta json 存好 train 段 min/max，防止数据泄漏）

输出：data/processed/sdwpf_clean.parquet, data/processed/sdwpf_meta.json,
      reports/cleaning_report.md, reports/figs/cleaning_*.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..paths import ProjectPaths

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

FEATURES = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
SHORT_GAP = 3          # 当前连续缺失的前 3 步用因果前值填补
KNN_K = 12             # 前 196 天训练段两个缺失场景均最优
TRAIN_DAYS = 196       # 8:1:1 按时间划分（前 196 天训练）
STUCK_WIN = 6          # 连续 6 步（1 h）同值视为传感器卡死
CURT_WSPD = 5.0        # 限电判据的最低风速
SIGMA = 3.0            # 残差异常阈值（稳健 σ 倍数）
RNG = np.random.default_rng(0)


# ─────────────────────────── 数据载入与宽表 ───────────────────────────

def load(paths: ProjectPaths) -> tuple:
    df = pd.read_csv(paths.raw / "sdwpf_245days_v1.csv")
    df["ts"] = pd.to_timedelta(df["Day"] - 1, unit="D") + pd.to_timedelta(df["Tmstamp"] + ":00")
    df["TurbID"] = df["TurbID"].astype(int)
    loc = pd.read_csv(paths.raw / "sdwpf_baidukddcup2022_turb_location.csv", encoding="utf-8-sig").set_index("TurbID")
    tids = sorted(loc.index.to_numpy())
    wide = {c: df.pivot(index="ts", columns="TurbID", values=c)[tids].to_numpy(dtype=float).copy()
            for c in FEATURES}
    order = df.pivot(index="ts", columns="TurbID", values="TurbID")[tids].columns
    ts_index = df.pivot(index="ts", columns="TurbID", values="TurbID")[tids].index
    return df, loc.loc[tids], wide, ts_index, order


# ─────────────────────────── 缺失填补 ───────────────────────────

def fill_short(X: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """只用过去观测填当前连续缺失的前 SHORT_GAP 步。"""
    out = X.copy()
    T, N = X.shape
    for j in range(N):
        isna = missing[:, j]
        if not np.any(isna):
            continue
        positions = np.arange(T)
        last_observed = np.maximum.accumulate(np.where(isna, -1, positions))
        fillable = isna & (last_observed >= 0) & (
            positions - last_observed <= SHORT_GAP)
        previous = pd.Series(np.where(isna, np.nan, X[:, j])).ffill().to_numpy()
        out[fillable, j] = previous[fillable]
    return out


def knn_fill(Xm: np.ndarray, D: np.ndarray, k: int, todo: np.ndarray) -> np.ndarray:
    """对 todo 标记的格子用 k 近邻（距离倒数平方加权）填补。"""
    out = Xm.copy()
    T, N = Xm.shape
    order = np.argsort(D, axis=1)
    for j in range(N):
        rows = np.where(todo[:, j])[0]
        if len(rows) == 0:
            continue
        neigh = order[j, 1 : k + 1]
        vals = Xm[:, neigh][rows]                       # 邻居同时刻值
        w = np.tile((1.0 / D[j, neigh] ** 2), (len(rows), 1))
        ok = np.isfinite(vals)
        w = w * ok
        den = w.sum(axis=1)
        num = np.nansum(vals * w, axis=1)
        fill = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        out[rows, j] = fill
    return out


def impute(X: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """三级递补主流程。返回（填补后数组, 填补位置 mask）。"""
    missing = ~np.isfinite(X)
    short = fill_short(X, missing)
    still = ~np.isfinite(short)
    filled = knn_fill(short, D, KNN_K, still & missing)
    imputed = missing & np.isfinite(filled)
    return filled, imputed


# ─────────────────────────── 功率曲线与残差 ───────────────────────────

def fit_power_curve(Wspd: np.ndarray, Patv: np.ndarray, excl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐机拟合分风速箱中位数功率曲线。excl：需排除的行（缺失/停机）。

    返回（curve_edges 中心风速表, curves [N × nbin]）。
    """
    edges = np.arange(0.0, 26.01, 0.5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    N = Wspd.shape[1]
    curves = np.full((N, len(centers)), np.nan)
    for j in range(N):
        v, p = Wspd[:, j], Patv[:, j]
        ok = np.isfinite(v) & np.isfinite(p) & ~excl[:, j] & (p > 0) & (v >= 0.25) & (v < 26)
        if ok.sum() < 500:
            continue
        idx = np.digitize(v[ok], edges) - 1
        idx = np.clip(idx, 0, len(centers) - 1)
        for b in range(len(centers)):
            sel = idx == b
            if sel.sum() >= 30:
                curves[j, b] = np.median(p[ok][sel])
        # 空箱（高风速段样本不足）用有效箱线性插值平滑，避免端部毛刺
        valid = np.isfinite(curves[j])
        if valid.sum() >= 2:
            curves[j] = np.interp(centers, centers[valid], curves[j][valid])
    return centers, curves


def curve_interp(v: np.ndarray, centers: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """在风速箱中心上线性插值出功率曲线值；区间外用端值延伸，无效风速 NaN。"""
    out = np.interp(v, centers, curve, left=curve[0], right=curve[-1])
    out[~np.isfinite(v)] = np.nan
    return out


def residual_flags(Wspd: np.ndarray, Patv: np.ndarray, resid: np.ndarray,
                   f_fault: np.ndarray, train_rows: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Fit residual scales on training rows and classify every row with them."""
    m_curtail = np.zeros_like(f_fault)
    m_outlier = np.zeros_like(f_fault)
    edges = np.arange(0.0, 26.01, 1.0)
    for j in range(Wspd.shape[1]):
        v, r = Wspd[:, j], resid[:, j]
        ok = np.isfinite(v) & np.isfinite(r) & (Patv[:, j] > 0) & ~f_fault[:, j]
        train_ok = ok & train_rows
        train_resid = r[train_ok]
        if train_resid.size == 0:
            continue
        mad_all = max(float(np.median(np.abs(train_resid - np.median(train_resid))) *
                            1.4826), 5.0)
        sigma = np.full(len(v), mad_all)
        for b in range(len(edges) - 1):
            in_bin = (v >= edges[b]) & (v < edges[b + 1])
            training_bin = train_ok & in_bin
            if training_bin.sum() >= 100:
                rb = r[training_bin]
                sig = np.median(np.abs(rb - np.median(rb))) * 1.4826
                sigma[in_bin] = max(sig, 5.0)
        z = r / np.where(sigma > 0, sigma, np.nan)
        cand = ok & np.isfinite(z)
        m_curtail[:, j] = cand & (z <= -SIGMA) & (v >= CURT_WSPD)
        m_outlier[:, j] = cand & (np.abs(z) > SIGMA) & (v >= 3.0) & ~m_curtail[:, j]
    return m_curtail, m_outlier


# ─────────────────────────── 主流程 ───────────────────────────

def main(paths: ProjectPaths = None) -> None:
    paths = paths or ProjectPaths.resolve()
    paths.processed.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)
    df, loc, wide, ts_index, tids = load(paths)
    T, N = wide["Wspd"].shape
    train_rows = np.asarray(ts_index < pd.Timedelta(days=TRAIN_DAYS))
    if not train_rows.any() or train_rows.all():
        raise ValueError("Training boundary must leave both training and future rows")
    xy = loc[["x", "y"]].to_numpy()
    D = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)

    L: list[str] = ["# SDWPF 清洗报告", "", "生成：ClawsGO Science Agent · 2026-09-24", ""]

    # ── P1 物理范围硬规则 ──
    m_missing0 = {c: ~np.isfinite(wide[c]) for c in FEATURES}       # 原始缺失
    n_out = {}
    for c in ["Etmp", "Itmp"]:
        bad = np.isfinite(wide[c]) & ((wide[c] < -40) | (wide[c] > 60))
        wide[c][bad] = np.nan
        n_out[c] = int(bad.sum())
    bad = np.isfinite(wide["Wspd"]) & ((wide["Wspd"] < 0) | (wide["Wspd"] > 40))
    wide["Wspd"][bad] = np.nan; n_out["Wspd"] = int(bad.sum())
    # Wdir 回绕到 [-180,180)
    wdir = wide["Wdir"]
    n_wrap = int(np.isfinite(wdir).sum() and ((np.abs(wdir[np.isfinite(wdir)]) > 180).sum()))
    wrapped = ((wdir + 180.0) % 360.0) - 180.0
    wide["Wdir"] = wrapped
    # Patv<0 截 0（怠机耗电）
    neg = wide["Patv"] < 0
    n_neg = int(neg.sum())
    wide["Patv"][neg] = 0.0
    L.append("## P1 物理范围硬规则\n")
    L.append(f"- 温度超 [−40,60] ℃ 置缺失：Etmp {n_out['Etmp']:,} 行、Itmp {n_out['Itmp']:,} 行")
    L.append(f"- Wspd 物理界外置缺失：{n_out['Wspd']} 行")
    L.append(f"- Wdir 回绕修正：{n_wrap:,} 行")
    L.append(f"- Patv<0 截 0（怠机耗电）：{n_neg:,} 行\n")

    # ── 缺失三级递补（先对所有特征做，含被 P1 置缺的）──
    imputed_any = {}
    for c in FEATURES:
        wide[c], imp = impute(wide[c], D)
        imputed_any[c] = imp
    total_imp = sum(int(v.sum()) for v in imputed_any.values())
    still_missing = {c: int((~np.isfinite(wide[c])).sum()) for c in FEATURES}
    L.append("## 缺失三级递补\n")
    L.append(f"- 当前连续缺失前 {SHORT_GAP} 步用因果前值填补；第 {SHORT_GAP + 1} 步起空间 kNN（k={KNN_K}，距离倒数平方加权）")
    L.append(f"- 10 个特征合计填补 {total_imp:,} 格；填补后仍未填补 {sum(still_missing.values()):,} 格")
    L.append("- 各列残余缺失：" + "，".join(f"{c} {still_missing[c]:,}" for c in FEATURES) + "\n")

    # ── P3a 停机/故障 flag（在残差分析前，用于排除功率曲线拟合）──
    f_fault = np.isfinite(wide["Patv"]) & np.isfinite(wide["Wspd"]) & (wide["Patv"] <= 0) & (wide["Wspd"] >= 3.0)

    # ── P2 功率曲线残差 ──
    centers, curves = fit_power_curve(wide["Wspd"][train_rows],
                                     wide["Patv"][train_rows],
                                     excl=f_fault[train_rows])
    resid = np.full_like(wide["Patv"], np.nan)
    for j in range(N):
        ok = np.isfinite(wide["Patv"][:, j]) & np.isfinite(wide["Wspd"][:, j])
        v = wide["Wspd"][ok, j]
        pc = curve_interp(v, centers, curves[j])
        r = wide["Patv"][ok, j] - pc
        resid[np.where(ok)[0], j] = r
    # 分风速箱稳健 σ 仅从训练段估计，随后固定阈值应用到未来段。
    m_curtail, m_outlier = residual_flags(
        wide["Wspd"], wide["Patv"], resid, f_fault, train_rows)
    # 离群 Patv 置缺失并重补
    n_out_patv = int(m_outlier.sum())
    wide["Patv"][m_outlier] = np.nan
    wide["Patv"], imp2 = impute(wide["Patv"], D)
    L.append("## P2 功率曲线残差异常\n")
    L.append(f"- 功率曲线与逐机×分箱稳健 σ 仅由前 {TRAIN_DAYS} 天拟合，再应用到全段")
    L.append(f"- 残差 |z|>{SIGMA} → 离群：{n_out_patv:,} 行，Patv 已置缺失并重补")
    L.append(f"- 残差 z≤−{SIGMA} 且 Wspd≥{CURT_WSPD} → 限电/降额 flag（f_curtail）：{int(m_curtail.sum()):,} 行\n")

    # ── P3b 卡死与全场停机 ──
    # 卡死检测仅用于 Wspd：风速计卡死是经典 SCADA 故障；Ndir 长时间恒定是正常对风行为，
    # Etmp/Itmp 分辨率粗连续同值常见，均不作卡死判定
    f_stuck = np.zeros_like(f_fault)
    for j in range(N):
        s = pd.Series(wide["Wspd"][:, j])
        grp = (s != s.shift()).cumsum()
        size = s.groupby(grp).transform("size").to_numpy()
        f_stuck[:, j] = (size >= STUCK_WIN) & np.isfinite(s.to_numpy())
    n_stuck = int(f_stuck.sum())
    day = (ts_index.days.to_numpy()) + 1
    daily_fault = pd.DataFrame({"day": day, "fault": f_fault.mean(axis=1)}).groupby("day")["fault"].mean()
    farm_days = [int(d) for d in daily_fault[daily_fault > 0.5].index]
    f_farm = np.isin(day, farm_days)
    L.append("## P3 工况异常\n")
    L.append(f"- 传感器卡死（Wspd，连续 ≥{STUCK_WIN} 步同值）：{n_stuck:,} 格，打 f_stuck flag（不改动数值）；"
             "Ndir 恒定属正常对风、温度恒定属分辨率粗，不作卡死判定")
    L.append(f"- 全场停机日（当日停机机组×时刻格占比 >50%）：Day {list(farm_days)}，打 f_farm flag（训练目标与主表评分排除）\n")

    # ── 汇总输出 ──
    # ── 汇总输出（宽表 (T, N) 行主序展开为长表）──
    out = pd.DataFrame({
        "ts": np.repeat(ts_index.to_numpy(), N),
        "Day": np.repeat(day, N),
        "TurbID": np.tile(np.array(tids), T),
    })
    for c in FEATURES:
        out[c] = wide[c].ravel()
    # 风向 sin/cos 特征
    rad = np.deg2rad(wide["Wdir"])
    out["Wsin"] = np.sin(rad).ravel()
    out["Wcos"] = np.cos(rad).ravel()
    out["m_missing"] = np.stack([m_missing0[c] for c in FEATURES], axis=2).any(axis=2).ravel()
    out["m_imputed"] = np.stack([imputed_any[c] for c in FEATURES], axis=2).any(axis=2).ravel() | imp2.ravel()
    # Preserve observation validity for auxiliary targets without changing the
    # existing row-level flags used by the published evaluation protocol.
    out["m_missing_wspd"] = m_missing0["Wspd"].ravel()
    out["m_imputed_wspd"] = imputed_any["Wspd"].ravel()
    out["m_missing_patv"] = m_missing0["Patv"].ravel()
    out["m_imputed_patv"] = (imputed_any["Patv"] | imp2).ravel()
    out["m_outlier"] = m_outlier.ravel()
    out["f_fault"] = f_fault.ravel()
    out["f_curtail"] = m_curtail.ravel()
    out["f_stuck"] = f_stuck.ravel()
    out["f_farm"] = np.repeat(f_farm, N)
    parquet_path = paths.processed / "sdwpf_clean.parquet"
    out.to_parquet(parquet_path, index=False)

    # train 段 min/max（供训练脚本归一化，防泄漏）
    train_end = pd.Timedelta(days=TRAIN_DAYS)
    tr = out[out["ts"] < train_end]
    meta = {
        "train_days": TRAIN_DAYS,
        "knn_k": KNN_K,
        "short_gap": SHORT_GAP,
        "scale": {c: {"min": float(tr[c].min()), "max": float(tr[c].max())}
                  for c in FEATURES + ["Wsin", "Wcos"]},
    }
    (paths.processed / "sdwpf_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 验证：功率曲线紧致度（清洗前后）+ 图 ──
    # 前：用原始负功率截 0 后、未剔除限电/离群的残差 MAD；后：剔除 flag 后
    ok_before = np.isfinite(resid) & (wide["Patv"] > 0)
    mad_before = np.nanmedian(np.abs(resid[ok_before] - np.nanmedian(resid[ok_before]))) * 1.4826
    ok_after = ok_before & ~m_curtail & ~m_outlier & ~f_fault
    mad_after = np.nanmedian(np.abs(resid[ok_after] - np.nanmedian(resid[ok_after]))) * 1.4826
    L.append("## 验证与 flag 汇总\n")
    L.append(f"- 功率曲线残差 MAD：清洗前 {mad_before:.1f} kW → 剔除限电/离群后 {mad_after:.1f} kW（紧致度提升 {100*(1-mad_after/mad_before):.0f}%）")
    n_rows = len(out)
    for f in ["m_missing", "m_imputed", "m_outlier", "f_fault", "f_curtail", "f_stuck", "f_farm"]:
        L.append(f"- `{f}`：{int(out[f].sum()):,} 行（{out[f].mean():.2%}）")
    L.append(f"\n输出：`data/processed/sdwpf_clean.parquet`（{n_rows:,} 行）+ `sdwpf_meta.json`（train 段 min/max）")

    # 图 1：清洗前后 1 号机功率曲线
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (v, p, ttl) in zip(axes, [
        (wide["Wspd"][:, 0], wide["Patv"][:, 0], "清洗后（含限电行）"),
        (wide["Wspd"][:, 0][ok_after[:, 0]], wide["Patv"][:, 0][ok_after[:, 0]], "清洗后（剔除限电/离群/停机）"),
    ]):
        sel = np.isfinite(v) & np.isfinite(p) & (p >= 0) & (v < 30)
        ax.hexbin(v[sel], p[sel], gridsize=80, cmap="viridis", bins="log", mincnt=1)
        ax.plot(centers, curves[0], color="#D55E00", lw=1.5, label="拟合功率曲线")
        ax.set_xlabel("风速 (m/s)"); ax.set_ylabel("功率 (kW)"); ax.set_title(f"1 号机：{ttl}")
        ax.legend()
    fig.tight_layout(); fig.savefig(paths.figures / "cleaning_powercurve.png", dpi=150); plt.close(fig)

    # 图 2：各 flag 逐日占比（折线并列对比）
    flags = ["f_fault", "f_curtail", "m_imputed", "m_outlier"]
    cmap = {"f_fault": "#8C6D31", "f_curtail": "#D55E00", "m_imputed": "#56B4E9", "m_outlier": "#CC79A7"}
    fig, ax = plt.subplots(figsize=(9, 3))
    for f in flags:
        per_day = pd.DataFrame({"day": day, "v": out[f].to_numpy().reshape(T, N).mean(axis=1)}).groupby("day")["v"].mean()
        ax.plot(per_day.index, per_day.values, color=cmap[f], lw=1.0, label=f)
    ax.set_xlabel("天"); ax.set_ylabel("行占比"); ax.set_title("各标记逐日占比")
    ax.legend(ncol=4)
    fig.tight_layout(); fig.savefig(paths.figures / "cleaning_flags.png", dpi=150); plt.close(fig)

    (paths.reports / "cleaning_report.md").write_text("\n".join(L), encoding="utf-8")
    print("清洗完成 →", parquet_path)
