"""论文图件统一尺寸、字体、颜色与保存格式。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SINGLE = (3.54, 2.4)
DOUBLE = (7.48, 3.0)


@dataclass(frozen=True)
class Style:
    color: str
    linestyle: str
    label: str


MODEL_STYLE = {
    "ours": Style("#D55E00", "-", "Ours"),
    "ours_no_consist": Style("#0072B2", "-", "Ours w/o consistency"),
    "ours_no_wake": Style("#009E73", "-", "Ours w/o wake prior"),
    "barest": Style("#CC79A7", "-", "Backbone only"),
    "persistence": Style("#999999", "--", "Persistence"),
    "seasonal_persistence": Style("#E69F00", "-", "Seasonal persistence"),
    "climatology": Style("#CC79A7", "-", "Climatology"),
    "trend_persistence": Style("#009E73", "-", "Trend persistence"),
    "farm_mean_persistence": Style("#56B4E9", "-", "Farm mean persistence"),
    "linear": Style("#0072B2", "-", "Linear"),
    "gbdt": Style("#AA4499", "-", "Gradient boosting"),
    "lstm_seq2seq": Style("#882255", "-", "LSTM encoder"),
    "patchtst": Style("#44AA99", "-", "PatchTST-inspired"),
    "agcrn": Style("#332288", "-", "AGCRN-inspired"),
    "baseline1": Style("#0072B2", "-", "Baseline 1"),
    "baseline2": Style("#56B4E9", "-", "Baseline 2"),
    "baseline3": Style("#009E73", "-", "Baseline 3"),
}
FALLBACK = ("#E69F00", "#CC79A7", "#0072B2", "#009E73", "#56B4E9", "#F0E442")


def model_style(model: str) -> Style:
    """未知模型按名称稳定映射；新正式模型应加入 MODEL_STYLE。"""
    if model in MODEL_STYLE:
        return MODEL_STYLE[model]
    index = int(hashlib.sha256(model.encode("utf-8")).hexdigest()[:8], 16) % len(FALLBACK)
    return Style(FALLBACK[index], "-", model.replace("_", " ").title())


def apply_figure_style() -> None:
    """figure-style 技能与项目规格共用的投稿图尺寸和文字层级。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def save_figure(fig: plt.Figure, out: Path, stem: str) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    paths = [out / (stem + ".pdf"), out / (stem + ".png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths
