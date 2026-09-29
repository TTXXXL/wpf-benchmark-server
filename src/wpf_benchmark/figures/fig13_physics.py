"""图 13：经验功率曲线和输出范围；caption: Physical-plausibility checks expose persistence errors."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ..paths import ProjectPaths
from .io import MissingMaterial, load_metrics, model_groups
from .style import DOUBLE, SINGLE, apply_figure_style, model_style, save_figure


def render(paths: ProjectPaths, out: Path) -> List[Path]:
    groups = model_groups(paths)
    if not groups:
        raise MissingMaterial("main evaluation JSON not found")
    names = sorted(groups)
    rows = {name: [load_metrics(paths, record)["D_physics"] for record in groups[name]]
            for name in names}
    apply_figure_style()
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE[0], 4.5) if len(names) > 4
                             else (SINGLE[0], 4.0))
    x = np.arange(len(names))
    violations = [np.mean([row["violation_rate_pct"] for row in rows[name]]) for name in names]
    truth = np.mean([row["truth_violation_rate_pct"] for row in rows[names[0]]])
    axes[0].bar(x, violations, color=[model_style(name).color for name in names], width=0.55)
    axes[0].axhline(truth, color="black", linestyle=":", linewidth=1,
                    label="Truth reference ({:.1f}%)".format(truth))
    axes[0].set(ylabel="Curve deviation (%)", title="a  Power-curve consistency")
    axes[0].set_ylim(0, max(violations + [truth]) * 1.38)
    axes[0].legend(frameon=False, loc="upper right")
    width = 0.35
    neg = [np.mean([row["neg_rate_pct"] for row in rows[name]]) for name in names]
    over = [np.mean([row["over_rated_rate_pct"] for row in rows[name]]) for name in names]
    axes[1].bar(x - width / 2, neg, width, label="Negative power", color="#0072B2")
    axes[1].bar(x + width / 2, over, width, label="Above rated", color="#E69F00")
    if all(value == 0 for value in neg):
        axes[1].scatter(x - width / 2, neg, s=10, color="#0072B2", zorder=3)
    axes[1].set(ylabel="Output-range rate (%)", title="b  Output bounds")
    axes[1].set_ylim(0, max(neg + over + [0.001]) * 1.45)
    axes[1].legend(frameon=False, loc="upper right", ncol=2)
    short = {"climatology": "Climatology",
             "farm_mean_persistence": "Farm mean\npersist.",
             "gbdt": "GBDT", "linear": "Linear",
             "persistence": "Persistence",
             "seasonal_persistence": "Seasonal\npersist.",
             "trend_persistence": "Trend\npersist.",
             "lstm_seq2seq": "LSTM enc.", "patchtst": "PatchTST-style",
             "agcrn": "AGCRN-style"}
    for ax in axes:
        labels = [short.get(name, model_style(name).label) for name in names]
        ax.set_xticks(x, labels, rotation=0 if len(names) > 4 else 15)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.4)
    return save_figure(fig, out, "fig13_physics")
