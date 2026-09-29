"""图 6：逐预测步误差；caption: Forecast error rises with lead time for persistence."""
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
    apply_figure_style()
    fig, ax = plt.subplots(figsize=DOUBLE if len(groups) > 4 else SINGLE)
    for model, records in sorted(groups.items()):
        runs = [load_metrics(paths, record) for record in records]
        steps = sorted((int(step) for step in runs[0]["B_per_horizon"]), key=int)
        values = np.array([[run["B_per_horizon"][str(step)]["MAE_kW"] for step in steps]
                           for run in runs], dtype=float)
        style = model_style(model)
        ax.plot(steps, values.mean(axis=0), label=style.label,
                color=style.color, linestyle=style.linestyle, marker="o", markersize=2.5)
        if len(runs) >= 2:
            std = values.std(axis=0, ddof=1)
            ax.fill_between(steps, values.mean(axis=0) - std,
                            values.mean(axis=0) + std, color=style.color, alpha=0.15)
    ax.set(xlabel="Forecast step (10 min each)", ylabel="MAE (kW)")
    ax.set_xticks(steps)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.4)
    if len(groups) > 4:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25),
                  ncol=3, fontsize=7)
    else:
        ax.legend(frameon=False, loc="upper left")
    return save_figure(fig, out, "fig06_horizon_mae")
