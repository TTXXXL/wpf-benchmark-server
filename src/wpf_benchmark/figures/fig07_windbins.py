"""图 7：风速分段误差；caption: Errors differ across measured wind-speed regimes."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ..paths import ProjectPaths
from .io import MissingMaterial, load_metrics, model_groups
from .style import DOUBLE, SINGLE, apply_figure_style, model_style, save_figure


def render(paths: ProjectPaths, out: Path, style: str = "bar") -> List[Path]:
    groups = model_groups(paths)
    if not groups:
        raise MissingMaterial("main evaluation JSON not found")
    if style not in ("bar", "heatmap"):
        raise ValueError("style must be bar or heatmap")
    if len(groups) > 5 and style == "bar":
        style = "heatmap"
    data = {model: [load_metrics(paths, record) for record in records]
            for model, records in sorted(groups.items())}
    bins = list(next(iter(data.values()))[0]["B_wind_bins"])
    def mean_for(runs, bin_name):
        values = [run["B_wind_bins"][bin_name]["MAE_kW"] for run in runs
                  if run["B_wind_bins"][bin_name]["MAE_kW"] is not None]
        return float(np.mean(values)) if values else float("nan")
    means = np.array([[mean_for(runs, bin_name) for bin_name in bins]
                      for runs in data.values()])
    counts = np.array([[min(run["B_wind_bins"][bin_name]["n"] for run in runs)
                        for bin_name in bins] for runs in data.values()])
    apply_figure_style()
    names = list(data)
    if style == "heatmap":
        fig, ax = plt.subplots(figsize=(SINGLE[0], max(2.4, 0.35 * len(names) + 1.0)))
        img = ax.imshow(np.ma.masked_invalid(means), cmap="Blues", aspect="auto", vmin=0)
        ax.set_xticks(range(len(bins)), bins, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), [model_style(name).label for name in names])
        for i in range(len(names)):
            for j in range(len(bins)):
                label = ("n.d." if not np.isfinite(means[i, j]) else
                         "{:.0f}{}".format(means[i, j], "*" if counts[i, j] < 1000 else ""))
                ax.text(j, i, label,
                        ha="center", va="center", fontsize=6.5,
                        color="white" if np.isfinite(means[i, j]) and
                        means[i, j] > np.nanmax(means) * 0.55 else "black")
        fig.colorbar(img, ax=ax, label="MAE (kW)", shrink=0.85)
    else:
        fig, ax = plt.subplots(figsize=DOUBLE if len(names) > 2 else SINGLE)
        x = np.arange(len(bins))
        width = 0.78 / len(names)
        for i, name in enumerate(names):
            positions = x - 0.39 + width * (i + 0.5)
            heights = np.nan_to_num(means[i], nan=0.0)
            bars = ax.bar(positions, heights, width=width, label=model_style(name).label,
                          color=model_style(name).color, alpha=0.85)
            for j, bar in enumerate(bars):
                if counts[i, j] < 1000:
                    bar.set_hatch("///")
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        "{:.0f}".format(means[i, j]) if np.isfinite(means[i, j]) else "n.d.",
                        ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x, bins, rotation=25, ha="right")
        ax.set_ylabel("MAE (kW)")
        maximum = np.nanmax(means) if np.isfinite(means).any() else 1
        ax.set_ylim(0, maximum * 1.2)
        ax.legend(frameon=False, loc="upper right")
    ax.set_xlabel("Measured wind speed (m/s); hatch/* marks n < 1k")
    return save_figure(fig, out, "fig07_windbins")
