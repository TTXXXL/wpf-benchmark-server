"""图 8：预测密度；caption: Point-wise density and wind-conditioned errors expose systematic patterns."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ..paths import ProjectPaths
from .io import MissingMaterial, aligned_array_runs
from .style import DOUBLE, apply_figure_style, save_figure


def render(paths: ProjectPaths, out: Path, model: str = "", max_points: int = 200000) -> List[Path]:
    runs = aligned_array_runs(paths)
    if model:
        if model not in runs:
            raise MissingMaterial("arrays for model {} not found".format(model))
        runs = {model: runs[model]}
    outputs: List[Path] = []
    apply_figure_style()
    for name, (_, arrays) in runs.items():
        if "wind_speed" not in arrays:
            raise MissingMaterial("arrays lack wind_speed; rerun with --save-arrays")
        mask = arrays["valid"] & np.isfinite(arrays["wind_speed"])
        truth = arrays["truth"][mask]
        pred = arrays["forecasts"][mask]
        wind = arrays["wind_speed"][mask]
        if not len(truth):
            raise MissingMaterial("no valid point predictions")
        if len(truth) > max_points:
            choice = np.random.default_rng(20260924).choice(len(truth), max_points, replace=False)
            truth, pred, wind = truth[choice], pred[choice], wind[choice]
        fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
        lim = float(max(np.max(truth), np.max(pred)))
        hb = axes[0].hexbin(truth, pred, gridsize=55, mincnt=1, bins="log", cmap="viridis")
        axes[0].plot([0, lim], [0, lim], "--", color="black", lw=0.8)
        axes[0].set(xlabel="Observed power (kW)", ylabel="Forecast power (kW)",
                    title="a  Forecast vs observed")
        fig.colorbar(hb, ax=axes[0], label="Count (log)")
        edges = np.linspace(0, max(25, float(np.nanmax(wind))), 11)
        centers, medians, lower, upper = [], [], [], []
        error = pred - truth
        for lo, hi in zip(edges[:-1], edges[1:]):
            selected = (wind >= lo) & (wind < hi)
            if selected.sum() < 5:
                continue
            centers.append((lo + hi) / 2)
            lower.append(float(np.quantile(error[selected], 0.25)))
            medians.append(float(np.median(error[selected])))
            upper.append(float(np.quantile(error[selected], 0.75)))
        axes[1].fill_between(centers, lower, upper, color="#56B4E9", alpha=0.25,
                             label="25th–75th percentile")
        axes[1].plot(centers, medians, color="#0072B2", marker="o", markersize=2,
                     label="Median error")
        axes[1].axhline(0, color="black", lw=0.7)
        axes[1].set(xlabel="Measured wind speed (m/s)", ylabel="Forecast error (kW)",
                    title="b  Error by wind speed")
        axes[1].legend(frameon=False, loc="upper right")
        outputs += save_figure(fig, out, "fig08_density_{}".format(name))
    return outputs
