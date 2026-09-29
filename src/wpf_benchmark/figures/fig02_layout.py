"""图 2：场站布局；caption: Turbine coordinates show the spatial extent of the SDWPF farm."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..paths import ProjectPaths
from .io import MissingMaterial
from .style import SINGLE, apply_figure_style, save_figure


def _dominant_direction(paths: ProjectPaths) -> Optional[float]:
    source = paths.processed / "sdwpf_clean.parquet"
    if not source.is_file():
        return None
    direction = pd.read_parquet(source, columns=["Wdir"])["Wdir"].to_numpy(dtype=float)
    direction = direction[np.isfinite(direction)][::20]
    if not len(direction):
        return None
    radians = np.deg2rad(direction)
    return float(np.rad2deg(np.arctan2(np.sin(radians).mean(),
                                       np.cos(radians).mean())) % 360)


def render(paths: ProjectPaths, out: Path, wind_direction: Optional[float] = None,
           scale_m: float = 500.0) -> List[Path]:
    source = paths.raw / "sdwpf_baidukddcup2022_turb_location.csv"
    if not source.is_file():
        raise MissingMaterial("turbine location CSV not found")
    if scale_m <= 0:
        raise ValueError("scale_m must be positive")
    frame = pd.read_csv(source, encoding="utf-8-sig")
    if not {"TurbID", "x", "y"}.issubset(frame.columns):
        raise ValueError("Location CSV needs TurbID,x,y")
    apply_figure_style()
    fig, ax = plt.subplots(figsize=(SINGLE[0], 3.2))
    ax.scatter(frame["x"], frame["y"], c="#777777", s=8, linewidths=0)
    ax.set(xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal", adjustable="box")
    xmin, xmax = float(frame["x"].min()), float(frame["x"].max())
    ymin, ymax = float(frame["y"].min()), float(frame["y"].max())
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - 0.12 * dx, xmax + 0.12 * dx)
    ax.set_ylim(ymin - 0.12 * dy, ymax + 0.25 * dy)
    scale_x = xmax - scale_m - 0.04 * dx
    scale_y = ymin - 0.08 * dy
    ax.plot([scale_x, scale_x + scale_m], [scale_y, scale_y], color="black", lw=1.2)
    ax.text(scale_x + scale_m / 2, scale_y + 0.015 * dy, "{} m".format(int(scale_m)),
            ha="center", fontsize=7)
    direction = wind_direction if wind_direction is not None else _dominant_direction(paths)
    if direction is not None:
        # 风向是气象约定的来向；箭头指向气流去向。
        radians = np.deg2rad(direction)
        arrow_length = min(dx, dy) * 0.17
        x0, y0 = xmin + 0.22 * dx, ymax + 0.18 * dy
        ax.annotate("", xy=(x0 - np.sin(radians) * arrow_length,
                             y0 - np.cos(radians) * arrow_length), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 1.2})
        ax.text(x0 + 0.04 * dx, y0 + 0.01 * dy, "Prevailing wind", fontsize=7,
                color="#0072B2", va="bottom")
    ax.text(0.02, 0.02, "n = {} turbines".format(len(frame)), transform=ax.transAxes,
            va="bottom", fontsize=7)
    return save_figure(fig, out, "fig02_layout")
