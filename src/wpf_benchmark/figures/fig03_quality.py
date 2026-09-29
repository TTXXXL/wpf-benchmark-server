"""图 3：数据质量；caption: Curtailment and data flags motivate quality-aware evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import PowerNorm

from ..paths import ProjectPaths
from .io import MissingMaterial
from .style import DOUBLE, apply_figure_style, save_figure


FLAGS = ("f_fault", "f_curtail", "m_missing", "m_outlier")
FLAG_LABELS = ("Fault", "Curtailment", "Missing", "Outlier")


def render(paths: ProjectPaths, out: Path, max_points: int = 50000) -> List[Path]:
    source = paths.processed / "sdwpf_clean.parquet"
    if not source.is_file():
        raise MissingMaterial("clean Parquet not found")
    columns = ["Day", "Wspd", "Patv"] + list(FLAGS) + ["m_imputed", "f_farm"]
    frame = pd.read_parquet(source, columns=columns)
    rng = np.random.default_rng(20260924)
    other_flags = ("f_fault", "m_missing", "m_imputed", "m_outlier", "f_farm")
    otherwise_clean = ~frame[list(other_flags)].any(axis=1)
    measured = frame["Wspd"].notna() & frame["Patv"].notna()
    regular = frame.loc[(~frame["f_curtail"]) & otherwise_clean & measured,
                        ["Wspd", "Patv"]]
    curtailed = frame.loc[frame["f_curtail"] & otherwise_clean & measured,
                          ["Wspd", "Patv"]]
    def sample(data: pd.DataFrame, limit: int) -> pd.DataFrame:
        if len(data) <= limit:
            return data
        return data.iloc[rng.choice(len(data), limit, replace=False)]
    regular = sample(regular, max_points)
    curtailed = sample(curtailed, min(5000, max_points // 10))
    daily = frame.groupby("Day")[list(FLAGS)].mean()
    apply_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
    ax = axes[0]
    ax.hexbin(regular["Wspd"], regular["Patv"], gridsize=45,
              cmap="Greys", bins="log", mincnt=1, linewidths=0)
    flagged = ax.scatter(curtailed["Wspd"], curtailed["Patv"], s=2, alpha=0.25,
                         c="#D55E00", rasterized=True)
    ax.set(xlabel="Wind speed (m/s)", ylabel="Power (kW)", title="a  Power and curtailment")
    ax.legend([Patch(facecolor="#888888"), flagged],
              ["Unflagged density", "Curtailment flag"], frameon=False, loc="upper left")
    rates = 100 * daily[list(FLAGS)].to_numpy().T
    heat = axes[1].imshow(rates, aspect="auto", interpolation="nearest", origin="upper",
                          extent=(daily.index.min() - 0.5, daily.index.max() + 0.5,
                                  len(FLAGS) - 0.5, -0.5), cmap="Blues",
                          norm=PowerNorm(gamma=0.45, vmin=0, vmax=100))
    axes[1].set(title="b  Data-quality flags", xlabel="Day")
    axes[1].set_yticks(range(len(FLAGS)), FLAG_LABELS)
    fig.colorbar(heat, ax=axes[1], label="Flagged rows (%)", shrink=0.75)
    return save_figure(fig, out, "fig03_quality")
