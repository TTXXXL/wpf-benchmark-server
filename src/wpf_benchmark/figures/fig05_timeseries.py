"""图 5：典型日曲线；caption: Forecast traces reveal ordinary, ramping and curtailed regimes."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ..paths import ProjectPaths
from .io import MissingMaterial, aligned_array_runs
from .style import DOUBLE, apply_figure_style, model_style, save_figure


DAY_NS = 86400 * 10**9


def render(paths: ProjectPaths, out: Path) -> List[Path]:
    runs = aligned_array_runs(paths)
    ref = next(iter(runs.values()))[1]
    if "wind_speed" not in ref or "curtail_mask" not in ref:
        raise MissingMaterial("arrays lack wind_speed or curtail_mask; rerun with --save-arrays")
    valid = ref["valid"][:, :, 0]
    times = ref["times"][1:len(valid) + 1]
    days = times // DAY_NS + 1
    unique = np.unique(days)
    if not len(unique):
        raise MissingMaterial("test arrays contain no days")
    truth = np.where(valid, ref["truth"][:, :, 0], 0).sum(axis=1) / 1000
    energy = np.array([truth[days == day].sum() / 6 for day in unique])
    normal = int(unique[np.argmin(np.abs(energy - np.median(energy)))])
    ramp = ref["ramp_mask"][:, :, 0].sum(axis=1)
    event_counts = np.array([ramp[days == day].sum() for day in unique])
    wind = ref["wind_speed"][:, :, 0]
    mean_wind = np.array([np.nanmean(wind[days == day]) for day in unique])
    ramp_day = int(unique[max(range(len(unique)), key=lambda i: (event_counts[i], mean_wind[i]))])
    curtail = ref["curtail_mask"][:, :, 0]
    curtail_rate = np.array([curtail[days == day].mean() for day in unique])
    curtailed_day = int(unique[np.argmax(curtail_rate)])
    selected = [("Typical day", normal), ("Ramp day", ramp_day),
                ("Curtailment day", curtailed_day)]
    apply_figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE[0], 3.0), sharey=True)
    for panel, (title, day) in enumerate(selected):
        ax = axes[panel]
        take = days == day
        hour = (times[take] % DAY_NS) / (3600 * 10**9)
        ax.plot(hour, truth[take], color="black", label="Truth", linewidth=1.3)
        for model, (_, arrays) in runs.items():
            pred = np.where(valid, arrays["forecasts"][:, :, 0], 0).sum(axis=1) / 1000
            style = model_style(model)
            ax.plot(hour, pred[take], label=style.label, color=style.color,
                    linestyle=style.linestyle, linewidth=0.9)
        if panel == 2:
            for x in hour[curtail[take].any(axis=1)]:
                ax.axvspan(x - 1/12, x + 1/12, color="#D55E00", alpha=0.12, lw=0)
            ax.text(0.98, 0.98, "Shading: curtailment", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="#D55E00")
        ax.set(title="{} (Day {})".format(title, day), xlabel="Time of day (h)")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        if panel == 0:
            ax.set_ylabel("Farm power (MW)")
            ax.legend(frameon=False, loc="upper left")
    return save_figure(fig, out, "fig05_timeseries")
