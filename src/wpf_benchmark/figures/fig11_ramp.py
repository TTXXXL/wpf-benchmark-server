"""图 11：爬坡案例；caption: Event-centred traces expose missed power ramps."""
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
    ref_metrics, ref = next(iter(runs.values()))
    if "last_power" not in ref:
        raise MissingMaterial("arrays lack last_power; rerun with --save-arrays")
    valid = ref["valid"][:, :, 0]
    times = ref["times"][1:len(valid) + 1]
    days = times // DAY_NS + 1
    event_count = ref["ramp_mask"][:, :, 0].sum(axis=1)
    if not np.any(event_count):
        raise MissingMaterial("no true ramp events in arrays")
    unique = np.unique(days)
    counts = np.array([event_count[days == day].sum() for day in unique])
    if len(unique) >= 3:
        starts = range(len(unique) - 2)
        best = max(starts, key=lambda i: counts[i:i+3].sum())
        window_days = unique[best:best+3]
    else:
        window_days = unique
    candidates = np.where(np.isin(days, window_days))[0]
    ranked = sorted(candidates, key=lambda i: event_count[i], reverse=True)
    chosen = []
    for index in ranked:
        if event_count[index] and all(abs(index - prior) >= 6 for prior in chosen):
            chosen.append(int(index))
        if len(chosen) == 2:
            break
    if not chosen:
        raise MissingMaterial("no concentrated ramp windows")
    truth = np.where(valid, ref["truth"][:, :, 0], 0).sum(axis=1) / 1000
    apply_figure_style()
    fig, axes = plt.subplots(len(chosen), 1, figsize=(DOUBLE[0], 2.3 * len(chosen)),
                             squeeze=False)
    threshold = ref_metrics["C_ramp"]["threshold_kW"]
    for panel, center in enumerate(chosen):
        ax = axes[panel, 0]
        take = (times >= times[center] - 3 * 3600 * 10**9) & (
            times <= times[center] + 3 * 3600 * 10**9)
        hours = (times[take] - times[center]) / (3600 * 10**9)
        ax.plot(hours, truth[take], color="black", lw=1.3, label="Truth")
        for name, (_, arrays) in runs.items():
            pred = np.where(valid, arrays["forecasts"][:, :, 0], 0).sum(axis=1) / 1000
            local_pred = arrays["forecasts"][take, :, 0]
            local_last = arrays["last_power"][take]
            event_interval = ((times[take] >= times[center] - 600 * 10**9) &
                              (times[take] <= times[center]))[:, None]
            predicted_event = ((np.abs(local_pred - local_last) >= threshold) &
                               valid[take] & event_interval)
            matching = np.any(predicted_event & ref["ramp_mask"][take, :, 0])
            status = "" if matching else (" (false alarm)" if np.any(predicted_event)
                                           else " (missed)")
            style = model_style(name)
            ax.plot(hours, pred[take], color=style.color, linestyle=style.linestyle,
                    label=style.label + status)
        ax.axvline(-1/6, color="#D55E00", linestyle=":", lw=0.8,
                   label="Event interval")
        ax.axvline(0, color="#D55E00", linestyle=":", lw=0.8)
        ax.set(xlabel="Hours from event", ylabel="Farm power (MW)",
               title="{}  Day {} event".format(chr(ord("a") + panel), days[center]))
        ax.legend(frameon=False, loc="upper right")
    return save_figure(fig, out, "fig11_ramp")
