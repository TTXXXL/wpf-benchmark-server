"""图 9：消融；caption: Ablation variants reveal the contribution of each component."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ..paths import ProjectPaths
from .io import MissingMaterial, load_metrics, model_groups
from .style import DOUBLE, apply_figure_style, model_style, save_figure
from .table_utils import escape_tex, write_table


def render(paths: ProjectPaths, out: Path) -> List[Path]:
    groups = model_groups(paths, experiment="ablation")
    if not groups:
        raise MissingMaterial("ablation runs not found")
    order = ("ours", "ours_no_consist", "ours_no_wake", "barest")
    names = [name for name in order if name in groups]
    names += sorted(set(groups) - set(names))
    measures = (("A_turbine", "MAE_kW"),
                ("A_turbine", "RMSE_kW"),
                ("D_physics", "violation_rate_pct"))
    values = np.array([[np.mean([load_metrics(paths, record)[group][key]
                                 for record in groups[name]])
                        for group, key in measures] for name in names])
    deviations = np.array([[np.std([load_metrics(paths, record)[group][key]
                                    for record in groups[name]], ddof=1)
                            if len(groups[name]) > 1 else 0.0
                            for group, key in measures] for name in names])
    apply_figure_style()
    fig, axes = plt.subplots(1, 3, figsize=DOUBLE)
    labels = ("MAE (kW)", "RMSE (kW)", "Curve deviation (%)")
    for j, ax in enumerate(axes):
        ax.bar(range(len(names)), values[:, j], yerr=deviations[:, j],
               capsize=2, error_kw={"linewidth": 0.7},
               color=[model_style(name).color for name in names])
        ax.set(title=chr(ord("a") + j) + "  " + labels[j], ylabel=labels[j])
        ax.set_xticks(range(len(names)), [model_style(name).label for name in names],
                      rotation=30, ha="right")
        ax.set_ylim(bottom=0)
    outputs = save_figure(fig, out, "fig09_ablation")
    def formatted(mean, std):
        return "{:.2f} $\\pm$ {:.2f}".format(mean, std)
    rows = []
    for i, name in enumerate(names):
        metrics = [load_metrics(paths, record) for record in groups[name]]
        direct = [float(item["direct_consist_error_kW"]) for item in metrics
                  if item.get("direct_consist_error_kW") is not None]
        direct_text = (formatted(float(np.mean(direct)),
                                 float(np.std(direct, ddof=1)) if len(direct) > 1 else 0.0)
                       if direct else "—")
        rows.append([escape_tex(model_style(name).label)] +
                    [formatted(values[i, j], deviations[i, j]) for j in range(3)] +
                    [direct_text, str(len(groups[name]))])
    outputs += write_table(out, "fig09_ablation", ["Variant", "MAE (kW)",
                          "RMSE (kW)", "Curve deviation (\\%)",
                          "Direct consistency (kW)", "Seeds"], rows, "lrrrrr")
    return outputs
