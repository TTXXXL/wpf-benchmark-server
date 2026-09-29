"""图 10：λ 扫描；caption: The sweep shows how forecast error changes with consistency weight."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ..paths import ProjectPaths
from .io import MissingMaterial, latest_cohort, load_index, load_metrics
from .style import SINGLE, apply_figure_style, save_figure
from .table_utils import write_table


def render(paths: ProjectPaths, out: Path, parameter: str = "lambda_consist") -> List[Path]:
    rows = []
    sweep = latest_cohort([record for record in load_index(paths, table="validation")
                           if record.experiment == "lambda_sweep"], paths)
    # A repeated lambda can belong to another model/configuration. Keep only
    # the newest full configuration at each weight, then aggregate its seeds.
    by_weight = {}
    for record in sweep:
        run = load_metrics(paths, record)
        value = run.get("model_config", {}).get(parameter)
        if value is None:
            continue
        value = float(value)
        if value <= 0:
            continue
        by_weight.setdefault(value, []).append((record, run))
    for value, candidates in by_weight.items():
        newest = max(candidates, key=lambda item: item[0].created)[0]
        for record, run in candidates:
            if record.model != newest.model or record.config_digest != newest.config_digest:
                continue
            validation = run.get("validation_metrics", {})
            if validation.get("MAE_kW") is not None:
                rows.append((value, float(validation["MAE_kW"]),
                             validation.get("direct_consist_error_kW")))
    if not rows:
        raise MissingMaterial("lambda_sweep runs with positive {} not found".format(parameter))
    grouped = {}
    for x, valid, direct in rows:
        grouped.setdefault(x, []).append((valid, direct))
    xvals = sorted(grouped)
    valid_values = [float(np.mean([row[0] for row in grouped[x]])) for x in xvals]
    std_values = [float(np.std([row[0] for row in grouped[x]], ddof=1))
                  if len(grouped[x]) > 1 else 0.0 for x in xvals]
    apply_figure_style()
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.errorbar(xvals, valid_values, yerr=std_values, color="#0072B2",
                marker="o", capsize=2, label="Validation MAE")
    ax.set_xscale("log")
    ax.set(xlabel=parameter.replace("_", " "), ylabel="MAE (kW)")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.4)
    outputs = save_figure(fig, out, "fig10_lambda")
    table_rows = []
    for x, mean, std in zip(xvals, valid_values, std_values):
        direct = [float(item[1]) for item in grouped[x] if item[1] is not None]
        table_rows.append(["{:.4g}".format(x), "{:.2f} $\\pm$ {:.2f}".format(mean, std),
                           "{:.2f}".format(np.mean(direct)) if direct else "—",
                           str(len(grouped[x]))])
    outputs += write_table(out, "fig10_lambda", ["Lambda", "Validation MAE (kW)",
                           "Direct consistency (kW)", "Seeds"], table_rows, "lrrr")
    return outputs
