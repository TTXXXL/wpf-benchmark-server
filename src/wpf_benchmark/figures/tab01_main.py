"""表 1：单机和场站主结果。"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from ..paths import ProjectPaths
from .io import MissingMaterial, load_metrics, model_groups
from .style import model_style
from .table_utils import escape_tex, write_table


def render(paths: ProjectPaths, out: Path) -> List[Path]:
    groups = model_groups(paths)
    if not groups:
        raise MissingMaterial("main evaluation JSON not found")
    rows = []
    for model, records in sorted(groups.items()):
        runs = [load_metrics(paths, record) for record in records]
        for scope, key in (("Turbine", "A_turbine"), ("Farm", "A_farm")):
            vals = []
            for metric in ("MAE_kW", "RMSE_kW", "NMAE_pct", "SS_vs_persistence_pct"):
                series = [run[key].get(metric) for run in runs if run[key].get(metric) is not None]
                vals.append(float(np.mean(series)) if series else None)
            rows.append((model, scope, vals))
    best = {}
    for scope in ("Turbine", "Farm"):
        for column in range(4):
            column_vals = [item[2][column] for item in rows
                           if item[1] == scope and item[2][column] is not None]
            best[scope, column] = ((max if column == 3 else min)(column_vals)
                                   if column_vals else None)
    table_rows = []
    for model, scope, values in rows:
        formatted = []
        for col, value in enumerate(values):
            if value is None:
                formatted.append("--")
                continue
            text = "{:.2f}".format(value)
            if value == best[scope, col]:
                text = r"\textbf{" + text + "}"
            formatted.append(text)
        table_rows.append([escape_tex(model_style(model).label), scope] + formatted)
    return write_table(out, "tab01_main",
                       ["Model", "Scale", "MAE (kW)", "RMSE (kW)", "NMAE (\\%)", "SS (\\%)"],
                       table_rows, "llrrrr",
                       "Farm NMAE divides each eligible farm-cell absolute error by the rated capacity of its valid turbines, then averages across cells.")
