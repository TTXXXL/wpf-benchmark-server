"""表 4：不同 f_curve 来源的敏感性结果。"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from ..paths import ProjectPaths
from .io import MissingMaterial, latest_cohort, load_index, load_metrics
from .table_utils import escape_tex, write_table


def render(paths: ProjectPaths, out: Path) -> List[Path]:
    groups = {}
    selected = latest_cohort([record for record in load_index(paths)
                              if record.experiment == "fcurve_sensitivity"], paths)
    candidates = {}
    for record in selected:
        result = load_metrics(paths, record)
        name = result.get("model_config", {}).get("f_curve_source")
        if name:
            candidates.setdefault(str(name), []).append((record, result))
    for name, runs in candidates.items():
        newest = max(runs, key=lambda item: item[0].created)[0]
        groups[name] = [result for record, result in runs
                        if record.model == newest.model and
                        record.config_digest == newest.config_digest]
    if not groups:
        raise MissingMaterial("fcurve_sensitivity runs with model_config.f_curve_source not found")
    columns = sorted(groups)
    specs = (("MAE (kW)", "A_turbine", "MAE_kW"),
             ("RMSE (kW)", "A_turbine", "RMSE_kW"),
             ("SS (\\%)", "A_turbine", "SS_vs_persistence_pct"),
             ("Curve deviation (\\%)", "D_physics", "violation_rate_pct"))
    rows = []
    for label, section, metric in specs:
        row = [label]
        for name in columns:
            vals = [result[section].get(metric) for result in groups[name]]
            vals = [value for value in vals if value is not None]
            row.append("{:.2f}".format(np.mean(vals)) if vals else "--")
        rows.append(row)
    return write_table(out, "tab04_fcurve", ["Metric"] + [escape_tex(col) for col in columns],
                       rows, "l" + "r" * len(columns))
