"""表 3：训练、推理与模型大小。"""
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
    for name, records in sorted(groups.items()):
        runs = [load_metrics(paths, record) for record in records]
        if any("timing" not in run or "model_size" not in run for run in runs):
            raise MissingMaterial("timing/model_size missing; rerun models with current runner")
        fit = np.mean([run["timing"]["fit_seconds"] for run in runs])
        predict = np.mean([100000 * run["timing"]["predict_seconds"] /
                           run["timing"]["n_predict_samples"] for run in runs])
        params = np.mean([run["model_size"]["n_params"] for run in runs])
        rows.append([escape_tex(model_style(name).label), "{:,.0f}".format(params),
                     "{:.2f}".format(fit), "{:.2f}".format(predict)])
    return write_table(out, "tab03_overhead",
                       ["Model", "Parameters", "Fit (s)", "Predict / 100k (s)"],
                       rows, "lrrr")
