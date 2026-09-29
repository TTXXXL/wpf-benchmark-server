"""图 12：学习图与尾流图；caption: Learned connectivity can be compared with wind-conditioned wake geometry."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..paths import ProjectPaths
from .io import MissingMaterial, latest_cohort, load_index
from .style import DOUBLE, apply_figure_style, save_figure


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def render(paths: ProjectPaths, out: Path, source: Optional[Path] = None,
           run_id: Optional[str] = None) -> List[Path]:
    if source is None:
        ours = [record for record in load_index(paths)
                if record.model == "ours"]
        if run_id is None:
            ours = latest_cohort(ours, paths)
        if run_id is not None:
            ours = [record for record in ours if record.run_id == run_id]
        if not ours:
            raise MissingMaterial("No matching ours main run for figure 12")
        selected = max(ours, key=lambda record: record.created)
        source = paths.reports / "interpret" / (selected.run_id + "_graph.npz")
    if not source.is_file():
        raise MissingMaterial("Graph npz not found: {}".format(source))
    with np.load(source, allow_pickle=False) as archive:
        for key in ("A_learned", "A_wake", "A_wake_prior", "wind_dirs",
                    "w_k", "turbine_ids", "convention"):
            if key not in archive:
                raise ValueError("Graph npz missing {}".format(key))
        learned, wake, prior, directions, weights, ids = (archive[key] for key in
             ("A_learned", "A_wake", "A_wake_prior", "wind_dirs", "w_k",
              "turbine_ids"))
        convention = str(archive["convention"].item())
    if learned.ndim != 2 or learned.shape[0] != learned.shape[1] or (
            wake.ndim != 3 or wake.shape[1:] != learned.shape or
            prior.shape != learned.shape or directions.shape != (wake.shape[0],)
            or weights.shape != directions.shape or ids.shape != (len(learned),)):
        raise ValueError("Graph npz has inconsistent matrix, direction or ID shapes")
    if not np.isfinite(learned).all() or not np.isfinite(prior).all():
        raise ValueError("Graph matrices must be finite")
    if not np.isclose(weights.sum(), 1.0, atol=1e-6) or (
            len(np.unique(ids)) != len(ids)):
        raise ValueError("Graph frequencies or turbine IDs are invalid")
    nonzero = prior.sum(axis=1) > 0
    if not np.allclose(prior.sum(axis=1)[nonzero], 1.0, atol=1e-6):
        raise ValueError("Nonzero wake prior rows must sum to one")
    mask = ~np.eye(len(learned), dtype=bool)
    x, y = learned[mask], prior[mask]
    pearson = _correlation(x, y)
    spearman = _correlation(pd.Series(x).rank(method="average").to_numpy(),
                            pd.Series(y).rank(method="average").to_numpy())
    apply_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
    lower, upper = float(min(learned.min(), prior.min())), float(max(learned.max(), prior.max()))
    ticks = np.unique(np.linspace(0, len(ids) - 1, min(5, len(ids))).astype(int))
    for ax, matrix, title in zip(axes, (learned, prior),
                                 ("a  Learned graph", "b  Train-frequency wake prior")):
        image = ax.imshow(matrix, cmap="viridis", interpolation="nearest",
                          vmin=lower, vmax=upper)
        ax.set(title=title, xlabel="Source turbine", ylabel="Target turbine")
        ax.set_xticks(ticks, [str(ids[tick]) for tick in ticks])
        ax.set_yticks(ticks, [str(ids[tick]) for tick in ticks])
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle("Off-diagonal r = {:.2f}; rank r = {:.2f}".format(pearson, spearman),
                 fontsize=8)
    outputs = save_figure(fig, out, "fig12_graph")
    columns = 4
    rows = int(np.ceil(len(directions) / columns))
    supplemental, panels = plt.subplots(rows, columns,
                                         figsize=(DOUBLE[0], max(2.2, rows * 2.1)),
                                         squeeze=False)
    for k, ax in enumerate(panels.flat):
        if k >= len(directions):
            ax.axis("off")
            continue
        ax.imshow(wake[k], cmap="viridis", interpolation="nearest", vmin=0,
                  vmax=max(float(wake.max()), 1e-6))
        ax.set(title="{:.0f}°; weight {:.2f}".format(directions[k], weights[k]),
               xlabel="Source", ylabel="Target")
    supplemental.suptitle("Directional wake matrices ({})".format(convention), fontsize=8)
    outputs += save_figure(supplemental, out, "fig12_graph_supplementary")
    return outputs
