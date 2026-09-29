"""论文图件与表格注册表及统一 CLI 入口。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..paths import ProjectPaths
from .io import MissingMaterial
from . import (fig02_layout, fig03_quality, fig05_timeseries, fig06_horizon,
               fig07_windbins, fig08_density, fig09_ablation, fig10_lambda,
               fig11_ramp, fig12_graph, fig13_physics, tab01_main,
               tab02_dataset, tab03_overhead, tab04_fcurve)


FIGURE_REGISTRY: Dict[str, Callable] = {
    "fig02": fig02_layout.render,
    "fig03": fig03_quality.render,
    "fig05": fig05_timeseries.render,
    "fig06": fig06_horizon.render,
    "fig07": fig07_windbins.render,
    "fig08": fig08_density.render,
    "fig09": fig09_ablation.render,
    "fig10": fig10_lambda.render,
    "fig11": fig11_ramp.render,
    "fig12": fig12_graph.render,
    "fig13": fig13_physics.render,
}
TABLE_REGISTRY: Dict[str, Callable] = {
    "tab01": tab01_main.render,
    "tab02": tab02_dataset.render,
    "tab03": tab03_overhead.render,
    "tab04": tab04_fcurve.render,
}


def _selected(raw: Optional[str], prefix: str, registry: Dict[str, Callable]) -> List[str]:
    if not raw:
        return list(registry)
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part.isdigit():
            raise ValueError("Expected comma-separated numbers, received: " + part)
        key = "{}{:02d}".format(prefix, int(part))
        if key not in registry:
            raise ValueError("Unknown artifact: " + key)
        values.append(key)
    return list(dict.fromkeys(values))


def generate(paths: ProjectPaths, out: Path, figures: Optional[str] = None,
             tables: Optional[str] = None, all_artifacts: bool = False,
             wind_direction: Optional[float] = None,
             graph_run_id: Optional[str] = None) -> Tuple[Dict[str, List[Path]], Dict[str, str]]:
    """各图独立运行；只对缺素材的图打印 SKIP，代码错误照常抛出。"""
    if all_artifacts or (figures is None and tables is None):
        figure_names, table_names = list(FIGURE_REGISTRY), list(TABLE_REGISTRY)
    else:
        figure_names = _selected(figures, "fig", FIGURE_REGISTRY) if figures else []
        table_names = _selected(tables, "tab", TABLE_REGISTRY) if tables else []
    made: Dict[str, List[Path]] = {}
    skipped: Dict[str, str] = {}
    for name in figure_names + table_names:
        function = FIGURE_REGISTRY.get(name) or TABLE_REGISTRY[name]
        try:
            if name == "fig02":
                produced = function(paths, out, wind_direction=wind_direction)
            elif name == "fig12":
                produced = function(paths, out, run_id=graph_run_id)
            else:
                produced = function(paths, out)
            made[name] = produced
            print("OK {}: {}".format(name, ", ".join(str(path) for path in produced)))
        except MissingMaterial as exc:
            skipped[name] = str(exc)
            print("SKIP {}: {}".format(name, exc))
    return made, skipped
