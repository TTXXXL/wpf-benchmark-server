"""End-to-end persistence regression check."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from ..paths import ProjectPaths
from .evaluator import Evaluator
from .report import result_to_markdown, save_result


def selftest(paths: Optional[ProjectPaths] = None) -> None:
    """Run the persistence forecast through both hygiene tables and write reports."""
    paths = paths or ProjectPaths.resolve()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ev = Evaluator(paths=paths)
        preds = ev.persistence_preds()
        main = ev.evaluate(preds, "persistence_selftest", "main")
        all_rows = ev.evaluate(preds, "persistence_selftest", "all")
    # References from the causal, train-only cleaning pipeline (k=12).
    for report, expected in ((main, 4948656), (all_rows, 5537952)):
        if abs(report["n_samples"] - expected) > expected * 0.001:
            raise AssertionError("Unexpected {} sample count: {}".format(
                report["table"], report["n_samples"]))
    if not np.isclose(main["A_turbine"]["SS_vs_persistence_pct"], 0.0, atol=1e-6):
        raise AssertionError("Persistence skill score should be zero")
    if main["C_ramp"]["n_positive_cells"] <= 0 or main["C_ramp"]["recall"] != 0:
        raise AssertionError("Persistence ramp recall should be zero with true positive cells")
    if not np.isnan(main["C_ramp"]["precision"]):
        raise AssertionError("Persistence ramp precision should be undefined")
    if main["A_farm"]["n"] <= 0:
        raise AssertionError("Farm aggregation is empty")
    if not 66.0 < main["A_turbine"]["MAE_kW"] < 67.0:
        raise AssertionError("Persistence MAE differs from the reference")
    physics = main["D_physics"]
    if not 0.7 <= physics["truth_violation_rate_pct"] <= 0.8:
        raise AssertionError("Truth curve violation rate differs from the reference")
    if not 17.5 <= physics["violation_rate_pct"] <= 18.0:
        raise AssertionError("Persistence curve violation rate differs from the reference")
    outputs = [save_result(main, "selftest_main", paths.evaluation),
               save_result(all_rows, "selftest_all", paths.evaluation)]
    report_path = paths.reports / "eval_selftest.md"
    report_path.write_text(result_to_markdown(main) + "\n\n---\n\n" +
                           result_to_markdown(all_rows), encoding="utf-8")
    print("SELFTEST OK: main {:,}; all {:,}; MAE {:.2f} kW; truth violations {:.2f}%; "
          "persistence violations {:.2f}%".format(
              main["n_samples"], all_rows["n_samples"],
              main["A_turbine"]["MAE_kW"],
              main["D_physics"]["truth_violation_rate_pct"],
              main["D_physics"]["violation_rate_pct"]))
    for path in outputs + [report_path]:
        print(path)
