"""End-to-end persistence regression check."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from ..paths import ProjectPaths
from .config import ProtocolConfig
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
        m2_ev = Evaluator(ProtocolConfig(eval_mask="m2"), paths=paths)
        m2 = m2_ev.evaluate(m2_ev.persistence_preds(), "persistence_selftest", "main")
    # References from the causal, train-only cleaning pipeline (k=12).
    for report, expected in ((main, 3188016), (m2, 3636220), (all_rows, 5537952)):
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
    if not 99.7 < main["A_turbine"]["MAE_kW"] < 100.0:
        raise AssertionError("Persistence MAE differs from the reference")
    if not 104.8 < m2["A_turbine"]["MAE_kW"] < 105.1:
        raise AssertionError("M2 persistence MAE differs from the reference")
    if not np.isclose(main["A_turbine"]["RMSE_kW"], 165.89, atol=0.01):
        raise AssertionError("M1 persistence RMSE differs from the reference")
    if not np.isclose(m2["A_turbine"]["RMSE_kW"], 173.09, atol=0.01):
        raise AssertionError("M2 persistence RMSE differs from the reference")
    base = np.isfinite(ev.dte["Patv"])
    for flag in ev.cfg.exclude_flags_main:
        base &= ~ev.dte[flag]
    for flag, expected in (("o_bad", 1760640), ("o_pitch", 1705044),
                           ("o_zero_windy", 100212), ("o_dir_abnormal", 0)):
        actual = int(ev._valid_stack(base & ev.dte[flag]).sum())
        if actual != expected:
            raise AssertionError("Official-rule overlap {} differs: {}".format(flag, actual))
    physics = main["D_physics"]
    if not 1.0 <= physics["truth_violation_rate_pct"] <= 1.2:
        raise AssertionError("Truth curve violation rate differs from the reference")
    if not 25.8 <= physics["violation_rate_pct"] <= 26.2:
        raise AssertionError("Persistence curve violation rate differs from the reference")
    outputs = [save_result(main, "selftest_main", paths.evaluation),
               save_result(m2, "selftest_m2", paths.evaluation),
               save_result(all_rows, "selftest_all", paths.evaluation)]
    report_path = paths.reports / "eval_selftest.md"
    report_path.write_text(result_to_markdown(main) + "\n\n---\n\n" +
                           result_to_markdown(m2) + "\n\n---\n\n" +
                           result_to_markdown(all_rows), encoding="utf-8")
    print("SELFTEST OK: M1 {:,} / {:.2f} kW; M2 {:,} / {:.2f} kW; all {:,}; "
          "truth violations {:.2f}%; persistence violations {:.2f}%".format(
              main["n_samples"], main["A_turbine"]["MAE_kW"],
              m2["n_samples"], m2["A_turbine"]["MAE_kW"],
              all_rows["n_samples"],
              main["D_physics"]["truth_violation_rate_pct"],
              main["D_physics"]["violation_rate_pct"]))
    for path in outputs + [report_path]:
        print(path)
