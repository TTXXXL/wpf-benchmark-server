"""Train-only physical curves and aligned A-D evaluation metrics."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..data.arrays import split_by_day, to_wide
from ..data.io import load_clean
from ..paths import ProjectPaths
from .config import ProtocolConfig
from .metrics import _fit_power_curves, _metrics, _safe_ratio


class Evaluator:
    """Evaluate any complete (T_test-horizon,N,horizon) power forecast."""

    def __init__(self, cfg: Optional[ProtocolConfig] = None,
                 data: Optional[pd.DataFrame] = None,
                 paths: Optional[ProjectPaths] = None):
        self.cfg = cfg or ProtocolConfig()
        self.paths = paths or ProjectPaths.resolve()
        columns = ["ts", "Day", "TurbID", "Patv", "Wspd"] + list(self.cfg.exclude_flags_main)
        if data is None:
            columns = list(dict.fromkeys(columns))
            tr = load_clean(self.paths, columns=columns,
                            filters=[("Day", "<=", self.cfg.train_days)])
            te = load_clean(self.paths, columns=columns,
                            filters=[("Day", ">", self.cfg.train_days + self.cfg.val_days)])
        else:
            tr, _, te = split_by_day(data, self.cfg)
        if tr.empty or te.empty:
            raise ValueError("Configured train/test split is empty")
        self.test_start_day = int(te["Day"].min())
        self.dtr = to_wide(tr, self.cfg)
        self.dte = to_wide(te, self.cfg)
        del tr, te
        if not np.array_equal(self.dtr["tids"], self.dte["tids"]):
            raise ValueError("Train and test turbine IDs do not match")
        T, N = self.dte["T"], self.dte["N"]
        self.T_eff = T - self.cfg.horizon
        if self.T_eff <= 0:
            raise ValueError("Test portion is shorter than the forecast horizon")
        excl_test = np.zeros((T, N), dtype=bool)
        excl_train = np.zeros_like(self.dtr["Patv"], dtype=bool)
        for flag in self.cfg.exclude_flags_main:
            excl_test |= self.dte[flag]
            excl_train |= self.dtr[flag]
        self.valid_main = np.isfinite(self.dte["Patv"]) & ~excl_test
        self.valid_all = np.isfinite(self.dte["Patv"])
        self.curve_centers, self.curves, self.curve_sigmas = _fit_power_curves(
            self.dtr["Wspd"], self.dtr["Patv"], excl_train, self.cfg)
        del excl_test, excl_train
        # The fitted curves are sufficient; retain no training grid during scoring.
        del self.dtr

    def _valid_stack(self, base: np.ndarray) -> np.ndarray:
        H, T = self.cfg.horizon, base.shape[0]
        return np.stack([base[h:T - H + h] for h in range(1, H + 1)], axis=2)

    def persistence_preds(self) -> np.ndarray:
        """P_hat(t+h) = P(t) for every horizon."""
        last = self.dte["Patv"][:self.T_eff]
        return np.broadcast_to(last[:, :, None],
                               (self.T_eff, self.dte["N"], self.cfg.horizon))

    def _curve_lookup(self, speed: np.ndarray):
        edges = np.arange(0.0, 26.0 + self.cfg.curve_bin_width,
                          self.cfg.curve_bin_width)
        idx = np.clip(np.searchsorted(edges, speed, side="right") - 1,
                      0, self.curves.shape[1] - 1)
        turbines = np.arange(self.dte["N"])[None, :, None]
        return self.curves[turbines, idx], self.curve_sigmas[turbines, idx]

    def evaluate(self, preds: np.ndarray, model_name: str,
                 table: str = "main") -> Dict[str, Any]:
        if table not in ("main", "all"):
            raise ValueError("table must be 'main' or 'all'")
        expected = (self.T_eff, self.dte["N"], self.cfg.horizon)
        if np.shape(preds) != expected:
            raise ValueError("Forecast shape must be {}, received {}".format(expected, np.shape(preds)))
        preds = np.asarray(preds, dtype=np.float32)
        if not np.isfinite(preds).all():
            raise ValueError("Forecasts must be finite at every issue time, turbine and horizon")
        cfg, H = self.cfg, self.cfg.horizon
        truth = self._valid_stack(self.dte["Patv"])
        valid = self._valid_stack(self.valid_main if table == "main" else self.valid_all)
        speed = self._valid_stack(self.dte["Wspd"])
        err = preds - truth
        n_valid = int(valid.sum())
        R: Dict[str, Any] = {"model": model_name, "table": table,
                             "n_samples": n_valid, "config": asdict(cfg)}

        R["A_turbine"] = _metrics(err[valid], cfg)
        valid_turbines = valid.sum(axis=1)
        farm_valid = ((valid_turbines / self.dte["N"] >= cfg.farm_min_valid_frac) &
                      (valid_turbines > 0))
        farm_error = np.where(valid, err, 0.0).sum(axis=1, dtype=np.float64)
        farm_metrics = _metrics(farm_error[farm_valid], cfg)
        farm_metrics["NMAE_pct"] = (
            float(100.0 * np.mean(
                np.abs(farm_error[farm_valid]) /
                (valid_turbines[farm_valid] * cfg.rated_power_kw),
                dtype=np.float64)) if np.any(farm_valid) else float("nan"))
        R["A_farm"] = dict(farm_metrics, n=int(farm_valid.sum()),
                            normalization="eligible_turbine_capacity")
        last = self.dte["Patv"][:self.T_eff, :, None]
        persistence_valid = valid & np.isfinite(last)
        persistence_mae = (float(np.abs(last - truth)[persistence_valid].mean(dtype=np.float64))
                           if np.any(persistence_valid) else float("nan"))
        model_mae = (float(np.abs(err[persistence_valid]).mean(dtype=np.float64))
                     if np.any(persistence_valid) else float("nan"))
        R["A_turbine"]["SS_vs_persistence_pct"] = (
            100.0 * (1.0 - model_mae / persistence_mae)
            if persistence_mae > 0 else float("nan"))
        R["A_turbine"]["persistence_MAE_kW"] = persistence_mae
        del farm_error

        R["B_per_horizon"] = {str(h + 1): dict(_metrics(err[:, :, h][valid[:, :, h]], cfg),
                                             n=int(valid[:, :, h].sum()))
                              for h in range(H)}
        wind_bins: Dict[str, Any] = {}
        for lo, hi in cfg.wind_bins:
            selected = valid & (speed >= lo) & (speed < hi)
            wind_bins["[{}, {})".format(lo, hi)] = dict(
                _metrics(err[selected], cfg), n=int(selected.sum()))
        R["B_wind_bins"] = wind_bins
        sample_errors = err[valid]
        R["B_bias"] = {
            "ME_kW": float(sample_errors.mean(dtype=np.float64)) if n_valid else float("nan"),
            "over_frac": float((sample_errors > 0).mean()) if n_valid else float("nan"),
            "under_frac": float((sample_errors < 0).mean()) if n_valid else float("nan")}
        del sample_errors
        blocks = []
        block_steps = cfg.stability_block_days * cfg.steps_per_day
        for start in range(0, self.T_eff, block_steps):
            stop = min(start + block_steps, self.T_eff)
            block_valid = valid[start:stop]
            first_day = self.test_start_day + start // cfg.steps_per_day
            last_day = self.test_start_day + (stop - 1) // cfg.steps_per_day
            blocks.append({"days": "{}-{}".format(first_day, last_day),
                           "n": int(block_valid.sum()),
                           **_metrics(err[start:stop][block_valid], cfg)})
        R["B_stability"] = blocks

        threshold = cfg.ramp_threshold_kw()
        ramp_true = valid & (np.abs(truth - last) >= threshold)
        ramp_pred = valid & (np.abs(preds - last) >= threshold)
        tp = int((ramp_true & ramp_pred).sum())
        n_positive_cells = int(ramp_true.sum())
        n_pred_positive_cells = int(ramp_pred.sum())
        precision = _safe_ratio(tp, n_pred_positive_cells)
        recall = _safe_ratio(tp, n_positive_cells)
        R["C_ramp"] = {
            "basis": "issue_time_turbine_horizon_cell",
            "threshold_kW": threshold,
            "n_positive_cells": n_positive_cells,
            "n_pred_positive_cells": n_pred_positive_cells,
            "n_true_positive_cells": tp,
            "recall": recall, "precision": precision,
            "f1": _safe_ratio(2 * tp, n_positive_cells + n_pred_positive_cells),
            "positive_cell_MAE_kW": float(np.abs(err[ramp_true]).mean(dtype=np.float64))
            if n_positive_cells else float("nan")}
        del ramp_true, ramp_pred, last

        # One h=1 forecast per target time gives each 10-minute period one vote.
        # The first test step and final H-1 steps lack such a forecast.
        days = (np.arange(self.T_eff) + 1) // cfg.steps_per_day
        target_valid = valid[:, :, 0]
        daily = []
        for day in np.unique(days):
            take = days == day
            mask = target_valid[take]
            actual_mwh = float(np.where(mask, truth[take, :, 0], 0.0).sum(dtype=np.float64) / 6000.0)
            forecast_mwh = float(np.where(mask, preds[take, :, 0], 0.0).sum(dtype=np.float64) / 6000.0)
            daily.append({"day": self.test_start_day + int(day),
                          "n_steps": int(take.sum()),
                          "actual_MWh": actual_mwh, "forecast_MWh": forecast_mwh,
                          "abs_error_MWh": abs(forecast_mwh - actual_mwh),
                          "APE_pct": 100.0 * abs(forecast_mwh - actual_mwh) / actual_mwh
                          if actual_mwh else float("nan")})
        R["C_daily_energy"] = {
            "basis": "h=1; each predicted 10-minute target counted once",
            "n_days": len(daily),
            "boundary_day_note": "Only target steps with an h=1 forecast are counted",
            "MAE_MWh": float(np.mean([row["abs_error_MWh"] for row in daily]))
            if daily else float("nan"),
            "MAPE_pct": float(np.mean([row["APE_pct"] for row in daily
                                        if np.isfinite(row["APE_pct"])]))
            if any(np.isfinite(row["APE_pct"]) for row in daily) else float("nan"),
            "per_day": daily}
        high_wind = valid & (speed >= cfg.high_wind_speed)
        high_n = int(high_wind.sum())
        R["C_high_wind"] = (dict(_metrics(err[high_wind], cfg), n=high_n)
                            if high_n > 100 else {"n": high_n, "note": "样本不足"})
        del high_wind

        curve, sigma = self._curve_lookup(speed)
        curve_valid = valid & np.isfinite(speed) & np.isfinite(curve) & np.isfinite(sigma)
        n_curve = int(curve_valid.sum())
        bound = cfg.violation_sigma * sigma
        R["D_physics"] = {
            "violation_rate_pct": 100.0 * _safe_ratio(
                int((curve_valid & (np.abs(preds - curve) > bound)).sum()), n_valid),
            "truth_violation_rate_pct": 100.0 * _safe_ratio(
                int((curve_valid & (np.abs(truth - curve) > bound)).sum()), n_valid),
            "curve_coverage_pct": 100.0 * _safe_ratio(n_curve, n_valid),
            "neg_rate_pct": 100.0 * _safe_ratio(int((valid & (preds < 0)).sum()), n_valid),
            "over_rated_rate_pct": 100.0 * _safe_ratio(
                int((valid & (preds > cfg.rated_power_kw)).sum()), n_valid)}
        # Paper 2: pinball, PICP, PINAW and CRPS.
        R["E_probabilistic"] = None
        return R
