"""Physics-informed AGCRN with supervised wind and optional prior losses."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..data.windows import to_feature_cube
from .neural import NeuralForecaster
from .power_curve import fit_power_curves, logistic_torch
from .registry import register_model
from .training import issue_indices, training_cube


class PINForecaster(NeuralForecaster):
    """The four ablations share all network parameters and differ in losses."""

    graph_model = True
    features = ("Wspd", "Wsin", "Wcos", "Patv")
    additional_columns = ("m_missing_wspd", "m_imputed_wspd")

    def __init__(self, hidden: int = 64, layers: int = 2, emb: int = 10,
                 temporal_stride: int = 6, dropout: float = 0.1,
                 lr: float = 0.001, batch: int = 16, epochs: int = 30,
                 patience: int = 5, stride: int = 6, weight_decay: float = 0.0,
                 lambda_consist: float = 0.1, lambda_wake: float = 0.1):
        super().__init__(hidden, layers, dropout, lr, batch, epochs, patience,
                         stride, weight_decay)
        self.emb = int(emb)
        self.temporal_stride = int(temporal_stride)
        self.lambda_consist = float(lambda_consist)
        self.lambda_wake = float(lambda_wake)
        if self.emb <= 0 or self.temporal_stride <= 0:
            raise ValueError("emb and temporal_stride must be positive")
        if self.lambda_consist < 0 or self.lambda_wake < 0:
            raise ValueError("Physics loss weights must be nonnegative")
        self.validation_metrics = None
        self.wake_prior = None

    @property
    def model_config(self) -> dict:
        return dict(super().model_config, emb=self.emb,
                    temporal_stride=self.temporal_stride,
                    lambda_consist=self.lambda_consist,
                    lambda_wake=self.lambda_wake)

    def _build_network(self, torch, n_turbines: int):
        from .networks import AGCRNNetwork
        return AGCRNNetwork(n_turbines, self.hidden, self.layers, self.emb,
                            self.config.horizon, self.temporal_stride,
                            output_wind=True)

    def _load_curves(self, train: pd.DataFrame) -> np.ndarray:
        metadata = {}
        if hasattr(self, "paths"):
            source = self.paths.processed / "sdwpf_meta.json"
            if source.is_file():
                metadata = json.loads(source.read_text(encoding="utf-8"))
        curves = metadata.get("power_curves")
        if curves is None:
            curves = fit_power_curves(train, self.config)
        try:
            params = np.asarray([curves[str(tid)]["params"]
                                 for tid in self.turbine_ids], dtype=np.float32)
        except KeyError as exc:
            raise ValueError("A turbine is missing its train-only power curve") from exc
        if params.shape != (len(self.turbine_ids), 4) or not np.isfinite(params).all():
            raise ValueError("Invalid train-only power curve parameters")
        return params

    def _load_wake_prior(self) -> None:
        if self.lambda_wake == 0:
            return
        if not hasattr(self, "paths"):
            raise ValueError("Wake training requires configured project paths")
        decision_path = self.paths.reports / "wake_calibration.json"
        if not decision_path.is_file():
            raise FileNotFoundError("Wake training requires the W0 calibration decision")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("decision") != "significant":
            raise ValueError("W0 did not approve wake loss for the main method")
        source = self.paths.processed / "wake_prior.npz"
        if not source.is_file():
            raise FileNotFoundError("Run prepare-wake-prior before training a wake variant")
        with np.load(source, allow_pickle=False) as archive:
            needed = ("A_wake", "A_wake_prior", "wind_dirs", "w_k", "turbine_ids")
            if any(key not in archive for key in needed):
                raise ValueError("wake_prior.npz is missing required arrays")
            if not np.array_equal(archive["turbine_ids"], self.turbine_ids):
                raise ValueError("Wake prior turbine order differs from training data")
            self.wake_prior = {key: archive[key].copy() for key in needed}
            self.wake_convention = str(archive["convention"].item()) if (
                "convention" in archive) else "unknown"
        if self.wake_convention != decision.get("selected_convention"):
            raise ValueError("Wake prior convention differs from the W0 decision")
        prior = self.wake_prior["A_wake_prior"]
        n = len(self.turbine_ids)
        if prior.shape != (n, n) or not np.isfinite(prior).all():
            raise ValueError("Wake prior must be a finite N x N matrix")
        rows = prior.sum(axis=1)
        if not np.allclose(rows[rows > 0], 1.0, atol=1e-6):
            raise ValueError("Nonzero wake prior rows must sum to one")

    @staticmethod
    def _wind_mask(frame: pd.DataFrame, turbine_ids: np.ndarray) -> np.ndarray:
        needed = ("m_missing_wspd", "m_imputed_wspd", "f_stuck")
        absent = [name for name in needed if name not in frame]
        if absent:
            raise ValueError("Re-run preprocess to create wind target flags: " +
                             ", ".join(absent))
        good = np.isfinite(frame["Wspd"].to_numpy(dtype=float))
        for name in needed:
            good &= ~frame[name].to_numpy(dtype=bool)
        aligned = frame.loc[:, ["ts", "TurbID"]].copy()
        aligned["_wind_valid"] = good.astype(np.float32)
        _, cube = to_feature_cube(aligned, ("_wind_valid",), turbine_ids)
        return cube[:, :, 0] > 0.5

    def _pin_batches(self, cube, mask_power, mask_wind, issues, shuffle):
        w, h = self.config.input_window, self.config.horizon
        past = np.arange(w) - w + 1
        future = np.arange(1, h + 1)
        selected = np.random.permutation(issues) if shuffle else issues
        p_index = self.features.index("Patv")
        v_index = self.features.index("Wspd")
        for first in range(0, len(selected), self.batch):
            take = selected[first:first + self.batch]
            x = cube[take[:, None] + past[None, :]].transpose(0, 2, 3, 1)
            steps = take[:, None] + future[None, :]
            power = cube[steps, :, p_index].transpose(0, 2, 1)
            wind = cube[steps, :, v_index].transpose(0, 2, 1)
            mp = mask_power[steps].transpose(0, 2, 1) & np.isfinite(power)
            mw = mask_wind[steps].transpose(0, 2, 1) & np.isfinite(wind)
            yield (np.nan_to_num(x), np.nan_to_num(power),
                   np.nan_to_num(wind), mp, mw)

    def _loss_terms(self, prediction, target_power, target_wind,
                    mask_power, mask_wind):
        torch = self.torch
        h = self.config.horizon
        pred_power, pred_wind = prediction[..., :h], prediction[..., h:]
        zero = prediction.sum() * 0.0
        n_power = mask_power.sum()
        n_wind = mask_wind.sum()
        both = mask_power & mask_wind
        n_both = both.sum()
        point_sum = ((pred_power - target_power).square() * mask_power).sum()
        wind_sum = ((pred_wind - target_wind).square() * mask_wind).sum()
        point = point_sum / n_power.clamp(min=1)
        wind = wind_sum / n_wind.clamp(min=1)
        consist_sum = zero
        if int(n_both.item()) > 0:
            p_index = self.features.index("Patv")
            v_index = self.features.index("Wspd")
            p_lo = float(self.scaler.minimum[p_index])
            p_span = float(max(self.scaler.maximum[p_index] - p_lo, 1e-6))
            v_lo = float(self.scaler.minimum[v_index])
            v_span = float(max(self.scaler.maximum[v_index] - v_lo, 1e-6))
            power_phys = pred_power * p_span + p_lo
            wind_phys = (pred_wind * v_span + v_lo).clamp(0.0, 30.0)
            curve = logistic_torch(wind_phys, self.curve_params_t)
            consist_sum = ((power_phys - curve).abs() * both).sum()
        consist = consist_sum / n_both.clamp(min=1) / self.config.rated_power_kw
        wake = zero
        if self.lambda_wake > 0:
            learned = self.network.adaptive_adjacency()
            difference = (learned - self.wake_prior_t).square()
            wake = (difference * self.wake_row_mask_t[:, None]).sum() / (
                self.wake_row_mask_t.sum().clamp(min=1) * difference.shape[1])
        total = point + wind + self.lambda_consist * consist + self.lambda_wake * wake
        return total, (point_sum, wind_sum, consist_sum, wake), (
            n_power, n_wind, n_both)

    def fit(self, train: pd.DataFrame, valid: Optional[pd.DataFrame] = None) -> None:
        if valid is None or valid.empty:
            raise ValueError("PIN requires a validation split")
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PIN requires PyTorch; install wpf-benchmark[deep]") from exc
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, train_tids, train_cube, train_mask, _ = training_cube(
            train, self.features, self.config, self.scaler)
        _, valid_tids, valid_cube, valid_mask, _ = training_cube(
            valid, self.features, self.config, self.scaler)
        if not np.array_equal(train_tids, valid_tids):
            raise ValueError("Training and validation turbine grids differ")
        self.turbine_ids = train_tids
        train_wind_mask = self._wind_mask(train, train_tids)
        valid_wind_mask = self._wind_mask(valid, valid_tids)
        self.curve_params = self._load_curves(train)
        self.curve_params_t = torch.tensor(self.curve_params, device=self.device)
        self._load_wake_prior()
        if self.wake_prior is not None:
            prior = self.wake_prior["A_wake_prior"]
            self.wake_prior_t = torch.tensor(prior, device=self.device)
            self.wake_row_mask_t = torch.tensor(prior.sum(axis=1) > 0,
                                                device=self.device, dtype=torch.float32)
        train_issues = issue_indices(len(train_cube), self.config.input_window,
                                     self.config.horizon, self.stride)
        valid_issues = issue_indices(len(valid_cube), self.config.input_window,
                                     self.config.horizon, self.stride)
        if not len(train_issues) or not len(valid_issues):
            raise ValueError("Training or validation split is too short for windows")
        self.network = self._build_network(torch, len(train_tids)).to(self.device)
        self.n_params = sum(p.numel() for p in self.network.parameters()
                            if p.requires_grad)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        best_loss = float("inf")
        best_state = None
        wait = 0
        lines = ["model={} device={} train_windows={} valid_windows={} params={}".format(
            self.name, self.device, len(train_issues), len(valid_issues), self.n_params)]
        for epoch in range(1, self.epochs + 1):
            epoch_metrics = []
            for training, cube, mp, mw, issues in (
                    (True, train_cube, train_mask, train_wind_mask, train_issues),
                    (False, valid_cube, valid_mask, valid_wind_mask, valid_issues)):
                self.network.train(training)
                sums = np.zeros(3, dtype=np.float64)
                counts = np.zeros(3, dtype=np.int64)
                wake_sum = 0.0
                total_sum = 0.0
                n_batches = 0
                for x, yp, yw, mask_p, mask_w in self._pin_batches(
                        cube, mp, mw, issues, training):
                    if not mask_p.any() and not mask_w.any():
                        continue
                    xt = torch.from_numpy(x.astype(np.float32, copy=False)).to(self.device)
                    pt = torch.from_numpy(yp.astype(np.float32, copy=False)).to(self.device)
                    wt = torch.from_numpy(yw.astype(np.float32, copy=False)).to(self.device)
                    mpt = torch.from_numpy(mask_p).to(self.device)
                    mwt = torch.from_numpy(mask_w).to(self.device)
                    with torch.set_grad_enabled(training):
                        prediction = self.network(xt)
                        loss, terms, valid_counts = self._loss_terms(
                            prediction, pt, wt, mpt, mwt)
                        if training:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    sums += [float(item.detach().cpu()) for item in terms[:3]]
                    counts += [int(item.item()) for item in valid_counts]
                    wake_sum += float(terms[3].detach().cpu())
                    total_sum += float(loss.detach().cpu())
                    n_batches += 1
                if counts[0] == 0 or n_batches == 0:
                    raise ValueError("No valid power targets in a PIN split")
                raw = sums / np.maximum(counts, 1)
                wake_mean = wake_sum / n_batches
                total_mean = total_sum / n_batches
                epoch_metrics.append((raw, wake_mean, total_mean, counts))
            (tr, tw, tt, tc), (va, vw, vt, vc) = epoch_metrics
            lines.append(("epoch={} train_point={:.8f} train_wind={:.8f} "
                          "train_consist={:.8f} train_wake={:.8f} train_total={:.8f} "
                          "valid_point={:.8f} valid_wind={:.8f} "
                          "valid_consist={:.8f} valid_wake={:.8f} valid_total={:.8f} "
                          "train_counts={} valid_counts={}").format(
                              epoch, tr[0], tr[1], tr[2] / self.config.rated_power_kw,
                              tw, tt, va[0], va[1],
                              va[2] / self.config.rated_power_kw, vw, vt,
                              tc.tolist(), vc.tolist()))
            if va[0] < best_loss - 1e-8:
                best_loss = float(va[0])
                best_state = {key: value.detach().cpu().clone() for key, value in
                              self.network.state_dict().items()}
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.network.load_state_dict(best_state)
        self.network.eval()
        lines.append("best_epoch={} best_valid_point_mse={:.8f}".format(
            best_epoch, best_loss))
        self.validation_metrics = self._validation_metrics(
            valid_cube, valid_mask, valid_wind_mask, valid_issues)
        if hasattr(self, "log_path"):
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _validation_metrics(self, cube, mask_power, mask_wind, issues) -> dict:
        p_index = self.features.index("Patv")
        p_lo = float(self.scaler.minimum[p_index])
        p_span = float(max(self.scaler.maximum[p_index] - p_lo, 1e-6))
        absolute = 0.0
        n_power = 0
        direct = 0.0
        n_direct = 0
        masked_consist = 0.0
        n_consist = 0
        from .power_curve import logistic_numpy
        for x, yp, _, mp, mw in self._pin_batches(
                cube, mask_power, mask_wind, issues, False):
            forecast = self.predict_batch_with_wind(x)
            power, wind = forecast["power"], forecast["wind_speed"]
            true = yp * p_span + p_lo
            absolute += float(np.abs(power - true)[mp].sum(dtype=np.float64))
            n_power += int(mp.sum())
            curve = logistic_numpy(wind, self.curve_params)
            difference = np.abs(power - curve)
            direct += float(difference.sum(dtype=np.float64))
            n_direct += int(power.size)
            both = mp & mw
            masked_consist += float(difference[both].sum(dtype=np.float64))
            n_consist += int(both.sum())
        return {"MAE_kW": absolute / n_power if n_power else None,
                "n": n_power,
                "consist_masked_kW": masked_consist / n_consist if n_consist else None,
                "consist_masked_n": n_consist,
                "direct_consist_error_kW": direct / n_direct if n_direct else None,
                "direct_consist_n": n_direct}

    def predict_batch_with_wind(self, histories: np.ndarray) -> dict:
        torch = self.torch
        x = np.nan_to_num(histories, nan=0.0, posinf=0.0,
                          neginf=0.0).astype(np.float32)
        if x.ndim != 4 or x.shape[1] != len(self.turbine_ids) or (
                x.shape[2] != len(self.features)):
            raise ValueError("PIN history shape or turbine order is invalid")
        output = []
        self.network.eval()
        with torch.no_grad():
            for first in range(0, len(x), self.batch):
                part = torch.from_numpy(x[first:first + self.batch]).to(self.device)
                output.append(self.network(part).cpu().numpy())
        pred = np.concatenate(output)
        h = self.config.horizon
        p_index = self.features.index("Patv")
        v_index = self.features.index("Wspd")
        p_lo = float(self.scaler.minimum[p_index])
        p_span = float(max(self.scaler.maximum[p_index] - p_lo, 1e-6))
        v_lo = float(self.scaler.minimum[v_index])
        v_span = float(max(self.scaler.maximum[v_index] - v_lo, 1e-6))
        power = np.clip(pred[..., :h] * p_span + p_lo,
                        0, self.config.rated_power_kw)
        wind = np.clip(pred[..., h:] * v_span + v_lo, 0, 30)
        return {"power": power.astype(np.float32),
                "wind_speed": wind.astype(np.float32)}

    def predict_batch(self, histories: np.ndarray, phases=None) -> np.ndarray:
        return self.predict_batch_with_wind(histories)["power"]

    def export_interpret(self, path: Path) -> Optional[Path]:
        if self.wake_prior is None:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.torch.no_grad():
            learned = self.network.adaptive_adjacency().cpu().numpy()
        np.savez_compressed(path, A_learned=learned.astype(np.float32),
                            A_wake=self.wake_prior["A_wake"],
                            A_wake_prior=self.wake_prior["A_wake_prior"],
                            wind_dirs=self.wake_prior["wind_dirs"],
                            w_k=self.wake_prior["w_k"],
                            turbine_ids=self.turbine_ids,
                            convention=np.asarray(self.wake_convention))
        return path


VARIANTS = {
    "ours": {"lambda_consist": 0.1, "lambda_wake": 0.1},
    "ours_no_consist": {"lambda_consist": 0.0, "lambda_wake": 0.1},
    "ours_no_wake": {"lambda_consist": 0.1, "lambda_wake": 0.0},
    "barest": {"lambda_consist": 0.0, "lambda_wake": 0.0},
}


def _register_variant(name: str, defaults: dict) -> None:
    def __init__(self, lambda_consist=None, lambda_wake=None, **kwargs):
        PINForecaster.__init__(
            self,
            lambda_consist=defaults["lambda_consist"] if lambda_consist is None
            else lambda_consist,
            lambda_wake=defaults["lambda_wake"] if lambda_wake is None
            else lambda_wake,
            **kwargs)
    variant = type("{}Forecaster".format(name.title().replace("_", "")),
                   (PINForecaster,), {"__init__": __init__})
    register_model(name)(variant)


for _name, _defaults in VARIANTS.items():
    _register_variant(_name, _defaults)
