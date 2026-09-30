"""Shared PyTorch training loop for normalized deep forecasting baselines."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .training import issue_indices, training_cube


class NeuralForecaster(BaseForecaster):
    features = ("Wspd", "Wsin", "Wcos", "Patv")
    history_scale = "normalized"
    graph_model = False
    loss_name = "mse"

    def __init__(self, hidden: int, layers: int = 2, dropout: float = 0.1,
                 lr: float = 0.001, batch: int = 256, epochs: int = 30,
                 patience: int = 5, stride: int = 6, weight_decay: float = 0.0):
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.batch = int(batch)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.stride = int(stride)
        self.weight_decay = float(weight_decay)
        if min(self.hidden, self.layers, self.batch, self.epochs,
               self.patience, self.stride) <= 0 or self.batch > 512 or self.epochs > 30:
            raise ValueError("Positive hyperparameters, batch <= 512 and epochs <= 30 are required")
        if self.lr <= 0 or self.dropout < 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")

    @property
    def model_config(self) -> dict:
        return {key: getattr(self, key) for key in (
            "hidden", "layers", "dropout", "lr", "batch", "epochs",
            "patience", "stride", "weight_decay")}

    def _build_network(self, torch, n_turbines: int):
        raise NotImplementedError

    def _normalized_zero_power(self):
        j = self.features.index("Patv")
        span = max(self.scaler.maximum[j] - self.scaler.minimum[j], 1e-6)
        return float(-self.scaler.minimum[j] / span)

    def _network_histories(self, histories):
        # Residual networks select the last finite Patv before filling inputs.
        # Preserve missing values identically in training and inference.
        if getattr(self, "current_power_skip", False):
            return histories
        return np.nan_to_num(histories, nan=0.0, posinf=0.0, neginf=0.0)

    def _batches(self, cube, valid_target, target, issues, shuffle):
        w, h = self.config.input_window, self.config.horizon
        past = np.arange(w) - w + 1
        future = np.arange(1, h + 1)
        if self.graph_model:
            selected = np.random.permutation(issues) if shuffle else issues
            for first in range(0, len(selected), self.batch):
                take = selected[first:first + self.batch]
                x = cube[take[:, None] + past[None, :]].transpose(0, 2, 3, 1)
                y = target[take[:, None] + future[None, :]].transpose(0, 2, 1)
                m = valid_target[take[:, None] + future[None, :]].transpose(0, 2, 1)
                m &= np.isfinite(y)
                yield self._network_histories(x), np.nan_to_num(y), m
        else:
            n = cube.shape[1]
            pairs = np.arange(len(issues) * n)
            if shuffle:
                pairs = np.random.permutation(pairs)
            for first in range(0, len(pairs), self.batch):
                batch = pairs[first:first + self.batch]
                take = issues[batch // n]
                turbines = batch % n
                x = cube[take[:, None] + past[None, :], turbines[:, None]].transpose(0, 2, 1)
                y = target[take[:, None] + future[None, :], turbines[:, None]]
                m = valid_target[take[:, None] + future[None, :], turbines[:, None]]
                m &= np.isfinite(y)
                yield np.nan_to_num(x), np.nan_to_num(y), m

    def fit(self, train: pd.DataFrame, valid: Optional[pd.DataFrame] = None) -> None:
        if valid is None or valid.empty:
            raise ValueError("Deep baselines require validation data for early stopping")
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Deep baselines require PyTorch; install wpf-benchmark[deep]") from exc
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, train_tids, train_cube, train_mask, train_target, _ = training_cube(
            train, self.features, self.config, self.scaler)
        _, valid_tids, valid_cube, valid_mask, valid_target, _ = training_cube(
            valid, self.features, self.config, self.scaler)
        if not np.array_equal(train_tids, valid_tids):
            raise ValueError("Training and validation turbine grids differ")
        self.turbine_ids = train_tids
        train_issues = issue_indices(len(train_cube), self.config.input_window,
                                     self.config.horizon, self.stride)
        valid_issues = issue_indices(len(valid_cube), self.config.input_window,
                                     self.config.horizon, self.stride)
        if not len(train_issues) or not len(valid_issues):
            raise ValueError("Training or validation split is too short for windows")
        self.network = self._build_network(torch, len(train_tids)).to(self.device)
        self.n_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        no_improve = 0
        lines = ["model={} device={} train_windows={} valid_windows={} params={}".format(
            self.name, self.device, len(train_issues), len(valid_issues), self.n_params)]
        if hasattr(self, "current_power_skip"):
            lines.append("loss={} current_power_skip={} power_anchor=latest_finite_history "
                         "power_anchor_fallback=zero_kw".format(self.loss_name, self.current_power_skip))
        for epoch in range(1, self.epochs + 1):
            metrics = []
            for training, c, m, y, issues in (
                (True, train_cube, train_mask, train_target, train_issues),
                (False, valid_cube, valid_mask, valid_target, valid_issues)):
                self.network.train(training)
                error_sum = 0.0
                n_valid = 0
                for x, y, mask in self._batches(c, m, y, issues, training):
                    if not mask.any():
                        continue
                    xt = torch.from_numpy(x.astype(np.float32, copy=False)).to(self.device)
                    yt = torch.from_numpy(y.astype(np.float32, copy=False)).to(self.device)
                    mt = torch.from_numpy(mask).to(self.device)
                    with torch.set_grad_enabled(training):
                        pred = self.network(xt)
                        error = pred - yt
                        pointwise = error.abs() if self.loss_name == "mae" else error.square()
                        loss_sum = (pointwise * mt).sum()
                        loss = loss_sum / mt.sum().clamp(min=1)
                        if training:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    error_sum += float(loss_sum.detach().cpu())
                    n_valid += int(mt.sum().item())
                if n_valid == 0:
                    raise ValueError("No valid targets in {} split".format(
                        "training" if training else "validation"))
                metrics.append(error_sum / n_valid)
            lines.append("epoch={} train_{}={:.8f} valid_{}={:.8f}".format(
                epoch, self.loss_name, metrics[0], self.loss_name, metrics[1]))
            if metrics[1] < best_loss - 1e-8:
                best_loss = metrics[1]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in
                              self.network.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        self.network.load_state_dict(best_state)
        self.network.eval()
        lines.append("best_epoch={} best_valid_{}={:.8f} stopped_epoch={}".format(
            best_epoch, self.loss_name, best_loss, epoch))
        if hasattr(self, "log_path"):
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def predict(self, history: np.ndarray) -> np.ndarray:
        return self.predict_batch(history[None])[0]

    def predict_batch(self, histories: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = self._network_histories(histories).astype(np.float32)
        b, n, f, w = x.shape
        if self.graph_model:
            if n != len(self.turbine_ids):
                raise ValueError("AGCRN turbine grid differs from training")
            batches = [x[i:i + self.batch] for i in range(0, b, self.batch)]
        else:
            flat = x.reshape(b * n, f, w)
            batches = [flat[i:i + self.batch] for i in range(0, len(flat), self.batch)]
        output = []
        self.network.eval()
        with torch.no_grad():
            for part in batches:
                output.append(self.network(torch.from_numpy(part).to(self.device)).cpu().numpy())
        pred = np.concatenate(output).reshape(b, n, self.config.horizon)
        j = self.features.index("Patv")
        low = self.scaler.minimum[j]
        span = max(self.scaler.maximum[j] - low, 1e-6)
        return np.clip(pred * span + low, 0, self.config.rated_power_kw).astype(np.float32)
