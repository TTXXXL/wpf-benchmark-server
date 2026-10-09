"""Portable inference checkpoints for registered PyTorch forecasters."""
from __future__ import annotations

from dataclasses import asdict
import inspect
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..data.scaling import FeatureScaler
from ..evaluation.config import ProtocolConfig


FORMAT_VERSION = 1


def save_checkpoint(model, path, metadata=None) -> Path:
    """Atomically save the fitted network and everything required for inference.

    The training loop has already restored its validation-best state. Optimizer
    and random-generator states are deliberately absent: this is not a training
    resume checkpoint. Only tensors and ordinary Python values are serialized.
    """
    if not hasattr(model, "network") or model.validation_metrics is None:
        raise ValueError("Fit the neural forecaster before saving a checkpoint")
    torch = model.torch
    description = json.loads(json.dumps({
        "format_version": FORMAT_VERSION,
        "model": model.name,
        "model_config": model.model_config,
        "protocol": asdict(model.config),
        "features": list(model.features),
        "history_scale": model.history_scale,
        "scaler": {"features": list(model.scaler.features),
                   "minimum": model.scaler.minimum.tolist(),
                   "maximum": model.scaler.maximum.tolist()},
        "turbine_ids": model.turbine_ids.tolist(),
        "validation_metrics": model.validation_metrics,
        "metadata": metadata or {},
        "torch_version": str(torch.__version__),
    }, allow_nan=False))
    description["state_dict"] = {
        key: value.detach().cpu().clone()
        for key, value in model.network.state_dict().items()}
    description["extra_state"] = model._checkpoint_extra_state()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                            dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(description, handle)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_checkpoint(path, device="cpu"):
    """Reconstruct a registered forecaster without training or data/metadata files."""
    import torch
    from . import get_model
    from .neural import NeuralForecaster

    options = {"map_location": "cpu"}
    # PyTorch 1.x (supported on Python 3.8) may lack this restricted loader.
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = True
    saved = torch.load(Path(path), **options)
    if not isinstance(saved, dict) or saved.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported forecaster checkpoint format")
    model_class = get_model(saved["model"])
    if not issubclass(model_class, NeuralForecaster):
        raise ValueError("Checkpoint must describe a neural forecaster")
    model = model_class(**saved["model_config"])
    if tuple(saved["features"]) != tuple(model.features) or (
            saved["history_scale"] != model.history_scale):
        raise ValueError("Checkpoint feature order or history scale differs from model")
    scale = saved["scaler"]
    minimum = np.asarray(scale["minimum"], dtype=np.float32)
    maximum = np.asarray(scale["maximum"], dtype=np.float32)
    if tuple(scale["features"]) != tuple(model.features) or (
            minimum.shape != (len(model.features),) or maximum.shape != minimum.shape or
            not np.isfinite(minimum).all() or not np.isfinite(maximum).all() or
            np.any(maximum < minimum)):
        raise ValueError("Invalid checkpoint feature scales")
    turbine_ids = np.asarray(saved["turbine_ids"], dtype=np.int64)
    if turbine_ids.ndim != 1 or not turbine_ids.size or (
            not np.array_equal(turbine_ids, saved["turbine_ids"]) or
            np.any(np.diff(turbine_ids) <= 0)):
        raise ValueError("Invalid checkpoint turbine order")
    protocol = dict(saved["protocol"])
    protocol["exclude_flags_main"] = tuple(protocol["exclude_flags_main"])
    protocol["wind_bins"] = tuple(tuple(pair) for pair in protocol["wind_bins"])
    model.configure(ProtocolConfig(**protocol),
                    FeatureScaler(tuple(model.features), minimum, maximum))
    model.torch = torch
    model.device = torch.device(device)
    model.turbine_ids = turbine_ids
    model.network = model._build_network(torch, len(turbine_ids)).to(model.device)
    model.network.load_state_dict(saved["state_dict"], strict=True)
    model.network.eval()
    model.n_params = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    model.validation_metrics = saved["validation_metrics"]
    model.checkpoint_metadata = saved["metadata"]
    model._restore_checkpoint_extra_state(saved["extra_state"])
    return model
