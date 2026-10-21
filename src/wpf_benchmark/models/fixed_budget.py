"""Opt-in Lite trajectories. Selection is performed only after training."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import random
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from .training import issue_indices, training_cube
from ..evaluation.selection import (audit_targets, grid_digest, json_digest,
                                    predict_grid, sha256, validation_scores, write_json)


@contextmanager
def preserve_rng(model):
    """Validation and portable reconstruction cannot advance training RNGs."""
    torch = model.torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mode = model.network.training
    try:
        model.network.eval()
        with torch.no_grad():
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        model.network.train(mode)


def state_digest(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def fit_trajectory(model, train, valid):
    import torch
    from .checkpoint import save_epoch_checkpoint, load_checkpoint

    if model.name != "agcrn_lite" or not hasattr(model, "trajectory_context"):
        raise ValueError("Fixed budget mode requires a Lite trajectory runner context")
    context = model.trajectory_context
    directory = context["directory"]
    # Never reuse an incomplete attempt, even if no epoch was saved.
    directory.mkdir(parents=True, exist_ok=False)
    model.torch = torch
    model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audit_targets(train, model.config)
    _, tids, cube, mask, target, _ = training_cube(
        train, model.features, model.config, model.scaler)
    grid = context["grid"]
    if not np.array_equal(tids, grid["turbine_ids"]):
        raise ValueError("Training and selection turbines differ")
    model.turbine_ids = tids
    issues = issue_indices(len(cube), model.config.input_window, model.config.horizon, model.stride)
    if not len(issues):
        raise ValueError("No training windows")
    model.network = model._build_network(torch, len(tids)).to(model.device)
    model.n_params = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.network.parameters(), lr=model.lr, weight_decay=model.weight_decay)
    identity = dict(context["metadata"], target_mask=model.config.target_mask,
                    budget_epochs=model.epochs, selection_days=context["selection_days"],
                    validation_grid_digest=grid_digest(grid),
                    initial_state_digest=state_digest(model.network.state_dict()),
                    training_input_sha256=hashlib.sha256(cube.tobytes() + tids.tobytes() + issues.tobytes()).hexdigest(),
                    scaler_digest=json_digest(dict(features=list(model.scaler.features),
                        minimum=model.scaler.minimum.tolist(), maximum=model.scaler.maximum.tolist())))
    manifest = dict(schema=1, identity=identity, status="running", last_completed_epoch=0,
                    candidates=[], selection_mode="fixed_budget_dual_mask", patience_ignored=True,
                    model_config=model.model_config, training_windows=len(issues),
                    optimizer="Adam", device=str(model.device))
    status_path = directory / "trajectory.json"
    write_json(status_path, manifest)
    history_path = directory / "history.jsonl"
    history_path.touch(exist_ok=False)
    # Include only issue rows contributing to the selection target days.
    from ..evaluation.selection import day_mask
    selection_rows = np.flatnonzero(day_mask(grid, context["selection_days"])[:, 0].any(axis=1))
    selection_grid = {key: (value[selection_rows] if key != "turbine_ids" else value)
                      for key, value in grid.items()}
    selection_issues = context["issues"][selection_rows]
    total_updates = 0
    started = perf_counter()
    try:
        for epoch in range(1, model.epochs + 1):
            epoch_started = perf_counter()
            model.network.train()
            sums, count, skipped, updates = {}, 0, 0, 0
            for x, y, selected in model._batches(cube, mask, target, issues, True):
                if not selected.any():
                    skipped += 1
                    continue
                xt = torch.from_numpy(x.astype(np.float32, copy=False)).to(model.device)
                yt = torch.from_numpy(y.astype(np.float32, copy=False)).to(model.device)
                mt = torch.from_numpy(selected).to(model.device)
                optimizer.zero_grad()
                loss_sums = model._loss_sums(xt, yt, mt)
                loss = loss_sums["total"] / mt.sum().clamp(min=1)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                for key, value in loss_sums.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
                count += int(mt.sum().item())
                updates += 1
            if count == 0:
                raise ValueError("No valid training targets")
            total_updates += updates
            state_before = state_digest(model.network.state_dict())
            with preserve_rng(model):
                prediction = predict_grid(model, context["cube"], selection_issues)
                scores = validation_scores(prediction, selection_grid, context["selection_days"])
                metrics = dict(epoch=epoch, selection_mode="fixed_budget_dual_mask", validation=scores)
                metadata = dict(identity, epoch=epoch, epoch_state_digest=state_before)
                checkpoint = directory / ("epoch_{:03d}.pt".format(epoch))
                if checkpoint.exists():
                    raise ValueError("Candidate checkpoint already exists")
                save_epoch_checkpoint(model, checkpoint, model.network.state_dict(), metrics, metadata)
                # Same small causal input, same device, bit-for-bit round trip.
                take = selection_issues[:min(model.batch, len(selection_issues))]
                offsets = take[:, None] + np.arange(model.config.input_window) - model.config.input_window + 1
                probe = model.scaler.transform_histories(context["cube"][offsets].transpose(0, 2, 3, 1))
                expected = model.predict_batch(probe)
                restored = load_checkpoint(checkpoint, device=str(model.device))
                actual = restored.predict_batch(probe)
                if not np.array_equal(expected, actual) or state_digest(restored.network.state_dict()) != state_before:
                    raise ValueError("Epoch checkpoint round-trip differs from its validation state")
                probe_sha = hashlib.sha256(expected.tobytes()).hexdigest()
                del restored
            if state_digest(model.network.state_dict()) != state_before:
                raise ValueError("Validation changed network state")
            row = dict(epoch=epoch, identity=identity, train={k: v / count for k, v in sums.items()},
                       optimizer_updates=updates, cumulative_optimizer_updates=total_updates,
                       effective_target_cells=count, skipped_empty_batches=skipped, validation=scores,
                       checkpoint=checkpoint.name, checkpoint_sha256=sha256(checkpoint),
                       epoch_state_digest=state_before, probe_prediction_sha256=probe_sha,
                       seconds=perf_counter() - epoch_started,
                       cuda_peak_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
            manifest["last_completed_epoch"] = epoch
            manifest["candidates"].append(dict(epoch=epoch, path=checkpoint.name, sha256=row["checkpoint_sha256"]))
            write_json(status_path, manifest)
            print("trajectory={} epoch={}/{} train_mae={:.8f} val_m1={:.12f} val_m2={:.12f}".format(
                identity["trajectory_id"], epoch, model.epochs, row["train"]["mae"],
                scores["m1"]["equal_horizon"]["MAE_kW"], scores["m2"]["equal_horizon"]["MAE_kW"]), flush=True)
        manifest.update(status="complete", seconds=perf_counter() - started,
                        history_sha256=sha256(history_path))
        write_json(status_path, manifest)
    except BaseException as error:
        manifest.update(status="failed", error="{}: {}".format(type(error).__name__, error),
                        seconds=perf_counter() - started)
        write_json(status_path, manifest)
        raise
    model.network.eval()
    model.validation_metrics = dict(selection_mode="fixed_budget_dual_mask", stopped_epoch=model.epochs,
                                    no_best_state_restored=True)
    model.trajectory_manifest = manifest
    if hasattr(model, "log_path"):
        model.log_path.parent.mkdir(parents=True, exist_ok=True)
        model.log_path.write_text("trajectory={} complete_epochs={} history={}\n".format(
            identity["trajectory_id"], model.epochs, history_path), encoding="utf-8")


def verify_trajectory(directory, expected=None):
    """Audit every epoch payload and its full-precision score, fail closed."""
    from .checkpoint import load_checkpoint
    from ..evaluation.selection import select_epoch
    directory = Path(directory)
    manifest = json.loads((directory / "trajectory.json").read_text("utf-8"))
    identity = manifest["identity"]
    budget = identity["budget_epochs"]
    if manifest["status"] != "complete" or manifest["last_completed_epoch"] != budget:
        raise ValueError("Incomplete training trajectory")
    if expected and any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Trajectory identity differs from frozen batch")
    history = directory / "history.jsonl"
    if sha256(history) != manifest["history_sha256"]:
        raise ValueError("History hash changed")
    rows = [json.loads(line) for line in history.read_text("utf-8").splitlines()]
    for selector in ("m1", "m2"):
        select_epoch(rows, selector, budget)
    if len(manifest["candidates"]) != budget or len(list(directory.glob("epoch_*.pt"))) != budget:
        raise ValueError("Candidate checkpoint set is incomplete or contains extras")
    configs, scaler_digests = [], []
    cumulative = 0
    for row, candidate in zip(rows, manifest["candidates"]):
        name = "epoch_{:03d}.pt".format(row["epoch"])
        if row["identity"] != identity or candidate != dict(
                epoch=row["epoch"], path=name, sha256=row["checkpoint_sha256"]) or row["checkpoint"] != name:
            raise ValueError("Candidate row identity mismatch")
        if sha256(directory / name) != candidate["sha256"]:
            raise ValueError("Candidate checkpoint hash changed")
        cumulative += row["optimizer_updates"]
        if row["optimizer_updates"] <= 0 or row["cumulative_optimizer_updates"] != cumulative:
            raise ValueError("Invalid optimizer update accounting")
        model = load_checkpoint(directory / name)
        if model.checkpoint_metadata != dict(identity, epoch=row["epoch"],
                                              epoch_state_digest=row["epoch_state_digest"]):
            raise ValueError("Checkpoint epoch or trajectory metadata differs")
        if (model.validation_metrics != dict(epoch=row["epoch"],
                selection_mode="fixed_budget_dual_mask", validation=row["validation"]) or
                state_digest(model.network.state_dict()) != row["epoch_state_digest"]):
            raise ValueError("Checkpoint state and validation score differ")
        if (model.config.target_mask != identity["target_mask"] or model.epochs != budget or
                model.selection_mode != "fixed_budget_dual_mask" or
                model.selection_days != identity["selection_days"]):
            raise ValueError("Checkpoint training configuration differs")
        from ..figures.io import config_digest
        if config_digest(model.config, model.model_config) != identity["config_digest"]:
            raise ValueError("Checkpoint config fingerprint differs")
        configs.append(json_digest(dict(protocol=asdict(model.config),
                                        model=model.model_config)))
        scaler_digests.append(json_digest(dict(features=list(model.scaler.features),
                            minimum=model.scaler.minimum.tolist(), maximum=model.scaler.maximum.tolist())))
    if len(set(configs)) != 1 or set(scaler_digests) != {identity["scaler_digest"]}:
        raise ValueError("Candidate configurations or scalers differ")
    result = json.loads((directory / "result.json").read_text("utf-8"))
    if (result["model_config"] != model.model_config or
            result["config"] != json.loads(json.dumps(asdict(model.config)))):
        raise ValueError("Run result configuration differs from candidates")
    for key in ("seed", "target_mask", "run_id", "experiment", "data_digest", "code_digest"):
        if result[key] != identity[key]:
            raise ValueError("Run result and trajectory identity differ")
    if (result["trajectory_manifest_sha256"] != sha256(directory / "trajectory.json") or
            result["grid_sha256"] != sha256(directory / "validation_grid.npz") or
            result["inputs_sha256"] != sha256(directory / "inference_inputs.npz")):
        raise ValueError("Run result artifact hashes differ")
    with np.load(directory / "validation_grid.npz", allow_pickle=False) as data:
        if grid_digest(data) != identity["validation_grid_digest"]:
            raise ValueError("Validation grid digest differs")
        with np.load(directory / "inference_inputs.npz", allow_pickle=False) as inputs:
            if (not np.array_equal(inputs["times"][inputs["issues"]], data["issue_times"]) or
                    not np.array_equal(inputs["turbine_ids"], data["turbine_ids"]) or
                    not np.array_equal(inputs["features"], model.features)):
                raise ValueError("Inference inputs and frozen grid differ")
    return manifest, rows


def select_trajectory(directory, expected=None):
    from ..evaluation.selection import select_epoch, TOLERANCE_KW
    directory = Path(directory)
    manifest, rows = verify_trajectory(directory, expected)
    budget = manifest["identity"]["budget_epochs"]
    selection = dict(schema=1, identity=manifest["identity"],
                     trajectory_manifest_sha256=sha256(directory / "trajectory.json"),
                     history_sha256=manifest["history_sha256"],
                     candidate_set_sha256=json_digest(manifest["candidates"]),
                     scores=[dict(epoch=row["epoch"], m1=row["validation"]["m1"]["equal_horizon"]["MAE_kW"],
                                  m2=row["validation"]["m2"]["equal_horizon"]["MAE_kW"]) for row in rows],
                     metric="MAE_kW_equal_horizon", tolerance_kw=TOLERANCE_KW,
                     tie="earliest epoch within 1e-8 kW of global minimum", selected={})
    for mask in ("m1", "m2"):
        epoch = select_epoch(rows, mask, budget)
        source = directory / rows[epoch - 1]["checkpoint"]
        target = directory / ("selected_by_" + mask + ".pt")
        expected_hash = rows[epoch - 1]["checkpoint_sha256"]
        if target.exists() and sha256(target) != expected_hash:
            raise ValueError("Selected alias changed")
        if not target.exists():
            temporary = target.with_suffix(".pt.tmp")
            shutil.copyfile(source, temporary)
            if sha256(temporary) != expected_hash:
                raise ValueError("Selected copy failed hash verification")
            temporary.replace(target)
        selection["selected"][mask] = dict(epoch=epoch, path=target.name, sha256=expected_hash,
            score_kw=rows[epoch - 1]["validation"][mask]["equal_horizon"]["MAE_kW"])
    destination = directory / "selection_manifest.json"
    if destination.exists() and json.loads(destination.read_text("utf-8")) != selection:
        raise ValueError("Existing selection manifest differs")
    write_json(destination, selection)
    return selection
