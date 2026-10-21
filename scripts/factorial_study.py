"""Schema-3 fixed-budget supervision x selection study, dispatched by run_study."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import numpy as np
import pandas as pd

import rerun_m1 as suite
from wpf_benchmark.data.io import load_clean, load_meta
from wpf_benchmark.data.scaling import FeatureScaler
from wpf_benchmark.data.schema import FLAG_COLUMNS, OFFICIAL_COLUMNS
from wpf_benchmark.evaluation.config import ProtocolConfig
from wpf_benchmark.evaluation.selection import (audit_targets, build_grid, grid_digest,
                                                json_digest, sha256, write_json)
from wpf_benchmark.models import get_model
from wpf_benchmark.models.fixed_budget import verify_trajectory, select_trajectory
from wpf_benchmark.paths import ProjectPaths


def load_plan(root, study):
    from run_study import relative_file, NAME
    import re
    if not re.fullmatch(NAME, study):
        raise ValueError("Invalid study name")
    plan = suite.read_json(root / "configs/studies" / (study + ".json"))
    if (plan["schema"] != 3 or not re.fullmatch(NAME, plan["experiment"]) or
            plan["seeds"] != [0, 1, 2] or plan["model"] != "agcrn_lite" or
            plan["selection_days"] != [197, 208] or plan["scoring_days"] != [209, 221] or
            [variant["target_mask"] for variant in plan["variants"]] != ["m1", "m2"]):
        raise ValueError("Factorial protocol differs from the registered six-trajectory design")
    jobs = []
    for seed in plan["seeds"]:
        for variant in plan["variants"]:
            mask = variant["target_mask"]
            name = variant["config"]
            recipe = suite.read_json(relative_file(root, name))
            cfg = ProtocolConfig(**recipe["protocol"])
            model = get_model("agcrn_lite")(**recipe["model"])
            if (cfg != ProtocolConfig(target_mask=mask) or model.epochs != 30 or
                    model.selection_mode != "fixed_budget_dual_mask" or model.selection_days != plan["selection_days"]):
                raise ValueError("Factorial recipe changed")
            args = ["run", "--model", "agcrn_lite", "--config", name,
                    "--target-mask", mask, "--seed", str(seed), "--repeat", "1",
                    "--experiment", plan["experiment"], "--tag", "factorial_{}_s{}".format(mask, seed),
                    "--validation-only", "--save-checkpoint", "--no-save-arrays", "--no-plots"]
            jobs.append(dict(key="{}_s{}".format(mask, seed), seed=seed, target_mask=mask,
                             config=name, args=args, protocol=json.loads(json.dumps(asdict(cfg))),
                             model_config=model.model_config))
    first = suite.read_json(root / jobs[0]["config"])
    second = suite.read_json(root / jobs[1]["config"])
    first["protocol"]["target_mask"] = "m2"
    if first != second:
        raise ValueError("The two recipes may differ only in target_mask")
    return plan, jobs


def asset_audit(root, plan):
    cfg = ProtocolConfig()
    features = ("Wspd", "Wsin", "Wcos", "Patv")
    paths = ProjectPaths(root)
    columns = list(dict.fromkeys(("ts", "Day", "TurbID") + features + FLAG_COLUMNS + OFFICIAL_COLUMNS))
    frame = load_clean(paths, columns=columns, filters=[("Day", "<=", 221)])
    if frame.Day.min() != 1 or frame.Day.max() != 221 or frame.TurbID.nunique() != 134:
        raise ValueError("Frozen SDWPF population or day coverage changed")
    counts = {"train": audit_targets(frame[frame.Day <= 196], cfg),
              "validation": audit_targets(frame[frame.Day > 196], cfg)}
    # Check uniqueness, turbine grid and day coverage even in training data.
    from wpf_benchmark.data.windows import to_feature_cube
    times, _ = to_feature_cube(frame, ["Day"], np.sort(frame.TurbID.unique()))
    if len(times) != 221 * 144 or not np.all(np.diff(times.astype("timedelta64[ns]").astype(np.int64)) == 600_000_000_000):
        raise ValueError("Frozen SDWPF time grid changed")
    meta = load_meta(paths)
    use_meta = meta.get("train_days") == 196
    scaler = (FeatureScaler.from_meta(meta, features) if use_meta else
              FeatureScaler.from_training(frame[frame.Day <= 196], features))
    grid, _, _ = build_grid(frame[frame.Day >= 196], cfg, features)
    digest = grid_digest(grid)
    if digest != plan["validation_grid_digest"]:
        raise ValueError("Dense validation grid differs from the previous frozen members/truth/anchor")
    return dict(counts=counts, validation_grid_digest=digest,
                selection_cells={mask: int((grid["valid_" + mask] &
                    ((grid["target_days"] >= 197) & (grid["target_days"] <= 208))[:, None, :]).sum())
                    for mask in ("m1", "m2")},
                scaler_source="sdwpf_meta.json" if use_meta else "configured_training_days",
                scaler_digest=json_digest(dict(features=list(scaler.features),
                    minimum=scaler.minimum.tolist(), maximum=scaler.maximum.tolist())))


def identity(root, study, plan, versions):
    from run_study import snapshot
    frozen = snapshot(root, study, plan, versions)
    for name, expected in plan["support_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError("Frozen study support source differs: " + name)
        frozen["sources"][name] = expected
    return frozen


def environment(root, versions):
    import torch
    def git(*args):
        result = subprocess.run(["git"] + list(args), cwd=str(root), capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(versions, cuda_version=torch.version.cuda,
                cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                git_head=git("rev-parse", "HEAD"), git_status=git("status", "--porcelain"),
                deterministic_cudnn=True, cudnn_benchmark=False,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def preflight(root, study, plan, versions):
    frozen = identity(root, study, plan, versions)  # Hashes before reading data.
    audit = asset_audit(root, plan)
    model = get_model("agcrn_lite")(**suite.read_json(root / plan["config"])["model"])
    meta = load_meta(ProjectPaths(root))
    if audit["scaler_source"] == "sdwpf_meta.json":
        scaler = FeatureScaler.from_meta(meta, model.features)
    else:
        scaler = FeatureScaler(tuple(model.features), np.zeros(4, np.float32), np.ones(4, np.float32))
    model.configure(ProtocolConfig(), scaler)
    import torch
    # Preserve global RNG because preflight must not select initialization.
    with torch.random.fork_rng(devices=[]):
        network = model._build_network(torch, 134)
    state_bytes = sum(value.numel() * value.element_size() for value in network.state_dict().values())
    required = int(6 * (32 * (state_bytes + 65536) + 128 * 1024 ** 2) * 1.25 + 1024 ** 3)
    # Space for both all candidates and a delivery ZIP; weights compress poorly.
    required *= 2
    free = shutil.disk_usage(root).free
    if free < required:
        raise ValueError("Insufficient disk space: need {} bytes; free {}".format(required, free))
    audit.update(required_free_bytes=required, free_bytes=free, per_epoch_state_bytes=state_bytes)
    frozen["asset_audit"] = {k: v for k, v in audit.items() if k not in ("free_bytes",)}
    return frozen, audit


def expected_job(plan, job):
    return dict(experiment=plan["experiment"], seed=job["seed"], target_mask=job["target_mask"],
                budget_epochs=30, selection_days=plan["selection_days"],
                validation_grid_digest=plan["validation_grid_digest"], **plan["digests"])


def checked_job(root, directory, plan, job, required=False):
    marker = directory / (job["key"] + ".done.json")
    if not marker.exists():
        if required:
            raise ValueError("Incomplete job: " + job["key"])
        return None
    completed = suite.read_json(marker)
    trajectory = root / completed["trajectory"]
    result = suite.read_json(trajectory / "result.json")
    if sha256(trajectory / "result.json") != completed["result_sha256"]:
        raise ValueError("Completed result changed")
    manifest, _ = verify_trajectory(trajectory, expected_job(plan, job))
    if (result["config"] != job["protocol"] or result["model_config"] != job["model_config"] or
            sha256(root / result["log_path"]) != result["log_sha256"]):
        raise ValueError("Training recipe or log differs")
    return trajectory, manifest


def deliver(root, directory, plan, jobs):
    files = {}
    for job in jobs:
        trajectory, _ = checked_job(root, directory, plan, job, True)
        for path in trajectory.iterdir():
            if path.is_file() and path.suffix in (".json", ".jsonl", ".pt", ".npz"):
                files[path.relative_to(root).as_posix()] = sha256(path)
        result = suite.read_json(trajectory / "result.json")
        files[result["log_path"]] = sha256(root / result["log_path"])
    # Explicit retries retain and return earlier failed attempts as evidence.
    for path in (root / "reports/trajectories" / plan["experiment"]).rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".pt", ".npz"):
            files[path.relative_to(root).as_posix()] = sha256(path)
    for path in directory.iterdir():
        if path.is_file() and path.suffix in (".json", ".log") and path.name != "delivery_manifest.json":
            files[path.relative_to(root).as_posix()] = sha256(path)
    frozen = suite.read_json(directory / "manifest.json")
    for name, expected in frozen["sources"].items():
        if sha256(root / name) != expected:
            raise ValueError("Source changed during delivery")
        files[name] = expected
    output_manifest = directory / "delivery_manifest.json"
    attempts = [attempt for path in directory.glob("*.attempts.json") for attempt in suite.read_json(path)]
    write_json(output_manifest, dict(schema=1, trajectories=6, candidates=180, selections=12,
               attempts=len(attempts), failed_attempts=sum(a["status"] == "failed" for a in attempts),
               files=files, analysis="Score Day209-221 locally; development mechanism study only"))
    files[output_manifest.relative_to(root).as_posix()] = sha256(output_manifest)
    target = directory / (plan["experiment"] + "_delivery.zip")
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sorted(files.items()):
            if sha256(root / name) != expected:
                raise ValueError("Artifact changed during delivery")
            archive.write(root / name, name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip():
            raise ValueError("Delivery CRC failed")
    temporary.replace(target)
    print("COMPLETE 6/6; candidates=180; selected=12\nDELIVERY: " + str(target), flush=True)
    return target


def run(root, study, args, versions):
    plan, jobs = load_plan(root, study)
    frozen, audit = preflight(root, study, plan, versions)
    print(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(),
        processed={name: dict(path=str(root / name), size=(root / name).stat().st_size, sha256=expected)
                   for name, expected in frozen["processed"].items()}, digests=frozen["digests"],
        asset_audit=audit), sort_keys=True), flush=True)
    print("PREFLIGHT_OK: no preprocessing; frozen snapshot verified", flush=True)
    if args.preflight:
        return 0
    directory = root / "reports/studies" / plan["experiment"]
    manifest = directory / "manifest.json"
    if not manifest.exists():
        if directory.exists() or args.collect_only:
            raise ValueError("Batch directory exists without manifest, or collect requested before training")
        # Reject orphan outputs rather than silently starting another batch.
        if (root / "reports/trajectories" / plan["experiment"]).exists():
            raise ValueError("Orphan trajectory outputs exist")
        directory.mkdir(parents=True)
        write_json(manifest, frozen)
    elif suite.read_json(manifest) != frozen:
        raise ValueError("Batch identity/config/data/environment changed")
    retry = getattr(args, "retry_job", None)
    if retry and (retry not in {job["key"] for job in jobs} or args.collect_only):
        raise ValueError("Invalid explicit retry job")
    for job in jobs:
        checked_job(root, directory, plan, job)
    for job in jobs:
        if checked_job(root, directory, plan, job) is not None:
            if retry == job["key"]:
                raise ValueError("Cannot retry an already complete job")
            continue
        if args.collect_only:
            raise ValueError("Cannot collect an incomplete batch")
        attempts_path = directory / (job["key"] + ".attempts.json")
        attempts = suite.read_json(attempts_path) if attempts_path.exists() else []
        if attempts and retry != job["key"]:
            raise ValueError("Previous attempt incomplete/failed: {}. Inspect failure, then use --retry-job {} explicitly".format(
                job["key"], job["key"]))
        if identity(root, study, plan, versions) != {k: v for k, v in frozen.items() if k != "asset_audit"}:
            raise ValueError("Frozen files changed before training")
        attempt_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        attempt = dict(attempt_id=attempt_id, status="running")
        attempts.append(attempt)
        write_json(attempts_path, attempts)
        before = set((root / "reports/trajectories" / plan["experiment"]).rglob("result.json"))
        try:
            suite.execute(root, job["args"], directory / (job["key"] + "_" + attempt_id + ".console.log"))
            if identity(root, study, plan, versions) != {k: v for k, v in frozen.items() if k != "asset_audit"}:
                raise ValueError("Frozen files changed during training")
            new = set((root / "reports/trajectories" / plan["experiment"]).rglob("result.json")) - before
            if len(new) != 1:
                raise ValueError("Expected exactly one new trajectory result")
            trajectory = new.pop().parent
            verify_trajectory(trajectory, expected_job(plan, job))
            write_json(directory / (job["key"] + ".done.json"), dict(
                trajectory=trajectory.relative_to(root).as_posix(), result_sha256=sha256(trajectory / "result.json")))
            checked_job(root, directory, plan, job, True)
            attempt.update(status="complete", trajectory=trajectory.relative_to(root).as_posix())
        except BaseException as error:
            attempt.update(status="failed", error="{}: {}".format(type(error).__name__, error))
            write_json(attempts_path, attempts)
            raise
        write_json(attempts_path, attempts)
    # Audit the paired initialization and normalization before any selection.
    completed = [checked_job(root, directory, plan, job, True) for job in jobs]
    for seed in plan["seeds"]:
        pair = [item[1]["identity"] for job, item in zip(jobs, completed) if job["seed"] == seed]
        if pair[0]["initial_state_digest"] != pair[1]["initial_state_digest"]:
            raise ValueError("Paired initializations differ")
        if pair[0]["training_input_sha256"] != pair[1]["training_input_sha256"]:
            raise ValueError("Paired training historical inputs differ")
    if {item[1]["identity"]["scaler_digest"] for item in completed} != {audit["scaler_digest"]}:
        raise ValueError("Trajectories do not share the frozen scaler")
    for job, (trajectory, _) in zip(jobs, completed):
        if args.collect_only and not (trajectory / "selection_manifest.json").exists():
            raise ValueError("Collect-only requires previously completed selection")
        select_trajectory(trajectory, expected_job(plan, job))
    return deliver(root, directory, plan, jobs)


def main(root, args):
    import shlex
    plan, jobs = load_plan(root, args.study)
    if args.dry_run:
        for job in jobs:
            print("python -m wpf_benchmark --root . " + " ".join(shlex.quote(x) for x in job["args"]))
        print("Then audit all 6 trajectories, select 12 states, and package 180 candidates. No test scoring.")
        return 0
    versions = environment(root, suite.dependencies(require_gbdt=False))
    if args.preflight:
        return run(root, args.study, args, versions)
    import fcntl
    lock_path = root / "reports/rerun_m1/.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another training batch holds the project lock")
        run(root, args.study, args, versions)
    return 0
