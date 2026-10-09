"""Run matched state-weight experiments once, resume completed jobs, deliver one ZIP.

The unchanged C controls retain their original test arrays. New candidates train
on Days 1-196 and early-stop on Days 197-221, without evaluating the test set.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

import stage2_delivery as stage2
from wpf_benchmark.evaluation.config import ProtocolConfig

ROOT = stage2.ROOT
BATCH = "stage3_state_weight_batch_20261009"
EXPERIMENT = BATCH
RECIPES = {
    "C": "configs/agcrn_lite_low_power_mae.json",
    "state002": "configs/agcrn_lite_low_power_state002.json",
    "state010": "configs/agcrn_lite_low_power_state010.json",
}
WEIGHTS = {"C": .05, "state002": .02, "state010": .10}


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: {}".format(path))
    return value


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def atomic_dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    stage2.dump(temporary, value)
    os.replace(str(temporary), str(path))


def inside(root, relative):
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ValueError("Artifact path leaves project: {}".format(relative))
    return path


def code_digest(root):
    # Same definition as result_provenance, applied to the actual --root used by
    # the training subprocess, rather than whichever package Python imported.
    source_root = root / "src/wpf_benchmark"
    sources = sorted(source_root.rglob("*.py"))
    if not sources:
        raise ValueError("Model source files missing: {}".format(source_root))
    value = hashlib.sha256()
    for source in sources:
        relative = source.relative_to(source_root)
        if ("figures" in relative.parts and relative.name != "io.py") or "__pycache__" in relative.parts:
            continue
        value.update(relative.as_posix().encode("utf-8"))
        value.update(source.read_bytes())
    return value.hexdigest()[:16]


@contextmanager
def batch_lock(directory):
    """OS releases this lock even after a crash; no stale-lock deletion is needed."""
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / "run.lock").open("a+b")
    try:
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("This batch is already running in another process") from exc
        yield
    finally:
        handle.close()


def make_plan(root):
    snapshot = stage2.check_snapshot(root)
    code = code_digest(root)
    base = read(ROOT / RECIPES["C"])
    recipes = {}
    for variant, relative in RECIPES.items():
        recipe = read(root / relative)
        expected = normalized(base)
        expected["model"]["state_loss_weight"] = WEIGHTS[variant]
        if recipe != expected:
            raise ValueError("Only state_loss_weight may change: {}".format(relative))
        recipes[variant] = recipe
    jobs = []
    for variant in RECIPES:
        for seed in range(5):
            tag = ("stage2_C_s{}" if variant == "C" else "stage3_" + variant + "_s{}").format(seed)
            jobs.append({"id": "{}_s{}".format(variant, seed), "variant": variant,
                         "seed": seed, "tag": tag, "recipe": RECIPES[variant],
                         "table": "main" if variant == "C" else "validation"})
    return {"schema": 1, "batch_id": BATCH, "snapshot": snapshot,
            "code_digest": code, "protocol": normalized(asdict(ProtocolConfig())),
            "recipes": recipes, "jobs": jobs}


def verify_record(root, path, job, plan):
    meta = read(path)
    run_id = meta.get("run_id", "")
    if not isinstance(run_id, str) or not re.fullmatch(re.escape(job["tag"]) + r"_\d{8}_\d{6}_\d{6}", run_id):
        raise ValueError("Unsafe/missing run_id")
    suffix = run_id[len(job["tag"])+1:]
    if path.name != job["tag"] + "_" + job["table"] + "_" + suffix + ".json":
        raise ValueError("Result filename and run_id disagree")
    if (meta.get("model") != "agcrn_lite" or meta.get("seed") != job["seed"]
            or meta.get("data_digest") != plan["snapshot"]["data_digest"]
            or meta.get("code_digest") != plan["code_digest"]
            or meta.get("model_config") != plan["recipes"][job["variant"]]["model"]
            or normalized(meta.get("config")) != plan["protocol"]
            or meta.get("target_mask") != "m1"):
        raise ValueError("Run identity/configuration mismatch: {}".format(path))
    if job["variant"] == "C":
        if meta.get("eval_mask") != "m1" or meta.get("table", "main") != "main":
            raise ValueError("Control must retain its M1 main result")
        arrays = root / "reports/eval" / (run_id + "_arrays.npz")
        if not arrays.is_file():
            raise ValueError("Control test arrays missing: {}".format(arrays))
    else:
        if (meta.get("table") != "validation" or meta.get("eval_mask") is not None
                or meta.get("validation_mask") != "m1"):
            raise ValueError("Candidate must be validation-only")
        metrics = meta.get("validation_metrics") or {}
        import math
        if not math.isfinite(float(metrics.get("MAE_kW", float("nan")))):
            raise ValueError("Candidate validation MAE is missing/nonfinite")
    checkpoint = meta.get("checkpoint")
    if (not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("path"), str)
            or not isinstance(checkpoint.get("sha256"), str)):
        raise ValueError("Result has no recorded checkpoint: {}".format(path))
    weight = inside(root, checkpoint["path"])
    if weight.name != run_id + "_best.pt":
        raise ValueError("Checkpoint filename and run_id disagree")
    if not weight.is_file() or stage2.sha(weight) != checkpoint["sha256"]:
        raise ValueError("Checkpoint missing or checksum differs: {}".format(weight))
    log = root / "reports/train" / (run_id + ".log")
    if not log.is_file():
        raise ValueError("Training log missing: {}".format(log))
    return {"run_id": run_id, "result": path.relative_to(root).as_posix(),
            "checkpoint": weight.relative_to(root).as_posix(),
            "train_log": log.relative_to(root).as_posix()}


def find_record(root, job, plan):
    # Reuse this batch, the documented earlier .10 validation run, or unchanged
    # stage-2 C controls. Never silently substitute another model/seed/mask/code.
    for path in sorted((root / "reports/eval").glob(job["tag"] + "_" + job["table"] + "_*.json"), reverse=True):
        meta = read(path)
        allowed = (stage2.EXPERIMENT,) if job["variant"] == "C" else (EXPERIMENT, "stage3_state_weight_20261009")
        if (meta.get("experiment") in allowed and meta.get("seed") == job["seed"]
                and meta.get("data_digest") == plan["snapshot"]["data_digest"]
                and meta.get("code_digest") == plan["code_digest"]):
            return verify_record(root, path, job, plan)
    return None


def command(job):
    cmd = [sys.executable, "-m", "wpf_benchmark", "run", "--model", "agcrn_lite",
           "--config", job["recipe"], "--seed", str(job["seed"]), "--repeat", "1",
           "--tag", job["tag"], "--experiment", stage2.EXPERIMENT if job["variant"] == "C" else EXPERIMENT,
           "--target-mask", "m1", "--eval-mask", "m1", "--save-checkpoint", "--no-plots"]
    cmd += ["--save-arrays"] if job["variant"] == "C" else ["--validation-only", "--no-save-arrays"]
    return cmd


def execute_training(root, job, console):
    cmd = command(job)
    print("TRAIN {}: {}".format(job["id"], " ".join(cmd)), flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    with console.open("x", encoding="utf-8") as output:
        output.write(json.dumps({"command": cmd, "python": sys.version}) + "\n")
        output.flush()
        process = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line); output.flush()
            status = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
        finally:
            process.stdout.close()
    if status:
        raise RuntimeError("Training exited with {}. See {}".format(status, console))


def collect(root, plan, state, directory):
    # The archive must describe exactly the code/data on which these models ran.
    if make_plan(root) != plan:
        raise ValueError("Code/configuration/data changed during the batch")
    artifacts = {}
    controls = []
    for job in plan["jobs"]:
        status = state["jobs"].get(job["id"], {})
        if status.get("status") != "complete":
            continue
        record = verify_record(root, inside(root, status["result"]), job, plan)
        for key in ("result", "checkpoint", "train_log"):
            path = root / record[key]
            artifacts["runs/" + job["id"] + "/" + path.name] = path
        if job["variant"] == "C":
            path = root / "reports/eval" / (record["run_id"] + "_arrays.npz")
            artifacts["runs/" + job["id"] + "/" + path.name] = path
            controls.append(record)
    # All attempts, including failures, remain available for diagnosis.
    for path in sorted(directory.glob("*.console.log")):
        artifacts["console/" + path.name] = path
    for relative in RECIPES.values():
        artifacts["recipe/" + Path(relative).name] = root / relative
    for path in (root / "src").rglob("*.py"):
        if "__pycache__" not in path.parts:
            artifacts["code/" + path.relative_to(root).as_posix()] = path
    for relative in ("pyproject.toml", "scripts/stage3_batch.py", "scripts/stage2_delivery.py",
                     "scripts/review_diagnostics/common.py", "docs/阶段3_一次跑完与统一交付.md"):
        if (root / relative).is_file():
            artifacts["code/" + relative] = root / relative
    artifacts["inputs/sdwpf_meta.json"] = root / "data/processed/sdwpf_meta.json"
    with tempfile.TemporaryDirectory(prefix="collect_", dir=str(directory)) as temporary:
        temp = Path(temporary)
        slice_info = None
        if controls:
            first = stage2.load_run(root / "reports/eval", controls[0]["run_id"])
            frame, slice_info = stage2.verified_inputs(root, first)
            import numpy as np
            for record in controls[1:]:
                other = stage2.load_run(root / "reports/eval", record["run_id"])
                for name in ("times", "turbine_ids", "truth_m1", "truth_m2", "valid_m1", "valid_m2",
                             "wind_speed", "last_power", "target_mask", "eval_mask"):
                    a, b = first["arrays"][name], other["arrays"][name]
                    equal = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)
                    if not equal:
                        raise ValueError("Control grids disagree for {}: {}".format(name, record["run_id"]))
            path = temp / "sdwpf_inference_slice.parquet"
            frame.to_parquet(path, index=False)
            artifacts["inputs/sdwpf_inference_slice.parquet"] = path
        complete = sum(entry.get("status") == "complete" for entry in state["jobs"].values())
        import numpy as np
        import pandas as pd
        import torch
        manifest = {"schema": 1, "batch_id": BATCH, "plan": plan, "jobs": state["jobs"],
                    "complete_jobs": complete, "expected_jobs": len(plan["jobs"]),
                    "complete": complete == len(plan["jobs"]), "input_slice": slice_info,
                    "candidate_test_evaluated": False, "internals_exported": False,
                    "selection_split": {"train_days": [1,196], "validation_days": [197,221], "test_days": [222,245]},
                    "analysis_order": "dense validation inference for all candidates and controls; select one weight; then test only the selected weight against C; do not fit/refit scalers on delivery inputs",
                    "independent_new_period": "not included; requires additional data",
                    "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "torch": str(torch.__version__)},
                    "artifacts": {name: {"bytes": path.stat().st_size, "sha256": stage2.sha(path)} for name,path in artifacts.items()}}
        fingerprint = digest(manifest)
        previous = state.get("archive", {})
        if previous.get("fingerprint") == fingerprint:
            archive = inside(root, previous["path"])
            if not archive.is_file() or stage2.sha(archive) != previous["sha256"]:
                raise ValueError("Existing batch archive missing/corrupt")
            with zipfile.ZipFile(archive) as package:
                if package.testzip() is not None:
                    raise ValueError("Existing batch archive CRC failed")
            print("DELIVERY (reused): {}".format(archive), flush=True)
            return archive, manifest["complete"]
        stage2.dump(temp / "batch_manifest.json", manifest)
        partial = temp / "delivery.zip"
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as package:
            for name,path in artifacts.items():
                package.write(path, name)
            package.write(temp / "batch_manifest.json", "batch_manifest.json")
        with zipfile.ZipFile(partial) as package:
            if package.testzip() is not None:
                raise ValueError("Batch archive CRC failed")
        archive = directory / (BATCH + "_" + stamp() + "_delivery.zip")
        # Same filesystem rename publishes only a complete, CRC-checked ZIP.
        os.replace(str(partial), str(archive))
        state["archive"] = {"path": archive.relative_to(root).as_posix(), "sha256": stage2.sha(archive), "fingerprint": fingerprint}
        atomic_dump(directory / "state.json", state)
        print("{} {}/{} jobs. DELIVERY: {} ({:.1f} MiB)".format(
            "COMPLETE" if manifest["complete"] else "PARTIAL", complete, len(plan["jobs"]), archive, archive.stat().st_size/1024**2), flush=True)
        return archive, manifest["complete"]


def run_batch(root, train=True):
    plan = make_plan(root)  # Check frozen files and recipes BEFORE starting any job.
    directory = root / "reports/stage3_batch" / BATCH
    with batch_lock(directory):
        plan_path = directory / "plan.json"
        if plan_path.is_file() and read(plan_path) != plan:
            raise ValueError("Saved batch plan differs; preserve it and resolve the code/data drift")
        atomic_dump(plan_path, plan)
        state_path = directory / "state.json"
        state = read(state_path) if state_path.is_file() else {"plan_fingerprint": digest(plan), "jobs": {}}
        if state.get("plan_fingerprint") != digest(plan):
            raise ValueError("Batch state fingerprint differs from the plan")
        if set(state.get("jobs", {})) - {job["id"] for job in plan["jobs"]}:
            raise ValueError("Batch state contains jobs outside this plan")
        for index,job in enumerate(plan["jobs"], 1):
            if make_plan(root) != plan:
                raise ValueError("Code/configuration/data changed during the batch")
            print("[{}/{}] {}".format(index, len(plan["jobs"]), job["id"]), flush=True)
            try:
                record = find_record(root, job, plan)
                console = None
                if record is None and train:
                    console = directory / (job["id"] + "_" + stamp() + ".console.log")
                    state["jobs"][job["id"]] = {"status": "running", "console": console.relative_to(root).as_posix()}
                    atomic_dump(state_path, state)
                    execute_training(root, job, console)
                    record = find_record(root, job, plan)
                    if record is None:
                        raise RuntimeError("No matching completed result after training")
                if record is None:
                    state["jobs"][job["id"]] = {"status": "missing"}
                else:
                    print("READY: {}".format(record["run_id"]), flush=True)
                    entry = dict(record, status="complete")
                    # Preserve the original attempt reference on resumption.
                    prior_console = state["jobs"].get(job["id"], {}).get("console")
                    if console or prior_console:
                        entry["console"] = console.relative_to(root).as_posix() if console else prior_console
                    state["jobs"][job["id"]] = entry
            except (OSError, ValueError, RuntimeError) as exc:
                state["jobs"].setdefault(job["id"], {})["status"] = "failed"
                state["jobs"][job["id"]]["error"] = str(exc)
                print("FAILED {}: {}; continuing remaining jobs".format(job["id"], exc), flush=True)
            atomic_dump(state_path, state)
        return collect(root, plan, state, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "collect"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.action == "plan":
            plan = make_plan(root)
            for job in plan["jobs"]:
                record = find_record(root, job, plan)
                job["action"] = "reuse" if record else "train"
                job["run_id"] = record["run_id"] if record else None
                job["command"] = command(job)
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        _, complete = run_batch(root, train=args.action == "run")
        return 0 if complete else 2
    except (OSError, ValueError, RuntimeError) as exc:
        print("STOP: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
