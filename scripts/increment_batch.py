"""Frozen 13-job validation-only increment study with resumable delivery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage3_batch import batch_lock, stamp, atomic_dump, inside
from wpf_benchmark.evaluation.config import ProtocolConfig
from wpf_benchmark.figures.io import result_provenance
from wpf_benchmark.paths import ProjectPaths

BATCH = "persistence_increment_v1_20261009"
EXPECTED_DATA = "033fb4a7a9e0daf8"
RECIPES = {label: "configs/increment_{}.json".format(label) for label in ("L0", "L1", "D0", "D1")}
ALIGN_KEYS = ("truth", "valid", "persistence", "last_power", "history_power6",
              "history_wind6", "times", "turbine_ids")
VALIDATION_GRID = (3588, 134, 12)
SUPPORT_FILES = ("scripts/increment_batch.py", "scripts/stage3_batch.py",
                 "scripts/stage2_delivery.py")


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: {}".format(path))
    return value


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value))


def make_plan(root, expected_data=EXPECTED_DATA):
    import numpy as np
    import pandas as pd
    import torch
    root = Path(root).resolve()
    for name in ("sdwpf_clean.parquet", "sdwpf_meta.json"):
        if not (root / "data/processed" / name).is_file():
            raise ValueError("Frozen processed input missing: {}".format(name))
    data, code = result_provenance(ProjectPaths(root))
    if data != expected_data:
        raise ValueError("Data digest {} differs from expected {}; preserve the frozen server inputs".format(data, expected_data))
    recipes, jobs = {}, [{"id": "P_s0", "variant": "P", "model": "persistence",
                         "seed": 0, "recipe": None}]
    for label, relative in RECIPES.items():
        config = read(root / relative)
        if normalized(asdict(ProtocolConfig(**config.get("protocol", {})))) != normalized(asdict(ProtocolConfig())):
            raise ValueError("First-round protocol must remain unchanged")
        kind = "tcn" if label.startswith("D") else "linear"
        expected = {"input_mode": "power_wind" if label.endswith("1") else "power",
                    "hidden": 32 if kind == "tcn" else 1,
                    "layers": 6 if kind == "tcn" else 1, "kernel_size": 3,
                    "dropout": 0.0, "lr": .001, "batch": 256, "epochs": 30,
                    "patience": 5, "stride": 6, "weight_decay": 0.0}
        if config.get("model") != expected or config.get("seed") != 0:
            raise ValueError("Recipe differs from frozen specification: {}".format(label))
        recipes[label] = config
        for seed in range(3):
            jobs.append({"id": "{}_s{}".format(label, seed), "variant": label,
                         "model": "increment_" + kind, "seed": seed, "recipe": relative})
    for job in jobs:
        job["tag"] = BATCH + "_" + job["id"]
    return {"schema": 1, "batch": BATCH, "data_digest": data, "code_digest": code,
            "batch_script_sha256": sha(__file__),
            "support_sha256": {name: sha(ROOT / name) for name in SUPPORT_FILES},
            "protocol": normalized(asdict(ProtocolConfig())), "recipes": recipes, "jobs": jobs,
            "test_evaluated": False,
            "environment": {"python": sys.version, "numpy": np.__version__,
                            "pandas": pd.__version__, "torch": str(torch.__version__),
                            "cuda": torch.version.cuda,
                            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
            "screening": {"overall_skill_min": .01, "winning_seeds_min": 2,
                          "h1_relative_worsening_max": .01, "history_group_worsening_kW_max": 1.0},
            "history_groups": {"H_calm": "all six Wspd finite and in [0,1)",
                               "last_power_lt10kW": "anchor in [0,10)",
                               "low_volatility": "all six Patv finite; population std <=5 kW"},
            "input_boundary": "existing cleaned inputs; hidden wind/spatial preprocessing retained"}


def command(root, job):
    argv = [sys.executable, "-m", "wpf_benchmark", "--root", str(root), "run",
            "--model", job["model"], "--seed", str(job["seed"]), "--tag", job["tag"],
            "--experiment", BATCH, "--validation-only", "--save-validation-arrays",
            "--save-checkpoint", "--no-plots"]
    if job["recipe"]:
        argv += ["--config", job["recipe"]]
    return argv


def verify_record(root, path, job, plan):
    import numpy as np
    meta = read(path)
    run_id = meta.get("run_id", "")
    if not isinstance(run_id, str) or not re.fullmatch(re.escape(job["tag"]) + r"_\d{8}_\d{6}_\d{6}", run_id):
        raise ValueError("Unsafe/mismatched run ID")
    recipe = {} if job["variant"] == "P" else plan["recipes"][job["variant"]]["model"]
    expected = {"model": job["model"], "seed": job["seed"], "experiment": BATCH,
                "data_digest": plan["data_digest"], "code_digest": plan["code_digest"],
                "model_config": recipe, "table": "validation", "target_mask": "m1",
                "validation_mask": "m1", "eval_mask": None, "config": plan["protocol"]}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError("Result identity mismatch in {}: {}".format(key, path))
    dense = meta.get("dense_validation") or {}
    if (dense.get("split"), dense.get("target_mask"), dense.get("anchor_policy")) != (
            "validation", "m1", "latest_finite_history_fallback_zero_physical_clip"):
        raise ValueError("Validation split, mask or anchor policy differs")
    arrays_path = inside(root, dense.get("path", ""))
    if arrays_path.name != run_id + "_validation_arrays.npz" or not arrays_path.is_file() or sha(arrays_path) != dense.get("sha256"):
        raise ValueError("Validation arrays missing/corrupt")
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(ALIGN_KEYS + ("forecasts",)) - set(archive.files):
            raise ValueError("Required validation arrays missing")
        shape = archive["forecasts"].shape
        if shape != VALIDATION_GRID or not np.isfinite(archive["forecasts"]).all():
            raise ValueError("Invalid validation forecast shape/values")
        if dense.get("grid") != dict(zip(("issues", "turbines", "horizon"), shape)):
            raise ValueError("Validation grid metadata differs")
        for key in ("truth", "valid", "persistence"):
            if archive[key].shape != shape:
                raise ValueError("Validation target grid mismatch")
        for key in ("last_power", "history_power6", "history_wind6", "turbine_ids"):
            expected_shape = {"last_power": shape[:2], "history_power6": shape[:2] + (6,),
                              "history_wind6": shape[:2] + (6,), "turbine_ids": (shape[1],)}[key]
            if archive[key].shape != expected_shape:
                raise ValueError("Diagnostic grid mismatch: {}".format(key))
        if not np.array_equal(archive["turbine_ids"], np.arange(1, shape[1] + 1)):
            raise ValueError("Turbine order differs from frozen SDWPF")
        if archive["valid"].dtype.kind != "b" or not np.isfinite(archive["truth"][archive["valid"]]).all():
            raise ValueError("Invalid scoring mask")
        times = archive["times"]
        if times.dtype.kind not in "iu" or times.shape != shape[:1] or not np.all(np.diff(times) == 600 * 10 ** 9):
            raise ValueError("Invalid issue time grid")
        first_issue = plan["protocol"]["train_days"] * 86400 * 10 ** 9
        if times[0] != first_issue:
            raise ValueError("Issue grid does not start at the validation boundary")
        boundary = (plan["protocol"]["train_days"] + plan["protocol"]["val_days"]) * 86400 * 10 ** 9
        if np.any(times + shape[2] * 600 * 10 ** 9 >= boundary):
            raise ValueError("Issue times leave validation period")
        anchor = archive["last_power"]
        if not np.isfinite(anchor).all() or np.any((anchor < 0) | (anchor > plan["protocol"]["rated_power_kw"])):
            raise ValueError("Invalid persistence anchor")
        if not np.array_equal(archive["persistence"], np.repeat(anchor[..., None], shape[2], axis=-1)):
            raise ValueError("Persistence reference differs from saved anchor")
        if job["variant"] == "P" and not np.array_equal(archive["forecasts"], archive["persistence"]):
            raise ValueError("Persistence reference differs from common anchor")
        from wpf_benchmark.evaluation.validation_predictions import grouped_metrics
        if dense.get("groups") != grouped_metrics(archive):
            raise ValueError("Saved dense metrics differ from validation arrays")
    artifacts = [path, arrays_path, root / "reports/train" / (run_id + ".log")]
    if job["variant"] != "P":
        checkpoint = meta.get("checkpoint") or {}
        weights = inside(root, checkpoint.get("path", ""))
        if weights.name != run_id + "_best.pt" or not weights.is_file() or sha(weights) != checkpoint.get("sha256"):
            raise ValueError("Validation-best checkpoint missing/corrupt")
        artifacts.append(weights)
    hashes = {}
    for artifact in artifacts:
        if not artifact.is_file():
            raise ValueError("Missing artifact: {}".format(artifact))
        hashes[artifact.relative_to(root).as_posix()] = sha(artifact)
    return {"result": path.relative_to(root).as_posix(), "run_id": run_id,
            "hashes": hashes, "dense_validation": dense}


def find_record(root, job, plan):
    records = []
    for path in sorted((root / "reports/eval").glob(job["tag"] + "_validation_*.json")):
        meta = read(path)
        if (meta.get("experiment"), meta.get("seed"), meta.get("data_digest"), meta.get("code_digest")) == (BATCH, job["seed"], plan["data_digest"], plan["code_digest"]):
            records.append(verify_record(root, path, job, plan))
    if len(records) > 1:
        raise ValueError("Duplicate completed results; resolve instead of selecting the best")
    return records[0] if records else None


def check_saved_hashes(root, entry):
    for relative, expected in entry.get("hashes", {}).items():
        path = inside(root, relative)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Previously completed artifact changed: {}".format(relative))


def collect(root, plan, state, directory):
    import numpy as np
    if make_plan(root, plan["data_digest"]) != plan:
        raise ValueError("Code, recipes or data changed during batch")
    artifacts, reference, complete = {}, None, 0
    for job in plan["jobs"]:
        entry = state["jobs"].get(job["id"], {})
        if entry.get("status") != "complete":
            continue
        check_saved_hashes(root, entry)
        record = verify_record(root, inside(root, entry["result"]), job, plan)
        arrays_path = inside(root, record["dense_validation"]["path"])
        with np.load(arrays_path, allow_pickle=False) as package:
            if reference is None:
                reference = {key: package[key] for key in ALIGN_KEYS}
            else:
                for key in ALIGN_KEYS:
                    a, b = reference[key], package[key]
                    same = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)
                    if not same:
                        raise ValueError("Across-job alignment differs: {}".format(key))
        for relative in record["hashes"]:
            path = inside(root, relative)
            artifacts["runs/" + job["id"] + "/" + path.name] = path
        complete += 1
    for path in directory.glob("*.console.log"):
        artifacts["console/" + path.name] = path
    for relative in RECIPES.values():
        artifacts["recipes/" + Path(relative).name] = root / relative
    for path in (ROOT / "src").rglob("*.py"):
        if "__pycache__" not in path.parts:
            artifacts["code/" + path.relative_to(ROOT).as_posix()] = path
    for relative in ("scripts/increment_batch.py", "scripts/stage3_batch.py", "scripts/stage2_delivery.py",
                     "docs/持续性增量实验_运行与交付.md", "pyproject.toml"):
        artifacts["code/" + relative] = ROOT / relative
    artifacts["inputs/sdwpf_meta.json"] = root / "data/processed/sdwpf_meta.json"
    manifest = {"plan": plan, "jobs": state["jobs"], "complete_jobs": complete,
                "expected_jobs": len(plan["jobs"]), "complete": complete == len(plan["jobs"]),
                "test_evaluated": False,
                "artifacts": {name: {"sha256": sha(path), "bytes": path.stat().st_size}
                              for name, path in artifacts.items()}}
    archive = directory / (BATCH + "_" + stamp() + "_delivery.zip")
    temporary = archive.with_suffix(".partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as package:
        for name, path in artifacts.items():
            package.write(path, name)
        package.writestr("batch_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(temporary) as package:
        if package.testzip() is not None:
            raise ValueError("Delivery ZIP failed CRC verification")
    os.replace(str(temporary), str(archive))
    print("{} {}/{} jobs. DELIVERY: {}".format("COMPLETE" if manifest["complete"] else "PARTIAL",
                                             complete, len(plan["jobs"]), archive), flush=True)
    return manifest["complete"]


def run_batch(root, expected_data=EXPECTED_DATA, train=True):
    root = Path(root).resolve()
    plan = make_plan(root, expected_data)
    directory = root / "reports/increment_batch" / BATCH
    with batch_lock(directory):
        saved_plan, saved_state = directory / "plan.json", directory / "state.json"
        if saved_plan.is_file() and read(saved_plan) != plan:
            raise ValueError("Saved plan differs; preserve previous results and resolve drift")
        atomic_dump(saved_plan, plan)
        state = read(saved_state) if saved_state.is_file() else {"plan_fingerprint": fingerprint(plan), "jobs": {}}
        if state.get("plan_fingerprint") != fingerprint(plan):
            raise ValueError("Batch-state fingerprint differs")
        if set(state.get("jobs", {})) - {job["id"] for job in plan["jobs"]}:
            raise ValueError("State contains jobs outside frozen plan")
        for i, job in enumerate(plan["jobs"], 1):
            if make_plan(root, expected_data) != plan:
                raise ValueError("Code/data changed during batch")
            print("[{}/{}] {}".format(i, len(plan["jobs"]), job["id"]), flush=True)
            previous = state["jobs"].get(job["id"], {})
            if previous.get("status") == "complete":
                # Preserve the old hash record and stop on tampering; do not
                # overwrite it with a failed status and later accept new hashes.
                check_saved_hashes(root, previous)
            try:
                record = find_record(root, job, plan)
                if record is None and train:
                    console = directory / (job["id"] + "_" + stamp() + ".console.log")
                    argv = command(root, job)
                    print("TRAIN: {}".format(" ".join(argv)), flush=True)
                    state["jobs"][job["id"]] = {"status": "running"}
                    atomic_dump(saved_state, state)
                    env = dict(os.environ)
                    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                    env["PYTHONUNBUFFERED"] = "1"
                    with console.open("x", encoding="utf-8") as output:
                        output.write(json.dumps({"command": argv, "python": sys.version}) + "\n")
                        output.flush()
                        subprocess.run(argv, cwd=str(root), env=env, stdout=output,
                                       stderr=subprocess.STDOUT, check=True)
                    record = find_record(root, job, plan)
                    if record is None:
                        raise RuntimeError("No completed result after training")
                state["jobs"][job["id"]] = dict(record, status="complete") if record else {"status": "missing"}
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
                state["jobs"][job["id"]] = {"status": "failed", "error": str(exc)}
                print("FAILED: {}; preserving logs and continuing".format(exc), flush=True)
            atomic_dump(saved_state, state)
        return collect(root, plan, state, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "collect"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expected-data-digest", default=EXPECTED_DATA,
                        help="Default is the frozen server snapshot; do not override to hide drift")
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        if args.action == "plan":
            plan = make_plan(root, args.expected_data_digest)
            for job in plan["jobs"]:
                job["command"] = command(root, job)
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        return 0 if run_batch(root, args.expected_data_digest, args.action == "run") else 2
    except (OSError, ValueError, RuntimeError) as exc:
        print("STOP: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
