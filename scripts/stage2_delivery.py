"""Check the frozen 40-run data, optionally train one C model, and collect a delivery.

No preprocessing, data repair, architecture change, or internal-output fabrication.
The delivered checkpoint and original-resolution inputs enable later no-fit inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_diagnostics.common import data_digest, load_run, equal_saved_values, STEP_NS

ORIGINAL_DATA = "033fb4a7a9e0daf8"
ORIGINAL_FILES = {
    "data/processed/sdwpf_clean.parquet": "f39562038b8cffd654c4cf09c18d15aecd5b0c50c43958f95d27893f4dcd1c82",
    "data/processed/sdwpf_meta.json": "0159403e88755daec5e060ff092304956d81b40a0166c71cf0cdaaa6b46a40a3",
}
EXPERIMENT = "stage2_internal_20261009"
ORIGINAL_C = {
    0: "lite_low_power_20261008_153714_518724",
    1: "lite_low_power_20261008_154523_133224",
    2: "lite_low_power_20261008_155315_140655",
    3: "lite_low_power_20261008_155928_137610",
    4: "lite_low_power_20261008_160613_761461",
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _sha_stream(handle)


def _sha_stream(handle):
    value = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        value.update(chunk)
    return value.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def check_snapshot(root, expected=None):
    expected = ORIGINAL_FILES if expected is None else expected
    actual = {}
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file():
            raise ValueError("Frozen file missing: {}. Restore the original server snapshot; do not run preprocess.".format(path))
        actual[relative] = sha(path)
        if actual[relative] != digest:
            raise ValueError("Frozen snapshot mismatch: {}\nExpected {}\nActual {}\nRestore the 40-run snapshot; do not overwrite it with preprocess.".format(relative, digest, actual[relative]))
    return {"data_digest": data_digest(root), "file_sha256": actual}


def check_c(model_config):
    recipe = json.loads((ROOT / "configs/agcrn_lite_low_power_mae.json").read_text(encoding="utf-8"))["model"]
    if model_config != recipe:
        raise ValueError("Run model configuration differs from the fixed C recipe")


def find_main(root, run_id):
    if Path(run_id).name != run_id:
        raise ValueError("run_id must not contain directories")
    candidates = []
    for path in (root / "reports/eval").glob("*_main_*.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("run_id") == run_id:
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError("Expected exactly one main JSON for {}".format(run_id))
    return candidates[0]


def find_completed(root, seed, snapshot):
    tag = "stage2_C_s{}".format(seed)
    for path in sorted((root / "reports/eval").glob(tag + "_main_*.json"), reverse=True):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("experiment") == EXPERIMENT and meta.get("seed") == seed and meta.get("data_digest") == snapshot["data_digest"]:
            check_c(meta["model_config"])
            return meta["run_id"]
    return None


def verified_inputs(root, run):
    import numpy as np
    import pandas as pd
    from wpf_benchmark.data.masks import target_valid
    a, meta = run["arrays"], run["metadata"]
    cfg = meta["config"]
    first_day = max(1, cfg["train_days"] + 1 - math.ceil(cfg["input_window"] / cfg["steps_per_day"]))
    last_day = int(a["times"][-1] // (86400 * 10**9) + 1)
    frame = pd.read_parquet(root / "data/processed/sdwpf_clean.parquet", filters=[("Day", ">=", first_day), ("Day", "<=", last_day)])
    frame["_ns"] = frame["ts"].astype("timedelta64[ns]").astype("int64")
    if frame.duplicated(["_ns", "TurbID"]).any():
        raise ValueError("Duplicate diagnostic input time/turbine keys")
    indexes = pd.MultiIndex.from_product([a["times"], a["turbine_ids"]], names=["_ns", "TurbID"])
    grid = frame.set_index(["_ns", "TurbID"]).reindex(indexes)
    O, N, H = a["forecasts"].shape
    columns = {"wind_speed": grid.Wspd.to_numpy().reshape(-1, N),
               "truth_m1": grid.Patv.to_numpy().reshape(-1, N),
               "truth_m2": grid.Patv_obs.to_numpy().reshape(-1, N)}
    for mask in ("m1", "m2"):
        columns["valid_" + mask] = target_valid(grid, mask, cfg["exclude_flags_main"]).reshape(-1, N)
    for name, values in columns.items():
        for h in range(H):
            if not equal_saved_values(values[h+1:h+1+O], a[name][..., h]):
                raise ValueError("Saved {} disagrees with frozen inputs at h={}".format(name, h+1))
    if not equal_saved_values(columns["truth_m1"][:O], a["last_power"]):
        raise ValueError("Issue powers disagree with the saved run")
    # Keep all validation/test history ticks, not just the 532 scored targets.
    frame = frame.drop(columns="_ns").sort_values(["ts", "TurbID"]).reset_index(drop=True)
    time_values = frame.ts.drop_duplicates().astype("timedelta64[ns]").astype("int64").to_numpy()
    if not np.all(np.diff(time_values) == STEP_NS) or len(frame) != len(time_values) * N:
        raise ValueError("Delivery inputs must retain a complete continuous turbine grid")
    return frame, {"first_day": first_day, "last_day": last_day, "rows": len(frame),
                   "purpose": "validation and test inference plus pre-validation history; use checkpoint scaler, never refit on this slice",
                   "saved_grid_comparison": "all wind, M1/M2 truths/masks, and issue powers exactly equal after saved dtype conversion"}


def collect(root, run_id, checkpoint=None, expected=None):
    snapshot = check_snapshot(root, expected)
    main_path = find_main(root, run_id)
    run = load_run(root / "reports/eval", run_id)
    meta = run["metadata"]
    if meta.get("model") != "agcrn_lite" or meta.get("target_mask") != "m1" or meta.get("eval_mask") != "m1":
        raise ValueError("Stage 2 requires the original M1 C recipe")
    check_c(meta["model_config"])
    if meta.get("data_digest") != snapshot["data_digest"]:
        raise ValueError("Run data fingerprint differs from the frozen files")
    declared = meta.get("checkpoint")
    if checkpoint is None:
        if not declared:
            raise ValueError("Run has no recorded checkpoint. Supply --checkpoint for an original server weight, or run one diagnostic model.")
        checkpoint = root / declared["path"]
    else:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError("Checkpoint missing: {}".format(checkpoint))
    checkpoint_sha = sha(checkpoint)
    relation = "user_supplied; matching saved forecasts must be checked during no-fit inference"
    if declared and checkpoint_sha == declared["sha256"]:
        relation = "main JSON checkpoint checksum verified"
    elif declared:
        raise ValueError("Checkpoint checksum differs from the main JSON")
    frame, slice_info = verified_inputs(root, run)
    artifacts = {"model_best.pt": checkpoint,
                 "run/" + main_path.name: main_path,
                 "run/" + Path(run["array_path"]).name: Path(run["array_path"]),
                 "inputs/sdwpf_meta.json": root / "data/processed/sdwpf_meta.json",
                 "recipe/agcrn_lite_low_power_mae.json": root / "configs/agcrn_lite_low_power_mae.json"}
    for path in (root / "reports/eval").glob("*_all_*.json"):
        if json.loads(path.read_text(encoding="utf-8")).get("run_id") == run_id:
            artifacts["run/" + path.name] = path
    train_log = root / "reports/train" / (run_id + ".log")
    if not train_log.is_file():
        raise ValueError("Training log missing: {}".format(train_log))
    artifacts["run/" + train_log.name] = train_log
    console = root / "reports/stage2" / (run_id + ".console.log")
    if console.is_file():
        artifacts["run/" + console.name] = console
    for path in (root / "src").rglob("*.py"):
        if "__pycache__" not in path.parts:
            artifacts["code/" + path.relative_to(root).as_posix()] = path
    for name in ("pyproject.toml",):
        if (root / name).is_file():
            artifacts["code/" + name] = root / name
    delivery_dir = root / "reports/stage2"
    delivery_dir.mkdir(parents=True, exist_ok=True)
    archive = delivery_dir / (run_id + "_delivery.zip")
    if archive.exists():
        raise ValueError("Delivery already exists; preserve it: {}".format(archive))
    with tempfile.TemporaryDirectory(prefix="stage2_", dir=str(delivery_dir)) as temporary:
        temp = Path(temporary)
        input_slice = temp / "sdwpf_inference_slice.parquet"
        frame.to_parquet(input_slice, index=False)
        artifacts["inputs/sdwpf_inference_slice.parquet"] = input_slice
        provenance = {"diagnostic_id": EXPERIMENT, "run_id": run_id, "seed": meta["seed"],
                      "source_experiment": meta["experiment"], "checkpoint_relation": relation,
                      "data_snapshot": snapshot, "code_digest": meta["code_digest"],
                      "input_slice": slice_info, "internals_exported": False,
                      "internals_next_step": "load weights, reproduce final predictions, then extract logits/q/low/normal without training",
                      "artifacts": {name: {"bytes": path.stat().st_size, "sha256": sha(path)} for name, path in artifacts.items()}}
        import numpy as np
        import pandas as pd
        provenance["environment"] = {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}
        try:
            import torch
            provenance["environment"]["torch"] = str(torch.__version__)
        except ImportError:
            provenance["environment"]["torch"] = None
        dump(temp / "delivery_manifest.json", provenance)
        partial = temp / "delivery.zip"
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as z:
            for name, path in artifacts.items():
                z.write(path, name)
            z.write(temp / "delivery_manifest.json", "delivery_manifest.json")
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None:
                raise ValueError("Delivery ZIP integrity check failed")
        # Exclusive publication keeps earlier completed deliveries intact.
        with archive.open("xb") as destination, partial.open("rb") as source:
            import shutil
            shutil.copyfileobj(source, destination)
    print("DELIVERY: {} ({:.1f} MiB)".format(archive, archive.stat().st_size / 1024**2))
    return archive


def run_once(root, seed):
    snapshot = check_snapshot(root)
    previous = find_completed(root, seed, snapshot)
    if previous:
        existing = root / "reports/stage2" / (previous + "_delivery.zip")
        if existing.is_file():
            with zipfile.ZipFile(existing) as z:
                if z.testzip() is not None:
                    raise ValueError("Existing delivery ZIP is corrupt")
            print("Already complete; no training. DELIVERY: {}".format(existing))
            return existing
        print("Reusing completed diagnostic run; no training: {}".format(previous))
        return collect(root, previous)
    original = ORIGINAL_C[seed]
    original_weight = root / "reports/checkpoints" / (original + "_best.pt")
    if original_weight.is_file():
        print("Original weight found; collecting without training: {}".format(original_weight))
        return collect(root, original, checkpoint=original_weight)
    tag = "stage2_C_s{}".format(seed)
    cmd = [sys.executable, "-m", "wpf_benchmark", "run", "--model", "agcrn_lite",
           "--config", "configs/agcrn_lite_low_power_mae.json", "--seed", str(seed),
           "--repeat", "1", "--tag", tag, "--experiment", EXPERIMENT,
           "--target-mask", "m1", "--eval-mask", "m1", "--save-checkpoint", "--save-arrays", "--no-plots"]
    print("Training exactly one unchanged C model: {}".format(" ".join(cmd)), flush=True)
    directory = root / "reports/stage2"
    directory.mkdir(parents=True, exist_ok=True)
    console = directory / (tag + ".console.log")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with console.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line); log.flush()
        status = process.wait()
    if status:
        raise RuntimeError("Training failed with exit {}. Keep {} for diagnosis; no delivery published.".format(status, console))
    completed = find_completed(root, seed, snapshot)
    if not completed:
        raise RuntimeError("Training did not produce a matching main JSON")
    import shutil
    shutil.copy2(console, directory / (completed + ".console.log"))
    return collect(root, completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "collect"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, choices=range(5), default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.action == "check":
            snapshot = check_snapshot(root)
            weights = [str(p.relative_to(root)) for p in (root / "reports").rglob("*") if p.is_file() and p.suffix.lower() in (".pt", ".pth", ".ckpt")]
            print(json.dumps({"frozen_snapshot": snapshot, "weights_found": weights,
                              "recommended_seed": args.seed, "training": "zero with matching original weights; otherwise one unchanged C run"}, ensure_ascii=False, indent=2))
        elif args.action == "run":
            if args.run_id or args.checkpoint:
                raise ValueError("Use collect with --run-id/--checkpoint; run uses the fixed C recipe")
            run_once(root, args.seed)
        else:
            if not args.run_id:
                raise ValueError("collect requires --run-id")
            collect(root, args.run_id, args.checkpoint)
    except (ValueError, RuntimeError) as exc:
        print("STOP: {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
