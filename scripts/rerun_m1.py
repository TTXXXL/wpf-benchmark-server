"""M1 experiment suite invoked by rerun_m1.sh (Linux server, Python 3.8+)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SINGLE = ("persistence", "seasonal_persistence", "climatology",
          "trend_persistence", "farm_mean_persistence", "linear")
REPEATED = ("gbdt", "lstm_seq2seq", "patchtst", "agcrn_lite", "agcrn",
            "barest", "ours")
PREPARE = (("preprocess",), ("fit-power-curve",), ("eval", "selftest"))
PROCESSED = ("data/processed/sdwpf_clean.parquet", "data/processed/sdwpf_meta.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def plan(root, experiment, repeat, first_seed):
    jobs = []
    for model in SINGLE + REPEATED:
        config = "configs/{}.json".format("ours_consist_only" if model == "ours" else model)
        settings = read_json(root / config)
        # This suite reruns the frozen default protocol, not a new tuning study.
        defaults = dict(input_window=144, horizon=12, train_days=196, val_days=25,
                        target_mask="m1", eval_mask="m1")
        if any(key not in defaults or value != defaults[key]
               for key, value in settings.get("protocol", {}).items()):
            raise ValueError("Expected default M1 protocol in " + config)
        if model in ("ours", "barest"):
            expected = 0.1 if model == "ours" else 0
            params = settings["model"]
            if params.get("lambda_wake") != 0 or params.get("lambda_consist") != expected:
                raise ValueError("Unexpected frozen physics weights in " + config)
        for seed in range(first_seed, first_seed + (1 if model in SINGLE else repeat)):
            key = "{}_s{}".format(model, seed)
            tag = "{}_{}".format(experiment, key)
            args = ["run", "--model", model, "--config", config,
                    "--target-mask", "m1", "--eval-mask", "m1", "--seed", str(seed),
                    "--repeat", "1", "--experiment", experiment, "--tag", tag,
                    "--save-arrays", "--no-plots"]
            jobs.append(dict(key=key, model=model, seed=seed, tag=tag, args=args))
    return jobs


def fingerprint(root, jobs, extra_sources=()):
    files = sorted((root / "src").rglob("*.py"))
    files += sorted({root / job["args"][4] for job in jobs})
    files += [root / "scripts/rerun_m1.py", root / "scripts/rerun_m1.sh"]
    files += [root / p for p in extra_sources]
    files += [root / "data/raw/sdwpf" / name for name in (
        "sdwpf_245days_v1.csv", "sdwpf_baidukddcup2022_turb_location.csv")]
    return {p.relative_to(root).as_posix(): sha256(p) for p in files}


def dependencies(require_gbdt=True):
    if not (3, 8) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError("Use the project's Python 3.8-3.12 environment")
    versions = {"python": sys.version, "executable": sys.executable}
    for name in ("numpy", "pandas", "pyarrow", "matplotlib", "torch"):
        module = importlib.import_module(name)
        versions[name] = module.__version__
    if require_gbdt:
        try:
            backend = importlib.import_module("lightgbm")
        except ImportError:
            backend = importlib.import_module("xgboost")
        versions[backend.__name__] = backend.__version__
    return versions


def execute(root, args, log):
    command = [sys.executable, "-u", "-m", "wpf_benchmark", "--root", str(root)] + list(args)
    print("+ " + " ".join(shlex.quote(x) for x in command), flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n+ " + " ".join(shlex.quote(x) for x in command) + "\n")
        handle.flush()
        with subprocess.Popen(command, cwd=str(root), env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            code = process.wait()
        if code:
            raise RuntimeError("Command failed ({}); see {}. Re-run the same command to resume."
                               .format(code, log))


def matches_job(result, job, experiment):
    if (result.get("model"), result.get("seed"), result.get("experiment"),
        result.get("target_mask"), result.get("eval_mask")) != (
            job["model"], job["seed"], experiment, "m1", "m1"):
        return False
    actual = result.get("model_config", {})
    return all(actual.get(key) == value for key, value in job.get("model_config", {}).items())


def collect(root, job, experiment):
    candidates = sorted((root / "reports/eval").glob(job["tag"] + "_main_*.json"),
                        key=lambda p: p.stat().st_mtime_ns, reverse=True)
    if not candidates:
        raise RuntimeError("No main result for " + job["key"])
    main = candidates[0]
    result = read_json(main)
    if not matches_job(result, job, experiment):
        raise RuntimeError("Result identity, M1 masks or model config do not match: " + str(main))
    all_table = main.with_name(main.name.replace(job["tag"] + "_main_",
                                                job["tag"] + "_all_", 1))
    files = [main, all_table, root / "reports/eval" / (result["run_id"] + "_arrays.npz"),
             root / "reports/train" / (result["run_id"] + ".log")]
    if any(not p.is_file() or p.stat().st_size == 0 for p in files):
        raise RuntimeError("Incomplete outputs for " + job["key"])
    if read_json(all_table).get("run_id") != result["run_id"]:
        raise RuntimeError("Main/all run IDs differ")
    return {"main": main.relative_to(root).as_posix(),
            "artifacts": {p.relative_to(root).as_posix(): p.stat().st_size for p in files}}


def completed(root, marker, job, experiment):
    if not marker.exists():
        return False
    saved = read_json(marker)
    for name, size in saved["artifacts"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != size:
            return False
    result = read_json(root / saved["main"])
    return matches_job(result, job, experiment)


def summary(root, directory, jobs, experiment):
    with (directory / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["job", "model", "seed", "loss", "current_power_skip", "config",
                         "target_mask", "eval_mask", "MAE_kW", "RMSE_kW",
                         "n_samples", "run_id", "main_json"])
        for job in jobs:
            marker = directory / (job["key"] + ".done.json")
            if not completed(root, marker, job, experiment):
                continue
            saved = read_json(marker)
            result = read_json(root / saved["main"])
            params = result.get("model_config", {})
            writer.writerow([job["key"], result["model"], result["seed"], params.get("loss", ""),
                             params.get("current_power_skip", ""), job["args"][4], "m1", "m1",
                             result["A_turbine"]["MAE_kW"], result["A_turbine"]["RMSE_kW"],
                             result["n_samples"], result["run_id"], saved["main"]])


def run_suite(root, directory, jobs, experiment, versions, extra_sources=()):
    manifest = {"schema": 1, "experiment": experiment, "jobs": jobs,
                "sources": fingerprint(root, jobs, extra_sources), "environment": versions}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise RuntimeError("Code/config/data/environment/plan changed. Use a new --experiment name.")
    save_json(manifest_path, manifest)
    prepared = directory / "prepared.json"
    if prepared.exists():
        saved = read_json(prepared)
        if any(not (root / p).is_file() or sha256(root / p) != saved[p] for p in PROCESSED):
            raise RuntimeError("Processed data changed. Use a new --experiment name.")
        print("[skip] preparation already verified", flush=True)
    else:
        for args in PREPARE:
            execute(root, args, directory / "prepare.log")
        save_json(prepared, {p: sha256(root / p) for p in PROCESSED})
    for i, job in enumerate(jobs, 1):
        marker = directory / (job["key"] + ".done.json")
        if completed(root, marker, job, experiment):
            print("[skip {}/{}] {}".format(i, len(jobs), job["key"]), flush=True)
            continue
        print("[run {}/{}] {}".format(i, len(jobs), job["key"]), flush=True)
        execute(root, job["args"], directory / (job["key"] + ".console.log"))
        save_json(marker, collect(root, job, experiment))
        summary(root, directory, jobs, experiment)
    summary(root, directory, jobs, experiment)
    print("Completed {} M1 runs. Summary: {}".format(len(jobs), directory / "results.csv"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="m1_rerun", help="New name starts a fresh suite")
    parser.add_argument("--repeat", type=int, default=5, help="Seeds per GBDT/deep/method model")
    parser.add_argument("--seed", type=int, default=0, help="First seed")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing or training")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.experiment):
        parser.error("experiment must be 1-64 letters/digits/underscore/hyphen")
    if args.repeat < 1 or args.seed < 0 or args.seed + args.repeat > 2 ** 32:
        parser.error("repeat must be positive; seeds must fit uint32")
    jobs = plan(ROOT, args.experiment, args.repeat, args.seed)
    print("M1 training + validation + main scoring; {} runs; experiment={}".format(
        len(jobs), args.experiment), flush=True)
    if args.dry_run:
        for command in list(PREPARE) + [j["args"] for j in jobs]:
            print("python -m wpf_benchmark " + " ".join(shlex.quote(x) for x in command))
        return 0
    versions = dependencies()  # Fail before expensive preparation if dependencies are missing.
    import fcntl  # Linux/Bash entry point; kernel releases lock on interruption.
    directory = ROOT / "reports/rerun_m1" / args.experiment
    directory.mkdir(parents=True, exist_ok=True)
    with (directory.parent / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another M1 suite is already running in this project")
        run_suite(ROOT, directory, jobs, args.experiment, versions)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, ImportError, OSError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
