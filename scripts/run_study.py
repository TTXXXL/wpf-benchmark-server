"""Run a frozen validation study, resume verified jobs, and package its results."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys
import zipfile

import rerun_m1 as suite


ROOT = Path(__file__).resolve().parents[1]
NAME = r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"


def relative_file(root, name):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("Invalid project-relative path")
    path = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Invalid project-relative path")
    path.resolve().relative_to(root.resolve())
    return path


def load_plan(root, study):
    if not re.fullmatch(NAME, study):
        raise ValueError("Invalid study name")
    path = root / "configs/studies" / (study + ".json")
    plan = suite.read_json(path)
    if plan.get("schema") != 1 or not re.fullmatch(NAME, plan["experiment"]):
        raise ValueError("Invalid study schema or experiment")
    if not plan["seeds"] or len(set(plan["seeds"])) != len(plan["seeds"]) or any(
            type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in plan["seeds"]):
        raise ValueError("Study seeds must be distinct uint32 integers")
    from wpf_benchmark.evaluation.config import ProtocolConfig
    from wpf_benchmark.models import get_model
    jobs = []
    for variant in plan["variants"]:
        if not re.fullmatch(NAME, variant["name"]) or variant["target_mask"] not in ("m1", "m2"):
            raise ValueError("Invalid study variant")
        config_name = variant.get("config", plan["config"])
        recipe = suite.read_json(relative_file(root, config_name))
        protocol = dict(recipe.get("protocol", {}), target_mask=variant["target_mask"], eval_mask="m1")
        for option in ("validation_mask", "m2_extra_target_weight"):
            if option in variant:
                protocol[option] = variant[option]
        effective = ProtocolConfig(**protocol)
        for seed in plan["seeds"]:
            key = "{}_s{}".format(variant["name"], seed)
            tag = "{}_{}".format(plan.get("tag_prefix", plan["experiment"]), key)
            if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
                raise ValueError("Invalid job tag")
            args = ["run", "--model", plan["model"], "--config", config_name,
                    "--target-mask", variant["target_mask"], "--eval-mask", "m1",
                    "--seed", str(seed), "--repeat", "1", "--experiment", plan["experiment"],
                    "--tag", tag, "--validation-only", "--save-checkpoint",
                    "--no-save-arrays", "--no-plots"]
            if effective.validation_mask is not None:
                args.extend(["--validation-mask", effective.validation_mask])
            if "m2_extra_target_weight" in variant:
                args.extend(["--m2-extra-target-weight", str(effective.m2_extra_target_weight)])
            jobs.append(dict(key=key, tag=tag, seed=seed, model=plan["model"],
                             config=config_name, features=list(recipe["model"].get(
                                 "input_features", get_model(plan["model"]).features)),
                             target_mask=variant["target_mask"], validation_mask=effective.early_stop_mask,
                             args=args, model_config=recipe["model"], protocol=protocol))
    if not jobs or len({job["key"] for job in jobs}) != len(jobs):
        raise ValueError("Study jobs must be nonempty and unique")
    return plan, jobs


def snapshot(root, study, plan, versions):
    data_hash = hashlib.sha256()
    processed = {}
    for name in suite.PROCESSED:
        path = relative_file(root, name)
        expected = plan["frozen_files"][name]
        if not path.is_file() or suite.sha256(path) != expected:
            raise RuntimeError("Frozen data mismatch: {}. Keep the original server snapshot; "
                               "do not rerun preprocessing.".format(name))
        processed[name] = expected
        data_hash.update(name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                data_hash.update(block)
    code_hash = hashlib.sha256()
    sources = {}
    source_root = root / "src/wpf_benchmark"
    for source in sorted(source_root.rglob("*.py")):
        relative = source.relative_to(source_root)
        sources[source.relative_to(root).as_posix()] = suite.sha256(source)
        if ("figures" in relative.parts and relative.name != "io.py") or "__pycache__" in relative.parts:
            continue
        code_hash.update(relative.as_posix().encode("utf-8"))
        code_hash.update(source.read_bytes())
    digests = dict(data_digest=data_hash.hexdigest()[:16], code_digest=code_hash.hexdigest()[:16])
    if digests != plan["digests"]:
        raise RuntimeError("Data or model source differs from the approved study")
    recipes = {plan["config"]} | {variant.get("config", plan["config"]) for variant in plan["variants"]}
    for name in sorted(recipes):
        expected = plan.get("recipe_sha256", {}).get(name)
        if expected is not None and suite.sha256(relative_file(root, name)) != expected:
            raise RuntimeError("Approved recipe differs: " + name)
    for name in ["scripts/run_study.py", "scripts/rerun_m1.py", "pyproject.toml",
                 "configs/studies/" + study + ".json"] + sorted(recipes):
        sources[name] = suite.sha256(relative_file(root, name))
    return dict(schema=1, study=study, plan=plan, sources=sources,
                processed=processed, digests=digests, environment=versions)


def verify_result(root, job, identity):
    candidates = sorted((root / "reports/eval").glob(job["tag"] + "_validation_*.json"))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError("Duplicate results for " + job["key"])
    path = candidates[0]
    result = suite.read_json(path)
    expected = dict(model=job["model"], seed=job["seed"],
                    experiment=identity["plan"]["experiment"], table="validation",
                    target_mask=job["target_mask"], validation_mask=job["validation_mask"], eval_mask=None)
    expected.update(identity["digests"])
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Result identity or frozen digests differ: " + job["key"])
    from wpf_benchmark.evaluation.config import ProtocolConfig
    protocol = dict(job["protocol"], target_mask=job["target_mask"], eval_mask="m1")
    canonical = json.loads(json.dumps(asdict(ProtocolConfig(**protocol))))
    if result.get("config") != canonical or any(
            result.get("model_config", {}).get(k) != v for k, v in job["model_config"].items()):
        raise RuntimeError("Result training configuration differs: " + job["key"])
    if identity["plan"].get("verify_features", False) and (
            result.get("features") != job["features"] or result.get("history_scale") != "normalized"):
        raise RuntimeError("Result input features differ: " + job["key"])
    run_id = result.get("run_id", "")
    if not re.fullmatch(re.escape(job["tag"]) + r"_\d{8}_\d{6}_\d{6}", run_id):
        raise RuntimeError("Invalid run ID")
    stamp = run_id[len(job["tag"]) + 1:]
    if path.name != job["tag"] + "_validation_" + stamp + ".json":
        raise RuntimeError("Result filename and run ID differ")
    if not math.isfinite(float(result["validation_metrics"]["MAE_kW"])):
        raise RuntimeError("Nonfinite validation score")
    checkpoints = suite.checkpoint_artifacts(root, result)
    if len(checkpoints) != 1:
        raise RuntimeError("Validation-best checkpoint is required")
    artifacts = [path, checkpoints[0], root / "reports/train" / (run_id + ".log")]
    if any(not item.is_file() or not item.stat().st_size for item in artifacts):
        raise RuntimeError("Incomplete outputs for " + job["key"])
    return dict(result=path.relative_to(root).as_posix(),
                hashes={item.relative_to(root).as_posix(): suite.sha256(item) for item in artifacts})


def checked_result(root, directory, job, identity):
    checked = verify_result(root, job, identity)
    marker = directory / (job["key"] + ".done.json")
    if marker.is_file() and (checked is None or suite.read_json(marker) != checked):
        raise RuntimeError("Completed outputs changed or disappeared: " + job["key"])
    return checked


def write_delivery(root, directory, jobs, identity):
    files = {}
    rows = []
    for job in jobs:
        checked = checked_result(root, directory, job, identity)
        if checked is None:
            raise RuntimeError("Study is incomplete: " + job["key"])
        files.update(checked["hashes"])
        result = suite.read_json(root / checked["result"])
        rows.append([job["key"], job["config"], ",".join(job["features"]),
                     job["target_mask"], job["validation_mask"],
                     job["protocol"].get("m2_extra_target_weight", 1.0), job["seed"],
                     result["validation_metrics"]["MAE_kW"], result["run_id"], checked["result"]])
    summary = directory / "results.csv"
    with summary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["job", "model_config_file", "input_features", "training_mask", "early_stop_mask", "m2_extra_target_weight", "seed",
                         "native_early_stop_MAE_kW", "run_id", "validation_json"])
        writer.writerows(rows)
    for path in [directory / "manifest.json", summary] + sorted(directory.glob("*.done.json")) + \
            sorted(directory.glob("*.console.log")):
        files[path.relative_to(root).as_posix()] = suite.sha256(path)
    for name in sorted({identity["plan"]["config"], "configs/studies/" + identity["study"] + ".json"} |
                       {job["config"] for job in jobs}):
        files[name] = suite.sha256(root / name)
    delivery_manifest = directory / "delivery_manifest.json"
    suite.save_json(delivery_manifest, dict(schema=1, jobs=len(jobs),
                    digests=identity["digests"], files=files,
                    scoring="Native early-stop scores only; dense M1/M2 analysis is performed locally."))
    files[delivery_manifest.relative_to(root).as_posix()] = suite.sha256(delivery_manifest)
    archive = directory / (identity["plan"]["experiment"] + "_delivery.zip")
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name, expected in sorted(files.items()):
            if suite.sha256(root / name) != expected:
                raise RuntimeError("Output changed during packaging: " + name)
            package.write(root / name, name)
    with zipfile.ZipFile(temporary) as package:
        if package.testzip() is not None:
            raise RuntimeError("Delivery ZIP failed CRC verification")
    temporary.replace(archive)
    print("DELIVERY: " + str(archive), flush=True)
    return archive


def run_study(root, study, versions, collect_only=False):
    plan, jobs = load_plan(root, study)
    identity = snapshot(root, study, plan, versions)
    directory = root / "reports/studies" / plan["experiment"]
    manifest = directory / "manifest.json"
    if manifest.is_file():
        if suite.read_json(manifest) != identity:
            raise RuntimeError("Code/config/data/environment changed; do not mix this batch")
    else:
        if collect_only:
            raise RuntimeError("Study manifest is missing")
        for path in (root / "reports/eval").glob("*_validation_*.json"):
            if suite.read_json(path).get("experiment") == plan["experiment"]:
                raise RuntimeError("Results exist without this study manifest; return them for analysis first")
        directory.mkdir(parents=True, exist_ok=True)
        suite.save_json(manifest, identity)
    # Check every completed artifact before launching another job.
    for job in jobs:
        checked_result(root, directory, job, identity)
    for index, job in enumerate(jobs, 1):
        if snapshot(root, study, plan, versions) != identity:
            raise RuntimeError("Frozen study changed during execution")
        checked = checked_result(root, directory, job, identity)
        if checked is None and not collect_only:
            print("[run {}/{}] {}".format(index, len(jobs), job["key"]), flush=True)
            suite.execute(root, job["args"], directory / (job["key"] + ".console.log"))
            if snapshot(root, study, plan, versions) != identity:
                raise RuntimeError("Frozen study changed during training")
            checked = verify_result(root, job, identity)
        elif checked is not None:
            print("[skip {}/{}] {}".format(index, len(jobs), job["key"]), flush=True)
        if checked is None:
            raise RuntimeError("Incomplete job: " + job["key"])
        suite.save_json(directory / (job["key"] + ".done.json"), checked)
    return write_delivery(root, directory, jobs, identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="m2_robustness", help="Plan in configs/studies/<name>.json")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing or training")
    parser.add_argument("--preflight", action="store_true", help="Verify data/code/recipes/environment without training or writing")
    parser.add_argument("--collect-only", action="store_true", help="Verify and repackage; never train")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    plan, jobs = load_plan(ROOT, args.study)
    print("Validation study: {} jobs; experiment={}".format(len(jobs), plan["experiment"]), flush=True)
    if args.dry_run:
        for job in jobs:
            print("python -m wpf_benchmark --root . " + " ".join(shlex.quote(x) for x in job["args"]))
        return 0
    sys.path.insert(0, str(ROOT / "src"))
    versions = suite.dependencies(require_gbdt=False)
    if args.preflight:
        identity = snapshot(ROOT, args.study, plan, versions)
        print("PREFLIGHT PASS: data={} code={}".format(
            identity["digests"]["data_digest"], identity["digests"]["code_digest"]), flush=True)
        return 0
    import fcntl  # Linux server; share the existing batch lock.
    lock_path = ROOT / "reports/rerun_m1/.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another training batch holds the project lock")
        run_study(ROOT, args.study, versions, args.collect_only)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, KeyError, TypeError, ImportError, OSError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
