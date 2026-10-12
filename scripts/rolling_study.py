"""Schema-2 forward studies: isolated preprocessing, best weights, bounded exports."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
import shlex
import zipfile

import numpy as np

import rerun_m1 as suite
from wpf_benchmark.evaluation.config import ProtocolConfig
from wpf_benchmark.evaluation.forward import atomic_json, export_inputs, export_predictions
from wpf_benchmark.figures.io import result_provenance
from wpf_benchmark.paths import ProjectPaths

RAW_NAMES = ("sdwpf_245days_v1.csv", "sdwpf_baidukddcup2022_turb_location.csv")
PHASES = ("fit", "calibration", "outer")
FOLD_STARTS = ((122, 137, 144, 150), (138, 153, 160, 166), (154, 169, 176, 182))


def load_plan(root, study):
    from run_study import relative_file, NAME
    if not re.fullmatch(NAME, study):
        raise ValueError("Invalid study name")
    plan = suite.read_json(relative_file(root, "configs/studies/" + study + ".json"))
    if plan.get("schema") != 2 or plan.get("kind") != "d1s_forward" or not re.fullmatch(NAME, plan["experiment"]):
        raise ValueError("Invalid forward study")
    if plan["seeds"] != list(range(5)) or plan["model"] != "agcrn_lite":
        raise ValueError("D1s study requires the fixed five C members")
    recipe = suite.read_json(relative_file(root, plan["config"]))
    if suite.sha256(root / plan["config"]) != plan["recipe_sha256"]:
        raise RuntimeError("Approved recipe changed")
    policy = plan["policy"]
    if (policy["q"] != .25 or policy["rule"] != "R1" or policy["strength"] != 1 or
            policy["support"] != "E1_common_history" or policy["primary"] != "member_mean_error"):
        raise ValueError("Frozen D1s policy changed")
    fixed = dict(variants=["D1s", "D0"], shadow="fixed_equal_mean_prediction",
        outer_threshold_update=False, calibration_recovery_limit_pct=.5,
        recovery_limit_pct=1., low_anchor_h1_limit_pct=1., overall_limit_pct=.5,
        pooled_m1_gain_min_pct=2., positive_folds_min=2, positive_members_min=4,
        unique_event_gain_min_exclusive_pct=0., increment_min_pp=2.,
        bootstrap_replicates=2000, bootstrap_seed=20261012,
        technical_invalid_is_success=False, calibration_rejected_is_active_success=False)
    if any(policy.get(k) != v for k, v in fixed.items()) or plan["export_batch_size"] != 32:
        raise ValueError("Approved protections, comparisons or export batch changed")
    previous = 0
    if len(plan["folds"]) != 3 or len({fold["id"] for fold in plan["folds"]}) != 3:
        raise ValueError("Exactly three distinct folds are required")
    for index, fold in enumerate(plan["folds"]):
        if not re.fullmatch(NAME, fold["id"]) or fold["train"][0] != 1:
            raise ValueError("Invalid expanding training fold")
        ordered = [fold[key] for key in ("train", "early_stop", *PHASES)]
        if any(len(b) != 2 or any(type(x) is not int for x in b) or b[0] > b[1] for b in ordered):
            raise ValueError("Invalid day bounds")
        if any(b[0] != a[1] + 1 for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Fold phases must be ordered and disjoint")
        if [b[1] - b[0] + 1 for b in ordered[1:]] != [14, 7, 6, 12] or fold["outer"][0] <= previous:
            raise ValueError("Phase lengths or outer nonoverlap changed")
        if (fold["id"] != "F" + str(index + 1) or
                tuple([fold["train"][1]] + [fold[p][0] for p in PHASES]) != FOLD_STARTS[index]):
            raise ValueError("Approved historical day matrix changed")
        previous = fold["outer"][1]
    return plan, recipe


def identity(root, study, plan, versions):
    sources = {p.relative_to(root).as_posix(): suite.sha256(p)
               for p in sorted((root / "src/wpf_benchmark").rglob("*.py"))}
    for name in ("scripts/run_study.py", "scripts/rolling_study.py", "scripts/rerun_m1.py",
                 "scripts/artifact_checks.py", "pyproject.toml", plan["config"],
                 "configs/studies/" + study + ".json"):
        sources[name] = suite.sha256(root / name)
    raw = {}
    for name in RAW_NAMES:
        path = root / "data/raw/sdwpf" / name
        actual = suite.sha256(path)
        if actual != plan["raw_sha256"][name]:
            raise RuntimeError("Approved raw data changed: " + name)
        raw[name] = actual
    return dict(schema=2, study=study, plan=plan, sources=sources, raw=raw, environment=versions)


def check_hashes(root, hashes):
    from run_study import relative_file
    for name, expected in hashes.items():
        path = relative_file(root, name)
        if not path.is_file() or suite.sha256(path) != expected:
            raise RuntimeError("Completed artifact changed or disappeared: " + name)


def hashes_for(root, files):
    return {p.relative_to(root).as_posix(): suite.sha256(p) for p in files}


def prepare_fold(root, directory, fold, recipe, collect_only=False):
    workspace = directory / fold["id"]
    marker = workspace / "prepared.json"
    cfg = ProtocolConfig(**dict(recipe.get("protocol", {}),
        train_days=fold["train"][1], val_days=14, target_mask="m1", eval_mask="m1"))
    effective = dict(recipe, protocol=asdict(cfg))
    if marker.is_file():
        saved = suite.read_json(marker)
        check_hashes(root, saved["files"])
        if saved["fold"] != fold or saved["recipe"] != json.loads(json.dumps(effective)):
            raise RuntimeError("Prepared fold identity changed")
        return workspace, cfg
    if collect_only:
        raise RuntimeError("Missing prepared fold: " + fold["id"])
    workspace.mkdir(parents=True, exist_ok=True)
    atomic_json(workspace / "recipe.json", effective)
    from wpf_benchmark.preprocessing.sdwpf import main as preprocess
    preprocess(ProjectPaths(workspace), train_days=cfg.train_days,
               end_day=fold["outer"][1], raw_paths=ProjectPaths(root))
    files = [workspace / "recipe.json"] + list((workspace / "data/processed").glob("*"))
    for phase in PHASES:
        output = workspace / "exports" / (phase + "_inputs.npz")
        export_inputs(ProjectPaths(workspace), cfg, fold[phase], output)
        files.append(output)
    saved = dict(fold=fold, recipe=effective, files=hashes_for(root, files))
    atomic_json(marker, saved)
    return workspace, cfg


def job_description(workspace, fold, recipe, plan, seed):
    tag = plan["experiment"] + "_" + fold["id"] + "_s" + str(seed)
    return dict(seed=seed, tag=tag, args=["--root", str(workspace), "run", "--model", "agcrn_lite",
        "--config", str(workspace / "recipe.json"), "--tag", tag, "--seed", str(seed),
        "--experiment", plan["experiment"], "--validation-only", "--save-checkpoint",
        "--no-save-arrays", "--no-plots"])


def verified_training(workspace, cfg, recipe, plan, job):
    candidates = list((workspace / "reports/eval").glob(job["tag"] + "_validation_*.json"))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError("Duplicate training result: " + job["tag"])
    result = suite.read_json(candidates[0])
    data_digest, code_digest = result_provenance(ProjectPaths(workspace))
    expected = dict(model="agcrn_lite", seed=job["seed"], table="validation", eval_mask=None,
        target_mask="m1", validation_mask="m1", experiment=plan["experiment"],
        data_digest=data_digest, code_digest=code_digest)
    if any(result.get(k) != v for k, v in expected.items()) or result["config"] != json.loads(json.dumps(asdict(cfg))):
        raise RuntimeError("Training identity differs: " + job["tag"])
    if any(result["model_config"].get(k) != v for k, v in recipe["model"].items()) or result.get("features") != ["Wspd", "Wsin", "Wcos", "Patv"]:
        raise RuntimeError("Training recipe or features differ")
    if not re.fullmatch(re.escape(job["tag"]) + r"_\d{8}_\d{6}_\d{6}", result["run_id"]):
        raise RuntimeError("Invalid training run identity")
    if candidates[0].name != job["tag"] + "_validation" + result["run_id"][len(job["tag"]):] + ".json":
        raise RuntimeError("Result filename differs from run identity")
    if not np.isfinite(result["validation_metrics"]["MAE_kW"]):
        raise RuntimeError("Invalid early-stop result")
    checkpoints = suite.checkpoint_artifacts(workspace, result)
    if len(checkpoints) != 1:
        raise RuntimeError("Best checkpoint required")
    log = workspace / "reports/train" / (result["run_id"] + ".log")
    if not log.is_file():
        raise RuntimeError("Training log missing")
    return result, [candidates[0], checkpoints[0], log]


def run(root, study, versions, collect_only=False):
    plan, recipe = load_plan(root, study)
    expected = identity(root, study, plan, versions)
    directory = root / "reports/studies" / plan["experiment"]
    manifest = directory / "manifest.json"
    if manifest.is_file():
        if suite.read_json(manifest) != expected:
            raise RuntimeError("Code/config/data/environment changed; refusing mixed study")
        for marker in directory.rglob("*.done.json"):
            check_hashes(root, suite.read_json(marker)["files"])
        for marker in directory.glob("*/prepared.json"):
            check_hashes(root, suite.read_json(marker)["files"])
    else:
        if collect_only or directory.exists():
            raise RuntimeError("Study manifest missing; refusing orphan outputs")
        atomic_json(manifest, expected)
    folded = []
    total = len(plan["folds"]) * len(plan["seeds"])
    for fold in plan["folds"]:
        print("PREPARE " + fold["id"], flush=True)
        workspace, cfg = prepare_fold(root, directory, fold, recipe, collect_only)
        folded.append((fold, workspace, cfg))
    # Audit every existing result before starting any new training.
    for fold, workspace, cfg in folded:
        for seed in plan["seeds"]:
            verified_training(workspace, cfg, recipe, plan,
                              job_description(workspace, fold, recipe, plan, seed))
    index = 0
    for fold, workspace, cfg in folded:
        for seed in plan["seeds"]:
            index += 1
            if identity(root, study, plan, versions) != expected:
                raise RuntimeError("Frozen study changed during execution")
            job = job_description(workspace, fold, recipe, plan, seed)
            checked = verified_training(workspace, cfg, recipe, plan, job)
            if checked is None and not collect_only:
                print("[run {}/{}] {}".format(index, total, job["tag"]), flush=True)
                suite.execute(root, job["args"], directory / (job["tag"] + ".console.log"))
                checked = verified_training(workspace, cfg, recipe, plan, job)
            if checked is None:
                raise RuntimeError("Incomplete training: " + job["tag"])
            result, files = checked
            marker = directory / (job["tag"] + ".done.json")
            saved = dict(files=hashes_for(root, files))
            if marker.is_file() and suite.read_json(marker) != saved:
                raise RuntimeError("Completed training artifacts changed")
            atomic_json(marker, saved)
    from wpf_benchmark.models.checkpoint import load_checkpoint
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for fold, workspace, cfg in folded:
        for seed in plan["seeds"]:
            job = job_description(workspace, fold, recipe, plan, seed)
            result, _ = verified_training(workspace, cfg, recipe, plan, job)
            checkpoint = workspace / result["checkpoint"]["path"]
            model = None
            for phase in PHASES:
                output = workspace / "exports" / (phase + "_s" + str(seed) + ".npz")
                marker = output.with_suffix(".done.json")
                if marker.is_file():
                    check_hashes(root, suite.read_json(marker)["files"])
                    continue
                if collect_only:
                    raise RuntimeError("Missing bounded prediction: " + str(output))
                if model is None:
                    model = load_checkpoint(checkpoint, device=device)
                    if model.checkpoint_metadata.get("run_id") != result["run_id"]:
                        raise RuntimeError("Checkpoint metadata differs from training result")
                print("EXPORT {} s{} {}".format(fold["id"], seed, phase), flush=True)
                export_predictions(model, ProjectPaths(workspace), fold[phase], output,
                    result["checkpoint"]["sha256"], suite.sha256(workspace / "exports" / (phase + "_inputs.npz")),
                    batch_size=plan["export_batch_size"])
                atomic_json(marker, dict(files=hashes_for(root, [output, output.with_suffix(".json")])))
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    if identity(root, study, plan, versions) != expected:
        raise RuntimeError("Frozen study changed before delivery")
    files = [manifest, root / "configs/studies" / (study + ".json"), root / plan["config"]]
    for _, workspace, _ in folded:
        files += [workspace / "recipe.json", workspace / "data/processed/sdwpf_meta.json",
                  workspace / "data/processed/sdwpf_preprocessing_fit.npz"]
        files += list((workspace / "exports").glob("*"))
        files += list((workspace / "reports/eval").glob("*_validation_*.json"))
        files += list((workspace / "reports/checkpoints").glob("*_best.pt"))
        files += list((workspace / "reports/train").glob("*.log"))
    files += list(directory.glob("*.done.json")) + list(directory.glob("*.console.log"))
    for name in expected["sources"]:
        files.append(root / name)
    records = hashes_for(root, files)
    delivery = directory / "delivery_manifest.json"
    atomic_json(delivery, dict(schema=2, jobs=total, files=records,
        scope="Best checkpoints, bounded phase inputs/predictions; no outer scores or policy selection"))
    records[delivery.relative_to(root).as_posix()] = suite.sha256(delivery)
    archive = directory / (plan["experiment"] + "_delivery.zip")
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name, expected_hash in sorted(records.items()):
            if suite.sha256(root / name) != expected_hash:
                raise RuntimeError("Artifact changed during packaging")
            package.write(root / name, name)
    with zipfile.ZipFile(temporary) as package:
        if package.testzip() is not None:
            raise RuntimeError("Delivery ZIP CRC failure")
    temporary.replace(archive)
    print("COMPLETE {}/{}\nDELIVERY: {}".format(total, total, archive), flush=True)
    return archive


def main(root, args):
    plan, recipe = load_plan(root, args.study)
    directory = root / "reports/studies" / plan["experiment"]
    print("Historical rolling study: 15 trainings; no outer scoring", flush=True)
    if args.dry_run:
        for fold in plan["folds"]:
            workspace = directory / fold["id"]
            print("PREPROCESS {} train<=Day{} end=Day{} (isolated)".format(
                fold["id"], fold["train"][1], fold["outer"][1]))
            for seed in plan["seeds"]:
                command = job_description(workspace, fold, recipe, plan, seed)["args"]
                print("python -m wpf_benchmark " + " ".join(shlex.quote(x) for x in command))
        return 0
    versions = suite.dependencies(require_gbdt=False)
    if args.preflight:
        identity(root, args.study, plan, versions)
        print("PREFLIGHT PASS: raw/config/source checked; no preprocessing or training", flush=True)
        return 0
    import fcntl
    lock_path = root / "reports/rerun_m1/.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another training batch holds the project lock")
        run(root, args.study, versions, args.collect_only)
    return 0
