"""Command line entry point for the SDWPF benchmark."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .evaluation.config import ProtocolConfig
from .paths import ProjectPaths


def _load_experiment_config(path: Optional[str], paths: ProjectPaths) -> Dict[str, Any]:
    if path is None:
        return {}
    source = Path(path)
    if not source.is_absolute():
        source = paths.root / source
    content = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("Experiment config must be a JSON object")
    return content


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="wpf-benchmark")
    parser.add_argument("--root", help="Project directory containing data/ and reports/")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("eda", help="Inspect raw SDWPF data")
    commands.add_parser("calibrate-k", help="Calibrate spatial neighbor count")
    commands.add_parser("preprocess", help="Produce cleaned Parquet data")
    commands.add_parser("fit-power-curve", help="Fit training-only turbine power curves into metadata")
    wake = commands.add_parser("calibrate-wake", help="Calibrate wind direction on training days")
    wake.add_argument("--train-days", type=int, default=196)
    prior = commands.add_parser("prepare-wake-prior", help="Build a directional wake graph")
    prior.add_argument("--train-days", type=int, default=196)
    commands.add_parser("models", help="List registered forecast models")
    evaluation = commands.add_parser("eval", help="Evaluation protocol tools")
    evaluation.add_argument("action", choices=("selftest",))
    run = commands.add_parser("run", help="Train and evaluate a forecast model")
    run.add_argument("--model", required=True, help="Registered model name")
    run.add_argument("--config", help="JSON file with protocol and model settings")
    run.add_argument("--tag", default="", help="Output filename prefix")
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--seed", type=int, default=None,
                     help="First random seed (default: config seed or 0)")
    run.add_argument("--repeat", type=int, default=1,
                     help="Sequential runs with seeds seed, seed+1, ...")
    run.add_argument("--experiment", default="standard")
    run.add_argument("--target-mask", choices=("m1", "m2"),
                     help="Training and validation target definition (default: M1)")
    run.add_argument("--eval-mask", choices=("m1", "m2"),
                     help="Main test scoring definition (default: M1)")
    run.add_argument("--validation-only", action="store_true",
                     help="Train and save validation metrics without loading test data")
    run.add_argument("--save-validation-arrays", action="store_true",
                     help="Export dense validation forecasts; requires --validation-only")
    run.add_argument("--no-plots", action="store_true",
                     help="Skip automatic paper-figure refresh after this run")
    re = commands.add_parser("rescore", help="Score saved M1/M2 forecasts under another mask")
    re.add_argument("--run-id", required=True)
    re.add_argument("--eval-mask", required=True, choices=("m1", "m2"))
    re.add_argument("--experiment", default="mask_rescore")
    re.add_argument("--tag", default="")
    arrays = run.add_mutually_exclusive_group()
    arrays.add_argument("--save-arrays", dest="save_arrays", action="store_true", default=True)
    arrays.add_argument("--no-save-arrays", dest="save_arrays", action="store_false")
    checkpoints = run.add_mutually_exclusive_group()
    checkpoints.add_argument("--save-checkpoint", dest="save_checkpoint", action="store_true",
                             default=True, help="Save the validation-best neural model (default)")
    checkpoints.add_argument("--no-save-checkpoint", dest="save_checkpoint", action="store_false",
                             help="Disable neural checkpoint output")
    plots = commands.add_parser("plots", help="Generate paper figures and tables")
    plots.add_argument("--figures", help="Comma-separated figure numbers, e.g. 6,7,13")
    plots.add_argument("--tables", help="Comma-separated table numbers, e.g. 1,2")
    plots.add_argument("--all", action="store_true", help="Attempt all registered artifacts")
    plots.add_argument("--out", help="Output directory (default: reports/figs/paper)")
    plots.add_argument("--wind-direction", type=float, help="Prevailing wind direction in degrees")
    plots.add_argument("--run-id", help="Exact ours run ID for figure 12")
    args = parser.parse_args(argv)

    if args.command == "models":
        from .models import MODEL_REGISTRY
        for name in sorted(MODEL_REGISTRY):
            print(name)
        return 0

    paths = ProjectPaths.resolve(args.root)
    if args.command == "eda":
        from .analysis.eda import main as run_eda
        run_eda(paths)
    elif args.command == "calibrate-k":
        from .analysis.k_selection import main as run_calibration
        run_calibration(paths)
    elif args.command == "preprocess":
        from .preprocessing.sdwpf import main as run_preprocessing
        run_preprocessing(paths)
    elif args.command == "fit-power-curve":
        from .data.io import load_clean
        from .models.power_curve import fit_power_curves
        protocol = ProtocolConfig()
        columns = ["TurbID", "Wspd", "Patv"] + list(protocol.exclude_flags_main)
        train = load_clean(paths, columns=columns,
                           filters=[("Day", "<=", protocol.train_days)])
        curves = fit_power_curves(train, protocol,
                                  paths.processed / "sdwpf_meta.json")
        print("Fitted {} turbine curves into sdwpf_meta.json".format(len(curves)))
    elif args.command == "calibrate-wake":
        from .analysis.wake_calibration import calibrate_wake
        result = calibrate_wake(paths, ProtocolConfig(train_days=args.train_days))
        print("Wake calibration: {} -> {}".format(
            result.get("selected_convention"), result["decision"]))
    elif args.command == "prepare-wake-prior":
        from .models.wake import prepare_wake_prior
        source = prepare_wake_prior(paths, train_days=args.train_days)
        print("Wake prior: {}".format(source))
    elif args.command == "eval":
        from .evaluation.selftest import selftest
        selftest(paths)
    elif args.command == "run":
        from .models import get_model
        from .runner import eval_forecaster, seed_all
        settings = _load_experiment_config(args.config, paths)
        protocol = ProtocolConfig(**settings.get("protocol", {}))
        if args.target_mask or args.eval_mask:
            protocol = replace(protocol,
                               target_mask=args.target_mask or protocol.target_mask,
                               eval_mask=args.eval_mask or protocol.eval_mask)
        if args.repeat <= 0:
            raise ValueError("--repeat must be positive")
        first_seed = int(settings.get("seed", 0) if args.seed is None else args.seed)
        for seed in range(first_seed, first_seed + args.repeat):
            seed_all(seed)
            model = get_model(args.model)(**settings.get("model", {}))
            results = eval_forecaster(model, protocol, args.tag, paths, args.batch_size,
                                      settings.get("model", {}), args.save_arrays,
                                      seed, args.experiment, args.validation_only,
                                      save_checkpoint=args.save_checkpoint,
                                      save_validation_arrays=args.save_validation_arrays)
            for table, result in results.items():
                mae = (result["validation_metrics"]["MAE_kW"] if
                       table == "validation" else result["A_turbine"]["MAE_kW"])
                print("seed {} {}: MAE {:.2f} kW -> {}".format(
                    seed, table, mae,
                    result["result_path"]))
                if result.get("dense_validation"):
                    dense_mae = result["dense_validation"]["groups"]["overall"]["model"]["MAE_kW"]
                    print("seed {} dense validation: MAE {:.2f} kW".format(seed, dense_mae))
            checkpoint = next(iter(results.values())).get("checkpoint")
            if checkpoint is not None:
                print("Best checkpoint: {}".format(paths.root / checkpoint["path"]))
        if not args.no_plots:
            from .figures import generate
            generate(paths, paths.figures / "paper")
    elif args.command == "rescore":
        from .evaluation.rescore import rescore
        results = rescore(paths, args.run_id, args.eval_mask,
                          args.experiment, args.tag)
        for table, result in results.items():
            print("{} {}: MAE {:.2f} kW -> {}".format(
                result["model"], table, result["A_turbine"]["MAE_kW"],
                result["result_path"]))
    elif args.command == "plots":
        from .figures import generate
        out = Path(args.out) if args.out else paths.figures / "paper"
        if not out.is_absolute():
            out = paths.root / out
        generate(paths, out, args.figures, args.tables, args.all,
                 args.wind_direction, args.run_id)
    return 0
