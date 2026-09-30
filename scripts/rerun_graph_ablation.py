"""Paired M1 graph experiments: architecture x loss x current-power skip."""
from __future__ import annotations

import argparse
import re
import shlex
import sys

import rerun_m1 as suite


ROOT = suite.ROOT
EXTRA_SOURCES = ("scripts/rerun_graph_ablation.py", "scripts/rerun_graph_ablation.sh")


def plan(root, experiment, repeat, first_seed, loss="both", include_no_skip=False):
    jobs = []
    for objective in (("mae", "mse") if loss == "both" else (loss,)):
        if objective not in ("mae", "mse"):
            raise ValueError("loss must be mae, mse or both")
        for skip in ((True, False) if include_no_skip else (True,)):
            for model in ("agcrn", "agcrn_lite"):
                stem = "{}_{}".format(model, objective)
                config = ("configs/{}.json".format(stem) if skip else
                          "configs/graph_ablation/{}_no_skip.json".format(stem))
                settings = suite.read_json(root / config)
                expected = dict(target_mask="m1", eval_mask="m1", input_window=144,
                                horizon=12, train_days=196, val_days=25)
                if any(key not in expected or value != expected[key]
                       for key, value in settings.get("protocol", {}).items()):
                    raise ValueError("Expected default M1 protocol in " + config)
                params = settings["model"]
                common = dict(hidden=64, layers=2, emb=10, dropout=0.0, lr=0.001,
                              batch=16, epochs=30, patience=5, stride=6, weight_decay=0.0,
                              loss=objective, current_power_skip=skip)
                common.update(cheb_k=2) if model == "agcrn" else common.update(temporal_stride=6)
                if params != common:
                    raise ValueError("Unexpected frozen graph ablation parameters in " + config)
                for seed in range(first_seed, first_seed + repeat):
                    key = "{}_{}_s{}".format(stem, "skip" if skip else "no_skip", seed)
                    tag = "{}_{}".format(experiment, key)
                    args = ["run", "--model", model, "--config", config,
                            "--target-mask", "m1", "--eval-mask", "m1", "--seed", str(seed),
                            "--repeat", "1", "--experiment", experiment, "--tag", tag,
                            "--save-arrays", "--no-plots"]
                    jobs.append(dict(key=key, model=model, seed=seed, tag=tag, args=args,
                                     model_config=params))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="m1_graph_ablation", help="Use a new name for a new plan")
    parser.add_argument("--repeat", type=int, default=5, help="Paired seeds per configuration")
    parser.add_argument("--seed", type=int, default=0, help="First seed")
    parser.add_argument("--loss", choices=("both", "mae", "mse"), default="both")
    parser.add_argument("--include-no-skip", action="store_true", help="Also train each matched no-skip control")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing or training")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.experiment):
        parser.error("experiment must be 1-64 letters/digits/underscore/hyphen")
    if args.repeat < 1 or args.seed < 0 or args.seed + args.repeat > 2 ** 32:
        parser.error("repeat must be positive; seeds must fit uint32")
    jobs = plan(ROOT, args.experiment, args.repeat, args.seed, args.loss, args.include_no_skip)
    print("M1 graph ablation; {} runs; experiment={}".format(len(jobs), args.experiment), flush=True)
    if args.dry_run:
        for command in list(suite.PREPARE) + [j["args"] for j in jobs]:
            print("python -m wpf_benchmark " + " ".join(shlex.quote(x) for x in command))
        return 0
    versions = suite.dependencies(require_gbdt=False)
    import fcntl
    directory = ROOT / "reports/rerun_m1" / args.experiment
    directory.mkdir(parents=True, exist_ok=True)
    # Share the full-suite lock so neither runner can overwrite preparation.
    with (directory.parent / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another M1 suite is already running in this project")
        suite.run_suite(ROOT, directory, jobs, args.experiment, versions, EXTRA_SOURCES)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, ImportError, OSError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
