"""Frozen 8-cell x 5-seed M1 review suite. Linux execution; portable dry-run."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import rerun_m1 as suite

ROOT = suite.ROOT
CELLS = (
    ('agcrn_noskip', 'agcrn', 'graph_ablation/agcrn_mae_no_skip', 'agcrn_mae_noskip'),
    ('agcrn_skip', 'agcrn', 'agcrn_mae', 'agcrn_mae_skip'),
    ('lite_noskip_mse', 'agcrn_lite', 'graph_ablation/agcrn_lite_mse_no_skip', 'lite_mse_noskip'),
    ('lite_skip_mse', 'agcrn_lite', 'agcrn_lite_mse', 'lite_mse_skip'),
    ('lite_noskip_mae', 'agcrn_lite', 'agcrn_lite_mae_noskip', 'lite_mae_noskip'),
    ('A', 'agcrn_lite', 'agcrn_lite_mae', 'lite_mae_skip'),
    ('B', 'agcrn_lite', 'agcrn_lite_low_power_no_aux', 'lite_mixture_no_aux'),
    ('C', 'agcrn_lite', 'agcrn_lite_low_power_mae', 'lite_low_power'),
)
EXTRA_SOURCES = ('scripts/rerun_low_power_review.py', 'scripts/artifact_checks.py')
PROTOCOL = dict(target_mask='m1', eval_mask='m1', input_window=144,
                horizon=12, train_days=196, val_days=25)


def without(value, *keys):
    return {key: item for key, item in value.items() if key not in keys}


def plan(root, experiment):
    settings = {label: suite.read_json(root / ('configs/' + config + '.json'))
                for label, _, config, _ in CELLS}
    for label, config in settings.items():
        if any(PROTOCOL.get(key) != value for key, value in config.get('protocol', {}).items()):
            raise ValueError('Frozen M1 protocol changed: ' + label)
        if config.get('seed') != 0:
            raise ValueError('Config seed baseline must be 0: ' + label)
        params = config['model']
        expected_loss = 'mse' if label in ('lite_noskip_mse', 'lite_skip_mse') else 'mae'
        expected_skip = label not in ('agcrn_noskip', 'lite_noskip_mse', 'lite_noskip_mae')
        if params.get('loss') != expected_loss or params.get('current_power_skip') is not expected_skip:
            raise ValueError('Wrong loss/anchor cell: ' + label)
    for left, right in (('agcrn_noskip', 'agcrn_skip'), ('lite_noskip_mse', 'lite_skip_mse'),
                        ('lite_noskip_mae', 'A')):
        if without(settings[left]['model'], 'current_power_skip') != without(settings[right]['model'], 'current_power_skip'):
            raise ValueError('Anchor pair differs beyond current_power_skip')
    if without(settings['lite_skip_mse']['model'], 'loss') != without(settings['A']['model'], 'loss'):
        raise ValueError('Lite loss pair differs beyond loss')
    if without(settings['B']['model'], 'state_loss_weight') != without(settings['C']['model'], 'state_loss_weight'):
        raise ValueError('B/C differ beyond state_loss_weight')
    for label, weight in (('B', 0.0), ('C', 0.05)):
        params = settings[label]['model']
        if (params.get('low_power_head') is not True or params.get('state_loss_weight') != weight
                or params.get('low_power_threshold_kw') != 10.0 or params.get('state_history_steps') != 12):
            raise ValueError('Low-power cell changed: ' + label)
        if without(params, 'low_power_head', 'low_power_threshold_kw', 'state_loss_weight', 'state_history_steps') != settings['A']['model']:
            raise ValueError('Low-power cells changed the A backbone/training recipe')
    jobs = []
    for label, model, config, tag in CELLS:
        for seed in range(5):
            args = ['run', '--model', model, '--config', 'configs/' + config + '.json',
                    '--target-mask', 'm1', '--eval-mask', 'm1', '--seed', str(seed), '--repeat', '1',
                    '--experiment', experiment, '--tag', tag, '--save-arrays', '--no-plots']
            jobs.append(dict(key='{}_s{}'.format(label, seed), label=label, model=model, seed=seed,
                             tag=tag, args=args, model_config=settings[label]['model']))
    return jobs


def processed_hashes(root):
    if not all((root / path).is_file() for path in suite.PROCESSED):
        raise RuntimeError('Cleaned data missing; use --prepare only in a fresh project copy')
    return {path: suite.sha256(root / path) for path in suite.PROCESSED}


def provenance(root):
    from wpf_benchmark.figures.io import result_provenance
    from wpf_benchmark.paths import ProjectPaths
    data, code = result_provenance(ProjectPaths.resolve(root))
    return dict(data_digest=data, code_digest=code)


def git_commit(root):
    try:
        result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError:
        return None
    return result.stdout.decode('ascii').strip() if result.returncode == 0 else None


def verify_freeze(root, jobs, manifest):
    if (suite.fingerprint(root, jobs, EXTRA_SOURCES) != manifest['sources'] or
            processed_hashes(root) != manifest['processed'] or provenance(root) != manifest['digests'] or
            git_commit(root) != manifest.get('git_commit')):
        raise RuntimeError('Frozen code/config/raw/processed data changed; stop and use a new experiment')


def verify_result(root, job, experiment, digests, reference=None):
    from artifact_checks import check_alignment, load_run
    from wpf_benchmark.evaluation.config import ProtocolConfig
    candidates = []
    for path in (root / 'reports/eval').glob(job['tag'] + '_main_*.json'):
        result = suite.read_json(path)
        if result.get('experiment') == experiment and result.get('seed') == job['seed']:
            candidates.append((path, result))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError('Duplicate result for ' + job['key'] + '; do not silently select newest')
    main, result = candidates[0]
    if not suite.matches_job(result, job, experiment) or any(result.get(key) != value for key, value in digests.items()):
        raise RuntimeError('Result identity/config/fingerprint mismatch: ' + str(main))
    protocol = json.loads(json.dumps(asdict(ProtocolConfig())))
    if result.get('config') != protocol:
        raise RuntimeError('Result full protocol changed: ' + str(main))
    run = load_run(root / 'reports/eval', result['run_id'])
    if reference is not None:
        check_alignment(reference, run)
    files = [main, main.with_name(main.name.replace(job['tag'] + '_main_', job['tag'] + '_all_', 1)),
             Path(run['array_path']), root / 'reports/train' / (result['run_id'] + '.log')]
    files.extend(suite.checkpoint_artifacts(root, result))
    if any(not path.is_file() or not path.stat().st_size for path in files):
        raise RuntimeError('Incomplete output for ' + job['key'] + '; inspect artifacts before retrying')
    if suite.read_json(files[1]).get('run_id') != result['run_id']:
        raise RuntimeError('Main/all run IDs differ')
    marker = dict(main=main.relative_to(root).as_posix(), run_id=result['run_id'],
                  artifacts={path.relative_to(root).as_posix(): path.stat().st_size for path in files},
                  artifact_sha256={path.relative_to(root).as_posix(): suite.sha256(path) for path in files})
    return marker, run


def run_suite(root, directory, jobs, experiment, versions, prepare=False):
    path = directory / 'manifest.json'
    sources = suite.fingerprint(root, jobs, EXTRA_SOURCES)
    identity = dict(schema=1, experiment=experiment, jobs=jobs, environment=versions,
                    sources=sources, git_commit=git_commit(root),
                    preparation_policy='fresh_once' if prepare else 'reuse_frozen')
    if path.exists():
        manifest = suite.read_json(path)
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise RuntimeError('Code/config/environment/plan changed; use a new experiment')
        verify_freeze(root, jobs, manifest)
    else:
        # Refuse to adopt old results under a new manifest or overwrite cleaned data.
        if any(suite.read_json(p).get('experiment') == experiment for p in (root / 'reports/eval').glob('*_main_*.json')):
            raise RuntimeError('Experiment already has results without this suite manifest; use a new name')
        if prepare:
            if any((root / name).exists() for name in suite.PROCESSED):
                raise RuntimeError('--prepare refuses existing cleaned data; use a fresh project copy')
            if any(p.is_file() and p.name not in ('.gitkeep', '.lock')
                   for p in (root / 'reports').rglob('*')):
                raise RuntimeError('--prepare requires a fresh project copy without old reports')
            for args in suite.PREPARE[:2]:
                suite.execute(root, args, directory / 'prepare.log')
        processed = processed_hashes(root)
        suite.execute(root, ('eval', 'selftest'), directory / 'prepare.log')
        manifest = dict(identity, processed=processed, digests=provenance(root))
        verify_freeze(root, jobs, manifest)
        suite.save_json(path, manifest)
    reference = None
    mappings = {}
    for number, job in enumerate(jobs, 1):
        verify_freeze(root, jobs, manifest)
        marker_path = directory / (job['key'] + '.done.json')
        checked = verify_result(root, job, experiment, manifest['digests'], reference)
        if marker_path.exists():
            if checked is None or suite.read_json(marker_path) != checked[0]:
                raise RuntimeError('Saved artifacts changed: ' + job['key'])
            print('[skip {}/{}] {}'.format(number, len(jobs), job['key']), flush=True)
        elif checked is None:
            print('[run {}/{}] {}'.format(number, len(jobs), job['key']), flush=True)
            suite.execute(root, job['args'], directory / (job['key'] + '.console.log'))
            verify_freeze(root, jobs, manifest)
            checked = verify_result(root, job, experiment, manifest['digests'], reference)
            if checked is None:
                raise RuntimeError('Run produced no matched main output: ' + job['key'])
            suite.save_json(marker_path, checked[0])
        else:
            # A complete, verified result survives interruption before marker creation.
            suite.save_json(marker_path, checked[0])
        if reference is None:
            reference = checked[1]
        mappings[job['key']] = checked[0]['run_id']
        suite.summary(root, directory, jobs, experiment)
    suite.save_json(directory / 'all_runs.json', mappings)
    suite.save_json(directory / 'abc_history_runs.json', {key: value for key, value in mappings.items() if key[0] in 'ABC' and key[1:3] == '_s'})
    suite.save_json(directory / 'p0_s0_runs.json', {('lite_skip_mae' if job['label'] == 'A' else job['label']): mappings[job['key']] for job in jobs
                    if job['seed'] == 0 and job['label'] != 'lite_noskip_mae'})
    print('Completed {} runs; frozen digests: {}'.format(len(jobs), manifest['digests']), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, help='New unique suite name; reuse it only to resume')
    parser.add_argument('--dry-run', action='store_true', help='Print 40 commands; no writes/preparation/training')
    parser.add_argument('--prepare', action='store_true', help='Prepare only a fresh copy with no cleaned data')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.experiment):
        parser.error('experiment must be 1-64 letters/digits/underscore/hyphen')
    jobs = plan(ROOT, args.experiment)
    print('8 cells x seeds 0..4 = 40 runs; experiment=' + args.experiment, flush=True)
    if args.dry_run:
        for command in (suite.PREPARE if args.prepare else (('eval', 'selftest'),)):
            print('python -m wpf_benchmark ' + ' '.join(shlex.quote(x) for x in command))
        for job in jobs:
            print('python -m wpf_benchmark ' + ' '.join(shlex.quote(x) for x in job['args']))
        return 0
    sys.path.insert(0, str(ROOT / 'src'))
    versions = suite.dependencies(require_gbdt=False)
    import fcntl  # Actual execution uses the existing Linux suite lock.
    directory = ROOT / 'reports/rerun_m1' / args.experiment
    directory.mkdir(parents=True, exist_ok=True)
    with (directory.parent / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another M1 suite is running in this project')
        run_suite(ROOT, directory, jobs, args.experiment, versions, args.prepare)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, ImportError, OSError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
