"""Strict saved-run validation and read-only diagnostic primitives (no training)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

DAY_NS = 86400 * 10**9
STEP_NS = 600 * 10**9
ALIGN_KEYS = ('truth_m1', 'valid_m1', 'truth_m2', 'valid_m2', 'times',
              'turbine_ids', 'wind_speed', 'last_power')
WIND_BINS = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 40)]
SUB_BINS = [(0, 1), (1, 2), (2, 2.5), (2.5, 3)]


def load_run(reports, run_id):
    reports = Path(reports)
    if Path(run_id).name != run_id or not run_id.endswith(tuple('0123456789')):
        raise ValueError('run_id must be a filename stem, without directories')
    match = re.fullmatch(r'(.+)_(\d{8}_\d{6}_\d{6})', run_id)
    if not match:
        raise ValueError('Invalid timestamped run_id: {}'.format(run_id))
    array_path = reports / (run_id + '_arrays.npz')
    metadata_path = reports / ('{}_main_{}.json'.format(*match.groups()))
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = set(ALIGN_KEYS) | {'forecasts'}
    missing = required - arrays.keys()
    if missing:
        raise ValueError('{} lacks arrays {}'.format(run_id, sorted(missing)))
    shape = arrays['forecasts'].shape
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError('Expected nonempty (issue,turbine,horizon) grid')
    for key in ('truth_m1', 'valid_m1', 'truth_m2', 'valid_m2', 'wind_speed'):
        if arrays[key].shape != shape:
            raise ValueError('{}: {} grid mismatch'.format(run_id, key))
    if arrays['times'].shape != (shape[0] + shape[2],):
        raise ValueError('{}: times length mismatch'.format(run_id))
    if arrays['times'].dtype.kind not in 'iu' or not np.all(np.diff(arrays['times']) == STEP_NS):
        raise ValueError('times must be a continuous 10-minute grid in nanoseconds')
    if arrays['turbine_ids'].shape != (shape[1],) or np.unique(arrays['turbine_ids']).size != shape[1]:
        raise ValueError('Invalid turbine_ids')
    if arrays['last_power'].shape != shape[:2]:
        raise ValueError('last_power grid mismatch')
    if not np.isfinite(arrays['forecasts']).all():
        raise ValueError('Forecasts must be finite')
    for mask in ('m1', 'm2'):
        valid = arrays['valid_' + mask]
        if valid.dtype.kind != 'b' or not np.isfinite(arrays['truth_' + mask][valid]).all():
            raise ValueError('Invalid mask or nonfinite scored targets: ' + mask)
    config = metadata.get('config', {})
    if config.get('horizon') != shape[2]:
        raise ValueError('{}: JSON horizon disagrees with arrays'.format(run_id))
    for key in ('target_mask', 'eval_mask'):
        if key in arrays and str(arrays[key].item()) != config.get(key):
            raise ValueError('{}: JSON/array {} mismatch'.format(run_id, key))
    return {'run_id': run_id, 'metadata': metadata, 'arrays': arrays,
            'array_path': str(array_path.resolve()), 'metadata_path': str(metadata_path.resolve())}


def check_alignment(reference, run):
    for key in ALIGN_KEYS:
        a, b = reference['arrays'][key], run['arrays'][key]
        if not np.array_equal(a, b, equal_nan=True):
            raise ValueError('{}: alignment mismatch in {}'.format(run['run_id'], key))
    if reference['metadata'].get('config') != run['metadata'].get('config'):
        raise ValueError('{}: protocol mismatch'.format(run['run_id']))


def load_manifest(reports, path):
    mapping = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(mapping, dict) or not mapping or not all(isinstance(v, str) for v in mapping.values()):
        raise ValueError('--runs must be a JSON object mapping labels to run_id strings')
    result = {}
    for label, run_id in mapping.items():
        run = load_run(reports, run_id)
        if result:
            check_alignment(next(iter(result.values())), run)
        result[label] = run
    digests = {r['metadata'].get('data_digest') for r in result.values()}
    if len(digests) != 1 or None in digests or '' in digests:
        raise ValueError('Saved-run data fingerprints missing or different')
    return result


def discover(reports, tag, experiment):
    result = {}
    for path in sorted(Path(reports).glob(tag + '_main_*.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta.get('experiment') != experiment:
            continue
        match = re.fullmatch(re.escape(tag) + r'_main_(\d{8}_\d{6}_\d{6})\.json', path.name)
        if not match:
            raise ValueError('Malformed run metadata filename: {}'.format(path))
        seed = meta.get('seed')
        if type(seed) is not int or seed not in range(5):
            raise ValueError('Expected integer seed 0..4: {}'.format(path))
        if seed in result:
            raise ValueError('Duplicate seed {} for {} / {}; inspect matching JSON files'.format(seed, tag, experiment))
        result[seed] = load_run(reports, tag + '_' + match.group(1))
    if set(result) != set(range(5)):
        raise ValueError('{} / {} must contain exactly seeds 0..4; found {}'.format(tag, experiment, sorted(result)))
    return result


def check_fingerprints(runs, require_code=True):
    keys = ('data_digest', 'code_digest') if require_code else ('data_digest',)
    for key in keys:
        values = {run['metadata'].get(key) for run in runs}
        if len(values) != 1 or None in values or '' in values:
            raise ValueError('{} missing or inconsistent; inspect run JSON and rerun under one frozen version'.format(key))


def metrics(prediction, truth, selection):
    n = int(selection.sum())
    if not n:
        return {'n': 0, 'MAE_kW': None, 'ME_kW': None, 'RMSE_kW': None, 'prediction_mean_kW': None}
    predicted = np.asarray(prediction[selection], dtype=np.float64)
    error = predicted - np.asarray(truth[selection], dtype=np.float64)
    if not np.isfinite(error).all():
        raise ValueError('Nonfinite scored forecast or target')
    return {'n': n, 'MAE_kW': float(np.abs(error).mean()), 'ME_kW': float(error.mean()),
            'RMSE_kW': float(np.sqrt(np.mean(error**2))), 'prediction_mean_kW': float(predicted.mean())}


def persistence(arrays):
    return np.broadcast_to(arrays['last_power'][..., None], arrays['forecasts'].shape)


def daily_errors(prediction, truth, selection, times):
    """Issue-day grouping: SDWPF relative days, 00:00 boundary, no timezone offset."""
    if prediction.shape != truth.shape or selection.shape != truth.shape or truth.ndim != 2:
        raise ValueError('Daily inputs must be aligned (issue,turbine) arrays')
    issue_days = np.asarray(times[:truth.shape[0]], dtype=np.int64) // DAY_NS
    rows = []
    for day in np.unique(issue_days):
        take = issue_days == day
        error = (prediction[take].astype(np.float64) - truth[take].astype(np.float64))[selection[take]]
        rows.append({'day': int(day + 1), 'n': int(error.size),
                     'me_kw': float(error.mean()) if error.size else None,
                     'mae_kw': float(abs(error).mean()) if error.size else None,
                     'sse': float(np.sum(error**2))})
    total = sum(row['sse'] for row in rows)
    top = sorted(rows, key=lambda row: (-row['sse'], row['day']))[:10]
    return {'day_definition': 'SDWPF relative Day = floor(issue_time_ns / 86400e9)+1; midnight boundary',
            'n_calendar_days': len(rows), 'n_days_with_selected_targets': sum(r['n'] > 0 for r in rows),
            'top10_days_share_of_sqerr_pct': 100 * sum(r['sse'] for r in top) / total if total else None,
            'top10_days': [r['day'] for r in top], 'daily': rows,
            'interpretation': 'Descriptive concentration; compare sample counts before attributing events.'}


def new_output(directory):
    path = Path(directory).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError('Output directory is nonempty; choose a new --out: {}'.format(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def data_digest(project_root):
    """Same path+bytes digest as figures/io.py:result_provenance, without code hashing."""
    value = hashlib.sha256()
    for relative in ('data/processed/sdwpf_clean.parquet', 'data/processed/sdwpf_meta.json'):
        value.update(relative.encode('utf-8'))
        path = Path(project_root) / relative
        if not path.is_file():
            raise ValueError('Missing data provenance input: {}'.format(path))
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                value.update(chunk)
    return value.hexdigest()[:16]


def equal_saved_values(local_values, saved_values):
    """Apply only the saved-array dtype conversion, then compare exactly (no loose tolerances)."""
    saved = np.asarray(saved_values)
    local = np.asarray(local_values)
    return local.shape == saved.shape and np.array_equal(local.astype(saved.dtype), saved, equal_nan=True)


def source_summary(runs):
    result = {}
    for name, run in runs.items():
        result[name] = {key: run[key] for key in ('run_id', 'array_path', 'metadata_path')}
        result[name].update({'code_digest': run['metadata'].get('code_digest'),
                             'data_digest': run['metadata'].get('data_digest')})
    return result
