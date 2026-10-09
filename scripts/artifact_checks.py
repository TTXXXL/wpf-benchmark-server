"""Saved-run integrity helpers used by resumable training (no analysis CLI)."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

STEP_NS = 600 * 10**9
ALIGN_KEYS = ('truth_m1', 'valid_m1', 'truth_m2', 'valid_m2', 'times',
              'turbine_ids', 'wind_speed', 'last_power')


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
