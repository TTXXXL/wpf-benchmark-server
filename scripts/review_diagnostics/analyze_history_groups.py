"""Historical calm cohorts and realized-transition diagnostics; never select thresholds on test errors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

try:
    from .audit_labels import clean_table
    from .common import (DAY_NS, STEP_NS, check_fingerprints, data_digest, equal_saved_values, load_manifest, metrics, new_output,
                         persistence, source_summary, write_csv, write_json)
except ImportError:
    from audit_labels import clean_table
    from common import (DAY_NS, STEP_NS, check_fingerprints, data_digest, equal_saved_values, load_manifest, metrics, new_output,
                        persistence, source_summary, write_csv, write_json)


def historical_calm(wind_grid, times, issue_times, k, threshold):
    """Only wind[t-k+1:t+1] is accessed; a nonfinite/negative tick makes history unknown."""
    if type(k) is not int or k < 1 or threshold <= 0 or not np.isfinite(threshold):
        raise ValueError('Positive integer K and finite positive threshold required')
    if len(times) != wind_grid.shape[0] or not np.all(np.diff(times) == STEP_NS):
        raise ValueError('Historical grid must be continuous at ten-minute resolution')
    indexes = np.searchsorted(times, issue_times)
    if np.any(indexes >= len(times)) or not np.array_equal(times[indexes], issue_times):
        raise ValueError('Issue time missing from history grid')
    if np.any(indexes - k + 1 < 0):
        raise ValueError('Not enough pre-issue history')
    known = np.ones((len(indexes), wind_grid.shape[1]), dtype=bool)
    calm = known.copy()
    for lag in range(k):
        selected_times = times[indexes - lag]
        if np.any(selected_times > issue_times):
            raise AssertionError('History definition accessed future values')
        wind = wind_grid[indexes - lag]
        observed = np.isfinite(wind) & (wind >= 0)
        known &= observed
        calm &= observed & (wind < threshold)
    return calm, known


def cohorts(history_calm, history_known, future_wind, threshold):
    future_known = np.all(np.isfinite(future_wind) & (future_wind >= 0), axis=2)
    future_calm = future_known & np.all(future_wind < threshold, axis=2)
    not_history_calm = history_known & ~history_calm
    both_known = history_known & future_known
    return {
        'H_calm': history_calm,
        'S1_persistent_calm': history_calm & future_calm,
        'S2_enter_calm': not_history_calm & future_calm,
        'S3_leave_calm': history_calm & future_known & ~future_calm,
        'S4_noncalm': not_history_calm & future_known & ~future_calm,
        'unknown_transition': ~both_known,
    }


def load_history(reference, project_root, max_k):
    arrays = reference['arrays']
    earliest = int(arrays['times'][0] - (max_k - 1) * STEP_NS)
    latest = int(arrays['times'][-1])
    frame = clean_table(project_root, earliest // DAY_NS + 1, latest // DAY_NS + 1,
                        ['Day', 'TurbID', 'ts', 'Wspd', 'Patv'])
    grid_times = np.arange(earliest, latest + STEP_NS, STEP_NS, dtype=np.int64)
    turbines = arrays['turbine_ids']
    wind = frame.pivot(index='ts_ns', columns='TurbID', values='Wspd').reindex(index=grid_times, columns=turbines)
    power = frame.pivot(index='ts_ns', columns='TurbID', values='Patv').reindex(index=grid_times, columns=turbines)
    # Check every saved target wind and issue power before using a local historical snapshot.
    start = max_k - 1
    for h in range(arrays['forecasts'].shape[2]):
        actual = wind.to_numpy()[start + h + 1:start + h + 1 + arrays['forecasts'].shape[0]]
        if not equal_saved_values(actual, arrays['wind_speed'][..., h]):
            raise ValueError('Local history snapshot disagrees with saved target wind grid')
    actual_power = power.to_numpy()[start:start + arrays['forecasts'].shape[0]]
    if not equal_saved_values(actual_power, arrays['last_power']):
        raise ValueError('Local history snapshot disagrees with saved issue power grid')
    return wind.to_numpy(), grid_times


def saved_history(arrays, max_k):
    """Recover the original experiment's wind; unknown pre-test values stay NaN.

    A target observed by a previous issue is historical at the current issue.
    The cohort selector still only reads timestamps <= its own issue time.
    """
    targets = arrays['wind_speed']
    count, turbines, horizon = targets.shape
    grid = np.full((count + horizon, turbines), np.nan, dtype=targets.dtype)
    grid[1:count + 1] = targets[..., 0]
    for h in range(1, horizon):
        grid[count + h] = targets[-1, :, h]
    for h in range(horizon):
        if not np.array_equal(grid[1 + h:1 + h + count], targets[..., h], equal_nan=True):
            raise ValueError('Saved overlapping target wind disagrees across horizons')
    padding = np.full((max_k - 1, turbines), np.nan, dtype=targets.dtype)
    first = int(arrays['times'][0] - (max_k - 1) * STEP_NS)
    times = first + np.arange(len(grid) + len(padding), dtype=np.int64) * STEP_NS
    return np.concatenate([padding, grid]), times


def paired_seed_summary(rows, runs):
    expected = {'{}_s{}'.format(model, seed) for model in ('A', 'B', 'C') for seed in range(5)}
    if set(runs) != expected:
        raise ValueError('--paired-seeds requires exactly A_s0..4, B_s0..4, C_s0..4')
    check_fingerprints(list(runs.values()))
    for label, run in runs.items():
        seed = run['metadata'].get('seed')
        if type(seed) is not int or seed != int(re.fullmatch(r'[ABC]_s([0-4])', label).group(1)):
            raise ValueError('Manifest seed label disagrees with metadata: ' + label)
        if run['metadata'].get('model_config') != runs[label[0] + '_s0']['metadata'].get('model_config'):
            raise ValueError('Model config changed across seeds: ' + label)
    groups = {}
    for row in rows:
        key = (row['K'], row['threshold_mps'], row['group'])
        groups.setdefault(key, {})[row['model']] = row
    output = []
    for (k, threshold, group), values in groups.items():
        base = values['persistence']
        summary = {'K': k, 'threshold_mps': threshold, 'group': group, 'n': base['n'],
                   'persistence_MAE_kW': base['MAE_kW'], 'seed_n': 5}
        for model in ('A', 'B', 'C'):
            entries = [values['{}_s{}'.format(model, seed)] for seed in range(5)]
            for metric in ('MAE_kW', 'ME_kW'):
                measurements = [entry[metric] for entry in entries]
                summary[model + '_' + metric + '_mean'] = float(np.mean(measurements)) if base['n'] else None
                summary[model + '_' + metric + '_sd'] = float(np.std(measurements, ddof=1)) if base['n'] else None
        for left, right in (('C', 'A'), ('B', 'A'), ('C', 'persistence')):
            differences = [values[left + '_s' + str(seed)]['MAE_kW'] -
                           (base['MAE_kW'] if right == 'persistence' else values[right + '_s' + str(seed)]['MAE_kW'])
                           for seed in range(5)] if base['n'] else []
            prefix = left + '_minus_' + right
            summary[prefix + '_MAE_kW_by_seed'] = differences
            summary[prefix + '_MAE_kW_mean'] = float(np.mean(differences)) if differences else None
            summary[prefix + '_MAE_kW_sd'] = float(np.std(differences, ddof=1)) if differences else None
            summary[prefix + '_all_seeds_negative'] = bool(all(x < 0 for x in differences)) if differences else None
        output.append(summary)
    return output


def analyze(runs, project_root, k, threshold, sensitivity=False, allow_mismatch=False,
            threshold_source='spec_fixed', selection_period='not selected using errors',
            history_source='local', paired_seeds=False):
    reference = next(iter(runs.values()))
    if paired_seeds:
        paired_seed_summary([], runs)  # Validate seeds/fingerprints/configs before computing metrics.
    if history_source not in ('local', 'saved-wind'):
        raise ValueError('Unknown history source')
    if history_source == 'local' and project_root is None:
        raise ValueError('Local history requires --project-root')
    if history_source == 'saved-wind' and allow_mismatch:
        raise ValueError('Saved-wind history does not use --allow-data-digest-mismatch')
    local_digest = data_digest(project_root) if history_source == 'local' else None
    expected = reference['metadata']['data_digest']
    if history_source == 'local' and local_digest != expected and not allow_mismatch:
        raise ValueError('Local data_digest differs from saved runs. Restore matching data, or use '
                         '--allow-data-digest-mismatch for explicitly exploratory locally reconstructed history only.')
    input_window = reference['metadata']['config'].get('input_window', 144)
    if k > input_window:
        raise ValueError('K cannot exceed the model historical input window')
    if threshold_source == 'validation' and selection_period.startswith('not selected'):
        raise ValueError('Validation-selected thresholds require --selection-period describing that validation period')
    settings = [(k, threshold)]
    if sensitivity:
        settings.extend((kk, vv) for kk in (6, 12, 36) for vv in (0.5, 1.0, 1.5)
                        if (kk, vv) != (k, threshold))
    if max(x[0] for x in settings) > input_window:
        raise ValueError('Sensitivity K exceeds model input window')
    maximum = max(x[0] for x in settings)
    wind, times = (load_history(reference, project_root, maximum) if history_source == 'local'
                   else saved_history(reference['arrays'], maximum))
    a = reference['arrays']
    if not a['valid_m1'].any():
        raise ValueError('History diagnosis requires nonempty M1 scored population')
    predictions = {name: run['arrays']['forecasts'] for name, run in runs.items()}
    predictions['persistence'] = persistence(a)
    rows, compositions, overlaps = [], [], []
    for kk, vv in settings:
        calm, known = historical_calm(wind, times, a['times'][:a['forecasts'].shape[0]], kk, vv)
        groups = cohorts(calm, known, a['wind_speed'], vv)
        partition = [groups[name] for name in groups if name != 'H_calm']
        if not np.all(np.sum(partition, axis=0) == 1):
            raise AssertionError('Diagnostic transitions do not form a disjoint complete partition')
        groups['overall'] = np.ones_like(calm)
        for name, group in groups.items():
            take = a['valid_m1'] & group[..., None]
            compositions.append({'K': kk, 'threshold_mps': vv, 'group': name,
                                 'issue_turbine_pairs': int(group.sum()),
                                 'issue_pair_share_pct': 100 * float(group.mean()),
                                 'valid_forecast_cells': int(take.sum()),
                                 'valid_cell_share_pct': 100 * int(take.sum()) / int(a['valid_m1'].sum())})
            baseline = metrics(predictions['persistence'], a['truth_m1'], take)
            for model, predicted in predictions.items():
                result = metrics(predicted, a['truth_m1'], take)
                delta = None if not result['n'] else result['MAE_kW'] - baseline['MAE_kW']
                rows.append({'K': kk, 'threshold_mps': vv, 'group': name, 'model': model,
                             **result, 'MAE_minus_persistence_kW': delta,
                             'contribution_to_all_M1_delta_kW': None if delta is None else
                             delta * result['n'] / int(a['valid_m1'].sum())})
        for first, ga in groups.items():
            for second, gb in groups.items():
                overlaps.append({'K': kk, 'threshold_mps': vv, 'first': first, 'second': second,
                                 'overlap_issue_pairs': int((ga & gb).sum())})
    return {'primary': {'K': k, 'threshold_mps': threshold, 'threshold_source': threshold_source,
                        'selection_period': selection_period},
            'metrics': rows, 'composition': compositions, 'overlap': overlaps,
            'local_data_digest': local_digest, 'experiment_data_digest': expected,
            'data_digest_matches': local_digest == expected if history_source == 'local' else None,
            'history_source': history_source,
            'evidence_status': ('saved_experiment_wind_history' if history_source == 'saved-wind' else
                                'matched_data' if local_digest == expected else 'exploratory_local_history_snapshot'),
            'paired_seed_summary': paired_seed_summary(rows, runs) if paired_seeds else [],
            'history_policy': 'All K cleaned causal wind ticks finite, nonnegative and below threshold; missing/negative => unknown. '
                              'No quality flags are used to claim raw sensor validity; imputed causal history can participate.',
            'future_policy': 'All H future wind ticks finite/nonnegative; any tick >= threshold is leave-calm. '
                             'Unknown future wind is excluded from S1/S2/S3/S4 and reported separately.',
            'inference_scope': 'H_calm uses historical information only; S1/S2/S3/S4 use realized future wind for diagnosis. '
                               'Saved-wind mode certifies the saved wind history only; first K issues lack full history and are unknown. '
                               'It does not restore the complete preprocessing snapshot. In local mode a matching target/issue grid '
                               'cannot certify pre-test historical inputs when the whole-data digest differs.',
            'sources': source_summary(runs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', required=True)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--project-root')
    parser.add_argument('--history-source', choices=['local', 'saved-wind'], default='local')
    parser.add_argument('--paired-seeds', action='store_true')
    parser.add_argument('--out', required=True)
    parser.add_argument('--history-steps', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=1.0)
    parser.add_argument('--sensitivity', action='store_true')
    parser.add_argument('--allow-data-digest-mismatch', action='store_true')
    parser.add_argument('--threshold-source', choices=['spec_fixed', 'validation'], default='spec_fixed')
    parser.add_argument('--selection-period', default='not selected using errors; frozen in implementation specification')
    args = parser.parse_args()
    result = analyze(load_manifest(args.reports, args.runs), args.project_root, args.history_steps,
                     args.threshold, args.sensitivity, args.allow_data_digest_mismatch,
                     args.threshold_source, args.selection_period, args.history_source, args.paired_seeds)
    output = new_output(args.out)
    write_json(output / 'history_groups.json', result)
    for key in ('metrics', 'composition', 'overlap'):
        write_csv(output / ('history_' + key + '.csv'), result[key])
    if result['paired_seed_summary']:
        write_csv(output / 'history_paired_seeds.csv', result['paired_seed_summary'])
    print(json.dumps({'primary': result['primary'], 'evidence_status': result['evidence_status'],
                      'primary_composition': [row for row in result['composition']
                                              if row['K'] == args.history_steps and row['threshold_mps'] == args.threshold]}))


if __name__ == '__main__':
    main()
