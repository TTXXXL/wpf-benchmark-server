"""Trace saved evaluation slices to the original CSV, with explicit censoring/provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as arrow_csv
import pyarrow.parquet as parquet

try:
    from .common import data_digest, equal_saved_values, load_run, new_output, source_summary, write_csv, write_json
except ImportError:
    from common import data_digest, equal_saved_values, load_run, new_output, source_summary, write_csv, write_json


def target_rows(arrays, selection):
    issue, turbine, horizon = np.nonzero(selection)
    rows = pd.DataFrame({'TurbID': arrays['turbine_ids'][turbine],
                         'ts_ns': arrays['times'][issue + horizon + 1],
                         'npz_truth': arrays['truth_m1'][selection],
                         'npz_wind': arrays['wind_speed'][selection]})
    rows = rows.drop_duplicates()
    if rows.duplicated(['TurbID', 'ts_ns']).any():
        raise ValueError('Repeated forecast horizons disagree on one target label or wind')
    return rows


def power_counts(values):
    values = np.asarray(values)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Raw audit values must be nonempty and finite')
    return {'negative': int((values < 0).sum()), 'zero': int((values == 0).sum()),
            'positive': int((values > 0).sum()), 'minimum_kw': float(values.min()),
            'maximum_kw': float(values.max())}


def clean_table(project_root, min_day, max_day, columns):
    table = parquet.read_table(Path(project_root) / 'data/processed/sdwpf_clean.parquet',
                               columns=columns, filters=[('Day', '>=', min_day), ('Day', '<=', max_day)])
    frame = table.to_pandas()
    frame['ts_ns'] = pd.to_timedelta(frame['ts']).astype('timedelta64[ns]').astype('int64')
    if frame.duplicated(['TurbID', 'ts_ns']).any():
        raise ValueError('Duplicate cleaned-data time/turbine key')
    return frame


def audit(run, project_root):
    arrays = run['arrays']
    valid, wind = arrays['valid_m1'], arrays['wind_speed']
    slices = {'wind_0_1': valid & (wind >= 0) & (wind < 1),
              'wind_0_3': valid & (wind >= 0) & (wind < 3),
              'truth_lt10': valid & (arrays['truth_m1'] < 10)}
    earliest, latest = arrays['times'][0], arrays['times'][-1]
    clean = clean_table(project_root, int(earliest // (86400 * 10**9) + 1),
                        int(latest // (86400 * 10**9) + 1), ['Day', 'TurbID', 'ts', 'Patv', 'Wspd'])
    raw = arrow_csv.read_csv(Path(project_root) / 'data/raw/sdwpf/sdwpf_245days_v1.csv',
                              convert_options=arrow_csv.ConvertOptions(
                                  include_columns=['TurbID', 'Day', 'Tmstamp', 'Patv']))
    raw = raw.to_pandas()
    raw = raw[raw.Day.between(int(earliest // (86400 * 10**9) + 1),
                              int(latest // (86400 * 10**9) + 1))].copy()
    clock = raw.Tmstamp.astype(str)
    raw['ts_ns'] = (((raw.Day - 1) * 86400 + clock.str[:2].astype(int) * 3600 +
                     clock.str[3:5].astype(int) * 60) * 10**9)
    raw = raw.rename(columns={'Patv': 'raw_power_kw'})
    if raw.duplicated(['TurbID', 'ts_ns']).any():
        raise ValueError('Duplicate original CSV time/turbine key')
    rows = []
    for label, take in slices.items():
        cells = int(take.sum())
        if not cells:
            rows.append({'slice': label, 'forecast_cells': 0, 'distinct_targets': 0})
            continue
        selected = target_rows(arrays, take)
        matched = selected.merge(clean, on=['TurbID', 'ts_ns'], how='left', validate='one_to_one', indicator=True)
        if not (matched['_merge'] == 'both').all():
            raise ValueError('Saved slice has missing local cleaned-data keys: ' + label)
        if not equal_saved_values(matched.Patv, matched.npz_truth):
            raise ValueError('Saved/local clean label mismatch: ' + label)
        if not equal_saved_values(matched.Wspd, matched.npz_wind):
            raise ValueError('Saved/local wind mismatch: ' + label)
        matched = matched.drop(columns='_merge').merge(
            raw[['TurbID', 'ts_ns', 'raw_power_kw']], on=['TurbID', 'ts_ns'],
            how='left', validate='one_to_one', indicator=True)
        if not (matched['_merge'] == 'both').all():
            raise ValueError('Missing original CSV target keys: ' + label)
        row = {'slice': label, 'forecast_cells': cells, 'distinct_targets': len(matched),
               'cells_per_distinct_target': cells / len(matched),
               'raw_power': power_counts(matched.raw_power_kw),
               'clean_power_unique': np.unique(matched.Patv).tolist(),
               'alignment': 'one-to-one keys; truth and wind exactly equal after the NPZ dtype conversion'}
        rows.append(row)
    local = data_digest(project_root)
    expected = run['metadata'].get('data_digest')
    if not expected:
        raise ValueError('Run lacks data_digest')
    return {'slices': rows, 'local_data_digest': local, 'experiment_data_digest': expected,
            'data_digest_matches': local == expected, 'sources': source_summary({'reference': run}),
            'provenance_scope': 'A differing whole-file digest is retained explicitly; value alignment certifies only the audited slices.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = audit(load_run(args.reports, args.run_id), args.project_root)
    output = new_output(args.out)
    write_json(output / 'label_audit.json', result)
    write_csv(output / 'label_audit.csv', result['slices'])
    print(json.dumps({'slices': [{key: value for key, value in row.items() if key != 'clean_power_unique'}
                                  for row in result['slices']],
                      'local_data_digest': result['local_data_digest'],
                      'experiment_data_digest': result['experiment_data_digest'],
                      'data_digest_matches': result['data_digest_matches']}, ensure_ascii=False))
    if not result['data_digest_matches']:
        print('DATA DIGEST MISMATCH: only the explicitly aligned slices are certified.')


if __name__ == '__main__':
    main()
