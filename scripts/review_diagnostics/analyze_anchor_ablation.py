"""Strict five-seed anchor analysis; negative anchored-minus-unanchored means improvement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .common import (WIND_BINS, SUB_BINS, check_alignment, check_fingerprints,
                         discover, metrics, new_output, persistence, source_summary, write_csv, write_json)
except ImportError:
    from common import (WIND_BINS, SUB_BINS, check_alignment, check_fingerprints,
                        discover, metrics, new_output, persistence, source_summary, write_csv, write_json)


def analyze(reports, experiment):
    cells = {tag: discover(reports, tag, experiment) for tag in ('lite_mae_skip', 'lite_mae_noskip')}
    runs = [run for cell in cells.values() for run in cell.values()]
    check_fingerprints(runs)
    reference = cells['lite_mae_skip'][0]
    for run in runs:
        check_alignment(reference, run)
    model_base = None
    for tag, cell in cells.items():
        for run in cell.values():
            configuration = dict(run['metadata'].get('model_config', {}))
            if configuration.pop('current_power_skip', None) is not (tag == 'lite_mae_skip'):
                raise ValueError('Effective model config does not match anchor tag')
            if model_base is None:
                model_base = configuration
            elif configuration != model_base:
                raise ValueError('Effective model configs differ beyond current_power_skip')
    a = reference['arrays']
    truth, valid, wind = a['truth_m1'], a['valid_m1'], a['wind_speed']
    bins = {'overall': valid}
    bins.update({'[{},{} )'.format(lo, hi).replace(' ', ''): valid & (wind >= lo) & (wind < hi)
                 for lo, hi in WIND_BINS})
    bins.update({'sub[{},{} )'.format(lo, hi).replace(' ', ''): valid & (wind >= lo) & (wind < hi)
                 for lo, hi in SUB_BINS})
    bins['truth_lt10kW'] = valid & (truth < 10)
    rows = []
    for name, mask in bins.items():
        row = {'group': name, 'n': int(mask.sum()),
               'persistence_MAE': metrics(persistence(a), truth, mask)['MAE_kW']}
        if not mask.any():
            row.update({'anchor_delta_MAE_mean': None, 'anchor_delta_MAE_sd': None,
                        'anchor_all_seeds_better': None})
            rows.append(row)
            continue
        values = {}
        for tag, cell in cells.items():
            values[tag] = [metrics(cell[seed]['arrays']['forecasts'], truth, mask) for seed in range(5)]
            row[tag + '_MAE_mean'] = float(np.mean([x['MAE_kW'] for x in values[tag]]))
            row[tag + '_MAE_sd'] = float(np.std([x['MAE_kW'] for x in values[tag]], ddof=1))
            row[tag + '_ME_mean'] = float(np.mean([x['ME_kW'] for x in values[tag]]))
        # Anchored - unanchored: negative delta consistently means improvement.
        delta = [on['MAE_kW'] - off['MAE_kW'] for on, off in
                 zip(values['lite_mae_skip'], values['lite_mae_noskip'])]
        row.update({'anchor_delta_MAE_mean': float(np.mean(delta)),
                    'anchor_delta_MAE_sd': float(np.std(delta, ddof=1)),
                    'anchor_delta_MAE_by_seed': delta,
                    'anchor_all_seeds_better': bool(all(value < 0 for value in delta))})
        rows.append(row)
    return {'experiment': experiment, 'sign_convention': 'anchored MAE - unanchored MAE; negative = improvement',
            'seeds': list(range(5)), 'groups': rows,
            'sources': source_summary({'{}_s{}'.format(tag, seed): run
                                       for tag, cell in cells.items() for seed, run in cell.items()}),
            'inference_scope': 'Training randomness on one fixed test segment; all-same-sign is descriptive.'}


def main(default_out=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', required=True)
    parser.add_argument('--experiment', default='anchor_ablation')
    parser.add_argument('--out', default=str(default_out) if default_out else None, required=default_out is None)
    args = parser.parse_args()
    result = analyze(args.reports, args.experiment)
    output = new_output(args.out)
    write_json(output / 'H_锚点消融.json', result)
    write_csv(output / 'H_锚点消融.csv', result['groups'])
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
