"""Score frozen forecasts under both saved masks; both targets retain negative-to-zero censoring."""
import argparse
import json

try:
    from .common import load_manifest, metrics, new_output, persistence, source_summary, write_csv, write_json
except ImportError:
    from common import load_manifest, metrics, new_output, persistence, source_summary, write_csv, write_json


def rescore(runs):
    reference = next(iter(runs.values()))['arrays']
    wind = reference['wind_speed']
    slices = {'overall': wind == wind, 'wind_0_3': (wind >= 0) & (wind < 3),
              'wind_0_1': (wind >= 0) & (wind < 1)}
    # Overall selection must not drop missing wind if the saved target mask accepts it.
    slices['overall'] = reference['valid_m1'] | reference['valid_m2']
    predictions = {name: run['arrays']['forecasts'] for name, run in runs.items()}
    predictions['persistence'] = persistence(reference)
    rows = []
    for mask in ('m1', 'm2'):
        valid, truth = reference['valid_' + mask], reference['truth_' + mask]
        for group, selection in slices.items():
            for name, predicted in predictions.items():
                rows.append({'model': name, 'mask': mask, 'group': group,
                             **metrics(predicted, truth, valid & selection)})
    populations = []
    for group, selection in slices.items():
        m1, m2 = reference['valid_m1'] & selection, reference['valid_m2'] & selection
        populations.append({'group': group, 'm1_n': int(m1.sum()), 'm2_n': int(m2.sum()),
                            'intersection_n': int((m1 & m2).sum()),
                            'm2_only_n': int((m2 & ~m1).sum()), 'm1_only_n': int((m1 & ~m2).sum())})
        for population, take in [('intersection', m1 & m2), ('m2_only', m2 & ~m1), ('m1_only', m1 & ~m2)]:
            masks = ('m1', 'm2') if population == 'intersection' else (('m2',) if population == 'm2_only' else ('m1',))
            for mask in masks:
                for name, predicted in predictions.items():
                    rows.append({'model': name, 'mask': mask, 'group': group,
                                 'population': population,
                                 **metrics(predicted, reference['truth_' + mask], take)})
    return {'metrics': rows, 'populations': populations, 'sources': source_summary(runs),
            'interpretation': 'Evaluation-population/target sensitivity of M1-trained frozen forecasts. '
                              'Both M1 and M2 censor original negative power to zero; this is not an uncensored-label control.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', required=True)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = rescore(load_manifest(args.reports, args.runs))
    output = new_output(args.out)
    write_json(output / 'm1_m2_rescore.json', result)
    write_csv(output / 'm1_m2_metrics.csv', result['metrics'])
    write_csv(output / 'm1_m2_populations.csv', result['populations'])
    print(json.dumps(result['populations'], ensure_ascii=False))


if __name__ == '__main__':
    main()
