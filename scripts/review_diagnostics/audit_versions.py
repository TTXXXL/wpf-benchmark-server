"""Read-only Git source fingerprints, changed-path evidence, and saved prediction equivalence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

try:
    from .common import load_manifest, new_output, source_summary, write_csv, write_json
except ImportError:
    from common import load_manifest, new_output, source_summary, write_csv, write_json

OLD_DIGESTS = {'d24aabd79d077fca', '3aba07e87c45311a', '024ad6146be65256', '8e8d82853cd1a139'}


def git(repository, *arguments, binary=False, input_bytes=None):
    process = subprocess.run(['git', '-C', str(repository)] + list(arguments),
                              input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return process.stdout if binary else process.stdout.decode('utf-8', errors='replace')


def historical_sources(repository, sha):
    # cat-file reads object bytes directly: git archive can apply export attributes/EOL conversion.
    prefix = 'src/wpf_benchmark/'
    paths = [path for path in git(repository, 'ls-tree', '-r', '--name-only', sha, prefix).splitlines()
             if path.endswith('.py')]
    if not paths:
        raise ValueError('Commit has no wpf_benchmark source: ' + sha)
    requests = ''.join('{}:{}\n'.format(sha, path) for path in paths).encode('utf-8')
    payload = git(repository, 'cat-file', '--batch', binary=True, input_bytes=requests)
    files = {}
    for path in paths:
        header, payload = payload.split(b'\n', 1)
        fields = header.split()
        if len(fields) != 3 or fields[1] != b'blob':
            raise ValueError('Expected Git blob: {}'.format(header))
        size = int(fields[2])
        if len(payload) < size + 1 or payload[size:size + 1] != b'\n':
            raise ValueError('Invalid Git batch payload')
        files[path[len(prefix):]] = payload[:size]
        payload = payload[size + 1:]
    if payload:
        raise ValueError('Unexpected remaining Git batch payload')
    return files


def source_digests(files):
    """Hash the same relative paths as result_provenance; report Git and checkout line endings."""
    hashes = {kind: hashlib.sha256() for kind in ('git_blob', 'lf', 'crlf')}
    for path, content in sorted(files.items()):
        if not path.endswith('.py') or ('figures/' in path and path != 'figures/io.py') or '__pycache__/' in path:
            continue
        lf = content.replace(b'\r\n', b'\n')
        versions = {'git_blob': content, 'lf': lf, 'crlf': lf.replace(b'\n', b'\r\n')}
        for kind, value in hashes.items():
            value.update(path.encode('utf-8'))
            value.update(versions[kind])
    return {kind: value.hexdigest()[:16] for kind, value in hashes.items()}


def normalized_sources(files):
    return {path: content.replace(b'\r\n', b'\n') for path, content in files.items()
            if path.endswith('.py') and ('figures/' not in path or path == 'figures/io.py')
            and '__pycache__/' not in path}


def changes_classification(paths):
    if any(path.startswith('src/wpf_benchmark/models/') or path in
           ('src/wpf_benchmark/runner.py', 'src/wpf_benchmark/data/scaling.py',
            'src/wpf_benchmark/data/windows.py', 'src/wpf_benchmark/data/masks.py',
            'src/wpf_benchmark/preprocessing/sdwpf.py') for path in paths):
        return 'potential_model_training_or_data_behavior_change'
    if any(path.startswith('src/wpf_benchmark/') for path in paths):
        return 'source_change_requires_execution_path_review'
    return 'outside_hashed_runtime_source; configuration_changes_still_require_review'


def scan(repository, targets, source_root=None):
    records = []
    checkout = None
    checkout_files = None
    if source_root is not None:
        root = Path(source_root)
        if not root.is_dir():
            raise ValueError('--source-root must be the existing src/wpf_benchmark directory')
        checkout_files = {path.relative_to(root).as_posix(): path.read_bytes()
                          for path in root.rglob('*.py')}
        digest = source_digests(checkout_files)
        checkout = {'source_root': str(root.resolve()), 'raw_digest': digest['git_blob'],
                    'lf_digest': digest['lf'], 'exact_normalized_source_matches': []}
    commits = git(repository, 'log', '--all', '--format=%H%x09%aI%x09%s').splitlines()
    for record in commits:
        sha, date, title = record.split('\t', 2)
        files = historical_sources(repository, sha)
        digests = source_digests(files)
        if checkout is not None and normalized_sources(checkout_files) == normalized_sources(files):
            different = [path for path in normalized_sources(files) if checkout_files[path] != files[path]]
            checkout['exact_normalized_source_matches'].append(
                {'commit': sha, 'bytewise_source_equal': not different,
                 'files_with_only_line_ending_differences': sorted(different)})
        paths = git(repository, 'diff-tree', '--root', '--no-commit-id', '--name-only', '-r', sha).splitlines()
        records.append({'commit': sha, 'date': date, 'title': title,
                        'code_digest': digests['git_blob'], 'lf_digest': digests['lf'],
                        'crlf_digest': digests['crlf'],
                        'matched_old_digests': sorted(set(digests.values()) & set(targets)),
                        'changed_paths': paths, 'classification': changes_classification(paths)})
    matched = {value for row in records for value in row['matched_old_digests']}
    if checkout is not None and checkout['exact_normalized_source_matches']:
        matched.add(checkout['raw_digest'])
    return {'commits': records, 'untraceable_digests': sorted(set(targets) - matched),
            'checkout_evidence': checkout,
            'method': 'Exact historical Git blobs plus explicit LF/CRLF variants using current result_provenance exclusions. '
                      'No match does not prove an invalid experiment: uncommitted source or older hash rules remain possible.',
            'inference_scope': 'Commit path classification is conservative evidence, not a proof of unchanged execution behavior.'}


def prediction_equivalence(runs):
    names = list(runs)
    if len(names) != 2:
        raise ValueError('--compare-runs requires exactly two labeled runs')
    left, right = (runs[name] for name in names)
    if left['metadata'].get('seed') != right['metadata'].get('seed'):
        raise ValueError('Prediction equivalence comparison requires the same seed')
    configs = [dict(item['metadata'].get('model_config', {})) for item in (left, right)]
    ignored_inactive_keys = []
    if all(not config.get('low_power_head', False) for config in configs):
        ignored_inactive_keys = ['low_power_head', 'low_power_threshold_kw', 'state_loss_weight', 'state_history_steps']
        for config in configs:
            for key in ignored_inactive_keys:
                config.pop(key, None)
    if configs[0] != configs[1]:
        raise ValueError('Prediction equivalence comparison requires the same effective model config')
    a, b = left['arrays']['forecasts'], right['arrays']['forecasts']
    identical = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    return {'sources': source_summary(runs), 'forecast_bytes_identical': identical,
            'raw_model_configs_equal': left['metadata'].get('model_config') == right['metadata'].get('model_config'),
            'ignored_inactive_low_power_options': ignored_inactive_keys,
            'max_absolute_difference_kw': float(np.max(np.abs(a.astype(float) - b.astype(float)))),
            'scope': 'This configuration/seed and saved test grid only; cannot certify other models or seeds.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source-root', help='Optional local src/wpf_benchmark; compare exact normalized bytes to Git')
    parser.add_argument('--reports')
    parser.add_argument('--compare-runs')
    args = parser.parse_args()
    result = scan(args.repository, OLD_DIGESTS, args.source_root)
    if args.compare_runs:
        if not args.reports:
            raise ValueError('--compare-runs requires --reports')
        result['prediction_equivalence'] = prediction_equivalence(load_manifest(args.reports, args.compare_runs))
    output = new_output(args.out)
    write_json(output / 'version_audit.json', result)
    write_csv(output / 'version_audit.csv', result['commits'])
    print(json.dumps({key: value for key, value in result.items() if key != 'commits'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
