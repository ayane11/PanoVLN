"""CPU-only strict paired comparison; run inside the container.

--expected-episodes must come from the evaluation plan, not completed output counts.
This checks result completeness, not equivalence of model/data/action settings.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.analyze_results import iter_result_paths, paired_bootstrap_comparison

REQUIRED_METRICS = ('success', 'spl', 'distance_to_goal', 'ndtw')


def read_complete_results(directory, expected):
    if expected <= 0:
        raise ValueError('expected episode count must be positive')
    rows = {}
    for path in iter_result_paths(Path(directory)):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                episode = row.get('id', row.get('episode_id'))
                scene = row.get('scene_id')
                if episode is None or scene is None:
                    raise ValueError('missing episode or scene identifier')
                row['id'], row['scene_id'] = str(episode), str(scene)
                key = (row['scene_id'], row['id'])
                for metric in REQUIRED_METRICS:
                    value = float(row[metric])
                    if not math.isfinite(value):
                        raise ValueError(f'nonfinite {metric}')
                    if metric == 'success' and value not in (0, 1):
                        raise ValueError('success must be 0 or 1')
                    if metric in ('spl', 'ndtw') and not 0 <= value <= 1:
                        raise ValueError(f'{metric} outside [0,1]')
                    if metric == 'distance_to_goal' and value < 0:
                        raise ValueError('negative distance')
                    row[metric] = value
                if key in rows and rows[key] != row:
                    raise ValueError(f'conflicting duplicate episode {key}')
                rows[key] = row
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f'{path}:{number}: {exc}') from exc
    if len(rows) != expected:
        raise ValueError(f'{directory}: expected {expected} episodes, found {len(rows)}')
    return rows


def compare(baseline, candidate, expected, samples=10000, seed=42):
    if samples <= 0:
        raise ValueError('bootstrap samples must be positive')
    a = read_complete_results(baseline, expected)
    b = read_complete_results(candidate, expected)
    if a.keys() != b.keys():
        raise ValueError(f'Episode sets differ: baseline-only={len(a.keys()-b.keys())}, candidate-only={len(b.keys()-a.keys())}')
    result = paired_bootstrap_comparison(list(a.values()), list(b.values()), num_samples=samples, seed=seed)
    result['strict_expected_episodes'] = expected
    result['baseline_path'] = str(Path(baseline).resolve())
    result['candidate_path'] = str(Path(candidate).resolve())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--expected-episodes', type=int, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.expected_episodes, args.bootstrap_samples, args.seed)
    # Never overwrite or merge previous reports; input shards are left intact.
    with Path(args.output).open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
