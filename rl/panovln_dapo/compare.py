"""Compare complete fresh evaluations by episode; reject incompatible protocols."""
import argparse
import json
import math
import random
from pathlib import Path

PROTOCOL = ('dtype', 'attention', 'do_sample', 'actions_per_replan',
            'stop_commit_max_actions', 'max_env_steps', 'max_decisions',
            'manifest_sha256', 'episode_keys')
METRICS = ('success', 'spl', 'distance_to_goal', 'oracle_success')


def read_run(path):
    path = Path(path)
    if (path / 'failure.json').exists():
        raise ValueError(f'Failed evaluation: {path}')
    config = json.loads((path / 'config.json').read_text())
    complete = json.loads((path / 'complete.json').read_text())
    records = [json.loads(line) for line in (path / 'results.jsonl').read_text().splitlines() if line]
    keys = config['episode_keys']
    indexed = {r['episode_key']: r for r in records}
    if (not keys or len(set(keys)) != len(keys) or len(indexed) != len(records)
            or set(indexed) != set(keys) or complete['episodes'] != len(keys)
            or complete['expected_episodes'] != len(keys)):
        raise ValueError('Incomplete, duplicated or mismatched evaluation episodes')
    for record in records:
        for metric in METRICS:
            if not math.isfinite(record[metric]):
                raise ValueError(f'Nonfinite metric: {metric}')
        if record['success'] not in (0, 1) or record['oracle_success'] not in (0, 1):
            raise ValueError('Success metrics must be binary')
        if not 0 <= record['spl'] <= 1 or record['distance_to_goal'] < 0:
            raise ValueError('Invalid SPL or navigation error')
    return config, indexed


def compare(baseline, candidate, samples=10000, seed=42):
    if samples < 100:
        raise ValueError('Use at least 100 bootstrap samples')
    left_config, left = read_run(baseline)
    right_config, right = read_run(candidate)
    for key in PROTOCOL:
        if left_config[key] != right_config[key]:
            raise ValueError(f'Evaluation protocol differs: {key}')
    keys = left_config['episode_keys']
    n = len(keys)
    rng = random.Random(seed)
    deltas = {m: [right[k][m] - left[k][m] for k in keys] for m in METRICS}
    boot = {m: [] for m in METRICS}
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        for metric in METRICS:
            boot[metric].append(sum(deltas[metric][i] for i in indices) / n)
    metrics = {}
    for metric in METRICS:
        ordered = sorted(boot[metric])
        metrics[metric] = {
            'baseline': sum(left[k][metric] for k in keys) / n,
            'candidate': sum(right[k][metric] for k in keys) / n,
            'delta_candidate_minus_baseline': sum(deltas[metric]) / n,
            'paired_episode_bootstrap_95_interval': [ordered[int(.025*samples)], ordered[min(samples-1,int(.975*samples))]],
        }
    wins = sum(left[k]['success'] == 0 and right[k]['success'] == 1 for k in keys)
    losses = sum(left[k]['success'] == 1 and right[k]['success'] == 0 for k in keys)
    return {'episodes': n, 'metrics': metrics, 'success_gains': wins, 'success_losses': losses,
            'bootstrap_samples': samples, 'seed': seed,
            'scope': 'Paired episode comparison only; episodes may share scenes. This does not establish multi-seed or unseen-scene gains.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    with args.output.open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
