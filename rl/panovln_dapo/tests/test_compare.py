import json
from pathlib import Path
import tempfile
import unittest
from panovln_dapo.compare import compare, PROTOCOL


class ComparisonTests(unittest.TestCase):
    def make_run(self, path, rewards):
        path.mkdir()
        config = {k: 'fixed' for k in PROTOCOL}
        config['episode_keys'] = ['a', 'b']
        (path/'config.json').write_text(json.dumps(config))
        (path/'complete.json').write_text(json.dumps({'episodes': 2, 'expected_episodes': 2}))
        records = [{'episode_key': k, 'success': r, 'spl': r*.5,
                    'distance_to_goal': 5-4*r, 'oracle_success': r}
                   for k, r in zip(['a', 'b'], rewards)]
        (path/'results.jsonl').write_text('\n'.join(json.dumps(r) for r in reversed(records)))

    def test_pair_by_identity_not_file_order(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'a', Path(directory)/'b'
            self.make_run(a, [0, 1]); self.make_run(b, [1, 1])
            report = compare(a, b, samples=100)
            self.assertEqual(report['success_gains'], 1)
            self.assertEqual(report['success_losses'], 0)
            self.assertEqual(report['metrics']['success']['delta_candidate_minus_baseline'], .5)
            self.assertEqual(report['metrics']['distance_to_goal']['delta_candidate_minus_baseline'], -2)
            config = json.loads((b/'config.json').read_text())
            config['actions_per_replan'] = 'changed'
            (b/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'protocol differs'): compare(a, b, samples=100)

    def test_incomplete_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'a', Path(directory)/'b'
            self.make_run(a, [0, 1]); self.make_run(b, [1, 1])
            record = (b/'results.jsonl').read_text().splitlines()[0]
            for content in [record, record+'\n'+record]:
                (b/'results.jsonl').write_text(content)
                with self.assertRaisesRegex(ValueError, 'Incomplete'): compare(a, b, samples=100)
