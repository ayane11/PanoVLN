import json
import tempfile
import unittest
from pathlib import Path
from src.memory.stream import build_episode_index, EpisodeSchedule


class StreamTests(unittest.TestCase):
    def fixture(self, path):
        rows = []
        for dataset in ('r2r', 'rxr'):
            for ep in range(4):
                for step in range(ep + 2):
                    rows.append(dict(dataset=dataset, episode_id=ep, step_index=step,
                        images=[f'{dataset}/{ep}/{i}.jpg' for i in range(step + 1)],
                        history_actions=['forward'] * step, instruction=f'go {ep}',
                        action_sequence=['stop'] if step == ep + 1 else ['forward']))
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def test_sharding_causality_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            self.fixture(path)
            index = build_episode_index(path)
            schedules = [EpisodeSchedule(index, r, 3) for r in range(3)]
            self.assertEqual(len(set(len(s.items) for s in schedules)), 1)
            owners = {}
            for rank, schedule in enumerate(schedules):
                last = {}
                for item in schedule:
                    ep = item['episode_uid']
                    self.assertEqual(owners.setdefault(ep, rank), rank)
                    self.assertGreaterEqual(item['step'], last.get(ep, -1))
                    last[ep] = item['step']
                    if item['task'] == 'summary':
                        self.assertFalse(item['episode_start'])
                        self.assertTrue(item['episode_end'])
            a = schedules[0]
            it = iter(a)
            first = next(it)
            next(it)  # prefetch is not consumption
            self.assertEqual(a.consumed, 0)
            a.consume(first)
            b = EpisodeSchedule(index, 0, 3)
            b.load_state_dict(a.state_dict())
            self.assertEqual(list(a), list(b))
            with self.assertRaises(ValueError):
                EpisodeSchedule(index, 0, 4).load_state_dict(a.state_dict())

    def test_subsummaries_and_budget(self):
        from src.memory.stream import select_episode_subset, training_budget
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            self.fixture(path)
            labels = Path(directory) / 'sub.jsonl'
            annotation = dict(dataset='r2r', episode_id=3, step_index=1,
                text='Walk to the doorway.', alignment_verified=True)
            labels.write_text(json.dumps(annotation) + '\n')
            index = build_episode_index(path, sub_summaries_path=labels)
            rows = index['episodes']['r2r::3']
            self.assertEqual([(r['step'], r['task']) for r in rows if r['step'] == 1],
                [(1, 'navigation'), (1, 'summary')])
            self.assertEqual(rows[2]['summary_text'], annotation['text'])
            subset = select_episode_subset(index, 4, 42)
            self.assertEqual(list(subset['episodes']), list(select_episode_subset(index, 4, 42)['episodes']))
            expected = sum(len(EpisodeSchedule(subset, 0, 2, 42, e)) // 2 for e in range(3))
            self.assertEqual(training_budget(subset, 2, 42, 3, 2), expected)
            with self.assertRaises(ValueError):
                training_budget(subset, 2, 42, 3, 2, expected + 1)
            annotation['alignment_verified'] = False
            labels.write_text(json.dumps(annotation))
            with self.assertRaises(ValueError):
                build_episode_index(path, sub_summaries_path=labels)

    def test_rejects_broken_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            self.fixture(path)
            rows = path.read_text().splitlines()
            row = json.loads(rows[1]); row['history_actions'] = []
            rows[1] = json.dumps(row)
            path.write_text('\n'.join(rows))
            with self.assertRaises(ValueError):
                build_episode_index(path)


if __name__ == '__main__':
    unittest.main()
