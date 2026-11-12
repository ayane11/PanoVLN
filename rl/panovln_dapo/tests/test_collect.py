import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from panovln_dapo.collect import restore_cursor
from panovln_dapo.sampling import EpisodeCursor


class CursorContinuationTests(unittest.TestCase):
    def test_next_round_starts_after_checkpoint_cursor(self):
        rows = [{'id': i} for i in range(12)]
        cursor = EpisodeCursor(rows, seed=42, epochs=20)
        cursor.take(5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.save({'cursor': cursor.state_dict(), 'policy_version': 3}, path/'data.pt')
            digest = hashlib.sha256((path/'data.pt').read_bytes()).hexdigest()
            (path/'vln_progress.json').write_text(json.dumps({'data_sha256': digest, 'policy_version': 3}))
            restored = EpisodeCursor(rows, seed=42, epochs=20)
            self.assertEqual(restore_cursor(restored, path), 3)
            self.assertEqual(restored.take(4), cursor.take(4))
            (path/'data.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'): restore_cursor(restored, path)

    def test_partial_metadata_not_treated_as_initial_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cursor = EpisodeCursor([{'id': 1}])
            self.assertEqual(restore_cursor(cursor, path), 0)
            (path/'optimizer.pt').touch()
            with self.assertRaisesRegex(ValueError, 'Incomplete'): restore_cursor(cursor, path)


class ExhaustedCollectionTests(unittest.TestCase):
    def test_exhausted_checkpoint_does_not_load_gpu_policy(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch
        from panovln_dapo.collect import run
        rows=[{'id':0}]
        cursor=EpisodeCursor(rows,seed=42,epochs=1);cursor.take(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'checkpoint';checkpoint.mkdir()
            torch.save({'cursor':cursor.state_dict(),'policy_version':1},checkpoint/'data.pt')
            checksum=hashlib.sha256((checkpoint/'data.pt').read_bytes()).hexdigest()
            (checkpoint/'vln_progress.json').write_text(json.dumps({'data_sha256':checksum,'policy_version':1}))
            manifest=root/'manifest';manifest.write_text('manifest')
            args=SimpleNamespace(output=root/'collection',manifest=str(manifest),checkpoint=str(checkpoint),seed=42,epochs=1,
                                 rollout_n=4,target_groups=8,generation_groups=1,max_generation_batches=16)
            with patch('panovln_dapo.collect.pq.read_table') as table, patch('panovln_dapo.collect.NativePolicy.load') as load:
                table.return_value.to_pylist.return_value=rows
                asyncio.run(run(args))
                load.assert_not_called()
            metrics=json.loads((args.output/'complete.json').read_text())
            self.assertTrue(metrics['data_exhausted']);self.assertEqual(metrics['valid_groups'],0)

class ResidentCollectionTests(unittest.TestCase):
    def test_resident_exhausted_cursor_never_reloads_checkpoint(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch
        from panovln_dapo.collect import run
        cursor=EpisodeCursor([{'id':0}],epochs=1);cursor.take(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'manifest';manifest.write_text('manifest')
            args=SimpleNamespace(output=root/'collection',manifest=str(manifest),checkpoint='initial',seed=42,epochs=1,
                                 rollout_n=8,target_groups=128,generation_groups=192,max_generation_batches=0)
            with patch('panovln_dapo.collect.restore_cursor') as restore,patch('panovln_dapo.collect.NativePolicy.load') as load:
                asyncio.run(run(args,policy=object(),cursor=cursor,policy_version=5))
                restore.assert_not_called();load.assert_not_called()
            payload=json.loads((args.output/'selected_batch.json').read_text())
            self.assertEqual(payload['policy_version'],5)
            self.assertEqual(payload['cursor'],cursor.state_dict())
