import json
from pathlib import Path
import tempfile
import unittest
from panovln_dapo.resident_train import prune_committed,prune_consumed_rollouts


class RetentionTests(unittest.TestCase):
    def test_only_audited_committed_checkpoints_and_rollouts_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for version in (5,10,15,20):
                checkpoint=root/f'checkpoint_v{version}';checkpoint.mkdir()
                (checkpoint/'optimizer.pt').write_bytes(b'optimizer')
                (root/f'audit_v{version}.json').write_text('{}')
            current=root/'checkpoint_v20'
            (root/'latest.json').write_text(json.dumps(dict(checkpoint=str(current),policy_version=20)))
            incomplete=root/'checkpoint_v25.incomplete';incomplete.mkdir()
            unaudited=root/'checkpoint_v12';unaudited.mkdir()
            rollout=root/'collection_v19_123';rollout.mkdir()
            for name in ('complete.json','config.json','selected_batch.json','v19_0_0_r0.json'):(rollout/name).write_text('{}')
            unfinished=root/'collection_v20_456';unfinished.mkdir();(unfinished/'selected_batch.json').write_text('{}')
            prune_committed(root,current,keep_last=2,keep_every=10)
            self.assertFalse((root/'checkpoint_v5').exists())
            for name in ('checkpoint_v10','checkpoint_v15','checkpoint_v20','checkpoint_v12','checkpoint_v25.incomplete'):
                self.assertTrue((root/name).is_dir())
            self.assertTrue((rollout/'config.json').exists())
            self.assertFalse((rollout/'selected_batch.json').exists())
            self.assertFalse((rollout/'v19_0_0_r0.json').exists())
            self.assertTrue((unfinished/'selected_batch.json').exists())
    def test_latest_mismatch_rejects_pruning(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'latest.json').write_text(json.dumps(dict(checkpoint=str(root/'checkpoint_v5'),policy_version=5)))
            with self.assertRaisesRegex(ValueError,'publication'):prune_committed(root,root/'checkpoint_v10')

    def test_zero_retains_all_formal_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for version in (5,10,15):
                (root/f'checkpoint_v{version}').mkdir()
                (root/f'audit_v{version}.json').write_text('{}')
            current=root/'checkpoint_v15'
            (root/'latest.json').write_text(json.dumps(dict(checkpoint=str(current),policy_version=15)))
            prune_committed(root,current,keep_last=0,keep_rollouts=True)
            for version in (5,10,15):self.assertTrue((root/f'checkpoint_v{version}').is_dir())


class ConsumedCacheTests(unittest.TestCase):
    def test_pending_batch_and_unfinished_and_foreign_files_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for version in (0,1,2):
                folder=root/f'collection_v{version}_123';folder.mkdir()
                for name in ('selected_batch.json',f'v{version}_0_0_r0.json','config.json','complete.json'):
                    (folder/name).write_text('{}')
            current=root/'collection_v1_123/selected_batch.json'
            unfinished=root/'collection_v0_456';unfinished.mkdir();(unfinished/'selected_batch.json').write_text('{}')
            checkpoint=root/'checkpoint_v1';checkpoint.mkdir();(checkpoint/'optimizer.pt').write_bytes(b'optimizer')
            prune_consumed_rollouts(root,current)
            self.assertFalse((root/'collection_v0_123/selected_batch.json').exists())
            self.assertFalse((root/'collection_v1_123/v1_0_0_r0.json').exists())
            for path in (current,root/'collection_v1_123/config.json',root/'collection_v2_123/selected_batch.json',unfinished/'selected_batch.json',checkpoint/'optimizer.pt'):
                self.assertTrue(path.is_file())
    def test_foreign_pending_batch_rejected_before_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,'outside'):
                prune_consumed_rollouts(root,root/'other/selected_batch.json')
