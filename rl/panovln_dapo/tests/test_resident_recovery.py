import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from panovln_dapo.resident_train import recover_export


class ResidentRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.initial=self.root/'initial';self.initial.mkdir()
        self.config={'checkpoint':str(self.initial)}
        self.source=self.root/'batch.json';self.source.write_bytes(b'batch')
        self.candidate=self.root/'checkpoint_v5';self.candidate.mkdir()
        self.progress=dict(policy_version=5,previous_durable_policy_version=0,
            resident_source_batch=str(self.source),source_batch_sha256=hashlib.sha256(b'batch').hexdigest(),
            cumulative_optimizer_steps=50)
        self.write()
    def write(self):
        (self.candidate/'vln_progress.json').write_text(json.dumps(self.progress))
    def test_complete_export_adopted_only_after_audit(self):
        with patch('panovln_dapo.resident_train.audit',return_value={'verified':True}) as audit:
            recover_export(self.config,self.root)
            audit.assert_called_once_with(self.initial,self.candidate,expected_updates=5)
        self.assertEqual(json.loads((self.root/'latest.json').read_text())['policy_version'],5)
    def test_audit_failure_does_not_publish(self):
        with patch('panovln_dapo.resident_train.audit',side_effect=ValueError('bad geometry')):
            with self.assertRaisesRegex(ValueError,'geometry'):recover_export(self.config,self.root)
        self.assertFalse((self.root/'latest.json').exists())
    def test_changed_batch_and_wrong_binding_rejected(self):
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'batch changed'):recover_export(self.config,self.root)
        self.progress['previous_durable_policy_version']=2;self.write()
        with self.assertRaisesRegex(ValueError,'previous'):recover_export(self.config,self.root)
    def test_incomplete_directory_is_never_adopted(self):
        self.candidate.rename(self.root/'checkpoint_v5.incomplete')
        with patch('panovln_dapo.resident_train.audit') as audit:
            recover_export(self.config,self.root)
            audit.assert_not_called()
        self.assertFalse((self.root/'latest.json').exists())
