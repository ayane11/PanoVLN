import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file
from panovln_dapo.checkpoint_audit import audit


class ExportAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.old=Path(self.tmp.name)/'old';self.old.mkdir()
        self.new=Path(self.tmp.name)/'new';self.new.mkdir()
        save_file({'panovggt.encoder':torch.ones(2),'model.weight':torch.zeros(2)},self.old/'model.safetensors')
        (self.new/'data.pt').write_bytes(b'data')
        (self.new/'optimizer.pt').write_bytes(b'optimizer')
        (self.new/'vln_progress.json').write_text(json.dumps({'data_sha256':hashlib.sha256(b'data').hexdigest(),
            'policy_version':1,'optimizer_steps':1,'changed_parameter_samples':1}))

    def write(self,geometry=1.,policy=1.):
        save_file({'panovggt.encoder':torch.full((2,),geometry),'model.weight':torch.full((2,),policy)},self.new/'model.safetensors')

    def test_changed_policy_with_frozen_geometry_passes(self):
        self.write();report=audit(self.old,self.new)
        self.assertEqual(report['counts']['changed_tensors'],1)
        self.assertTrue(report['frozen_geometry_unchanged'])

    def test_periodic_checkpoint_requires_explicit_interval_and_binding(self):
        self.write()
        path=self.new/'vln_progress.json';progress=json.loads(path.read_text())
        progress.update(policy_version=5,previous_durable_policy_version=0)
        path.write_text(json.dumps(progress))
        with self.assertRaisesRegex(ValueError,'interval'):audit(self.old,self.new)
        self.assertEqual(audit(self.old,self.new,expected_updates=5)['policy_version'],5)
        progress['previous_durable_policy_version']=1;path.write_text(json.dumps(progress))
        with self.assertRaisesRegex(ValueError,'bound'):audit(self.old,self.new,expected_updates=5)

    def test_corruption_and_fake_update_rejected(self):
        self.write(geometry=2.)
        with self.assertRaisesRegex(ValueError,'Frozen geometry'):audit(self.old,self.new)
        self.write(policy=0.)
        with self.assertRaisesRegex(ValueError,'did not change'):audit(self.old,self.new)
        self.write(policy=float('nan'))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):audit(self.old,self.new)
        self.write();(self.new/'data.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):audit(self.old,self.new)

    def test_legacy_prefix_and_tied_head_alias(self):
        for p in (self.old,self.new):
            (p/'config.json').write_text(json.dumps({'tie_word_embeddings':True}))
        save_file({'panovggt.encoder':torch.ones(2),
                   'model.language_model.language_model.language_model.embed_tokens.weight':torch.zeros(2),
                   'model.language_model.visual.weight':torch.zeros(2),
                   'lm_head.weight':torch.zeros(2)},self.old/'model.safetensors')
        save_file({'panovggt.encoder':torch.ones(2),
                   'model.language_model.embed_tokens.weight':torch.ones(2),
                   'model.visual.weight':torch.ones(2)},self.new/'model.safetensors')
        self.assertTrue(audit(self.old,self.new)['verified_tied_head_alias'])
        save_file({'panovggt.encoder':torch.ones(2),
                   'model.language_model.embed_tokens.weight':torch.ones(2),
                   'model.visual.weight':torch.ones(2),
                   'lm_head.weight':torch.zeros(2)},self.new/'model.safetensors')
        with self.assertRaisesRegex(ValueError,'disagree'):audit(self.old,self.new)

    def test_original_dtype_cast_effect_is_reported(self):
        save_file({'panovggt.encoder':torch.ones(2),'model.weight':torch.ones(2,dtype=torch.bfloat16)},self.old/'model.safetensors')
        save_file({'panovggt.encoder':torch.ones(2),'model.weight':torch.ones(2)+1e-5},self.new/'model.safetensors')
        report=audit(self.old,self.new)
        self.assertEqual(report['counts']['changed_tensors'],1)
        self.assertEqual(report['counts'].get('changed_tensors_after_original_dtype_cast',0),0)

    def test_second_update_audits_previous_policy_not_fixed_baseline(self):
        from panovln_dapo.after_update_eval import update_initial_checkpoint
        run=Path(self.tmp.name)/'actor_run';run.mkdir()
        self.write()
        self.new.rename(run/'checkpoint');self.new=run/'checkpoint'
        (self.old/'vln_progress.json').write_text(json.dumps({'policy_version':1}))
        progress=json.loads((self.new/'vln_progress.json').read_text());progress['policy_version']=2
        (self.new/'vln_progress.json').write_text(json.dumps(progress))
        command=['python','-m','panovln_dapo.actor','--checkpoint',str(self.old),'--output',str(self.new)]
        (run/'run_manifest.json').write_text(json.dumps({'command':command}))
        initial=update_initial_checkpoint(run)
        self.assertEqual(initial,self.old)
        self.assertEqual(audit(initial,self.new)['policy_version'],2)
        command[-1]=str(run/'different')
        (run/'run_manifest.json').write_text(json.dumps({'command':command}))
        with self.assertRaisesRegex(ValueError,'export path'):
            update_initial_checkpoint(run)


class ChunkedAuditTests(unittest.TestCase):
    def test_late_chunk_corruption_is_not_skipped(self):
        from panovln_dapo.checkpoint_audit import compare_tensor_chunks
        old=torch.ones(17);new=old.clone();new[-1]=2
        self.assertEqual(compare_tensor_chunks(old,new,'weight',chunk_elements=4),(True,True))
        new[-1]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            compare_tensor_chunks(old,new,'weight',chunk_elements=4)

    def test_original_dtype_rounding_matches_full_comparison(self):
        from panovln_dapo.checkpoint_audit import compare_tensor_chunks
        old=torch.ones(17,dtype=torch.bfloat16);new=old.float()+1e-5
        self.assertEqual(compare_tensor_chunks(old,new,'weight',chunk_elements=4),(True,False))
