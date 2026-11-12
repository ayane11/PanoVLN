"""Controller safety gates with fake processes; no GPU training claims."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from panovln_dapo import run_official_experiment as driver


class OfficialExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.commands=[];self.model_fails=False;self.free_gib=122
        for name in ('data','assets_audit','render','initial'):(self.root/name).mkdir()
        (self.root/'data/train.parquet').write_bytes(b'fixture')
        (self.root/'assets_audit/complete.json').write_text(json.dumps(dict(passed=True,episodes=6901,scenes=725)))
        (self.root/'render/complete.json').write_text(json.dumps(dict(passed=True,scenes=725)))

    def spawn(self,command,**kwargs):
        self.commands.append((command,kwargs['env'].get('CUDA_VISIBLE_DEVICES')))
        module=command[command.index('-m')+1]
        code=0
        if module=='panovln_dapo.official_model_gate':
            output=Path(command[command.index('--output')+1]);output.mkdir()
            (output/'complete.json').write_text(json.dumps(dict(passed=not self.model_fails)))
            code=int(self.model_fails)
        if module=='panovln_dapo.collect':
            output=Path(command[command.index('--output')+1]);output.mkdir()
            (output/'complete.json').write_text(json.dumps(dict(valid_groups=8,partial_batch=False)))
        class Child:
            pid=12345678
            def poll(self):return None
            def wait(self,**kw):return code
        return Child()

    def run_driver(self, extra=()):
        argv=['driver','--root',str(self.root),'--render-pid','1234','--render-audit',str(self.root/'render'),
              '--checkpoint',str(self.root/'initial'),'--baseline',str(self.root/'baseline'),
              '--eval-config',str(self.root/'eval.yaml')]
        argv.extend(extra)
        class Response:
            status=200
            def __enter__(self):return self
            def __exit__(self,*args):pass
        with patch('sys.argv',argv), patch('panovln_dapo.first_update.identity',return_value=None), \
             patch.object(driver.subprocess,'check_output',return_value='1, 20\n3, 20\n5, 2300\n'), \
             patch.object(driver.subprocess,'Popen',side_effect=self.spawn), \
             patch.object(driver.shutil,'disk_usage',return_value=SimpleNamespace(free=self.free_gib*1024**3)), \
             patch.object(driver.time,'sleep'),patch.object(driver.os,'killpg') as kill, \
             patch.object(driver.socket,'socket') as sock, \
             patch.object(driver.urllib.request,'build_opener') as opener, \
             patch('panovln_dapo.compare.read_run',return_value=({'manifest':'eval.parquet','max_episodes':64},[])), \
             patch('panovln_dapo.compare.compare',return_value={'mock_only':True}):
            sock.return_value.__enter__.return_value.connect_ex.return_value=1
            opener.return_value.open.return_value=Response()
            driver.main()
            self.assertTrue(all(call.args[0]==12345678 for call in kill.call_args_list))

    def test_failed_scene_gate_never_loads_model_or_trains(self):
        (self.root/'render/complete.json').write_text(json.dumps(dict(passed=False,scenes=725)))
        with self.assertRaisesRegex(RuntimeError,'render gate'):self.run_driver()
        self.assertFalse(self.commands)

    def test_failed_model_gate_never_collects_or_updates(self):
        self.model_fails=True
        with self.assertRaisesRegex(RuntimeError,'model_gate failed'):self.run_driver()
        self.assertEqual(len(self.commands),1)

    def test_three_gpu_update_after_gates_and_full_batch(self):
        self.run_driver()
        modules=[c[c.index('-m')+1] for c,_ in self.commands]
        self.assertLess(modules.index('panovln_dapo.official_model_gate'),modules.index('panovln_dapo.collect'))
        actor=next(c for c,_ in self.commands if 'panovln_dapo.run_actor' in c)
        self.assertEqual(actor[actor.index('--gpus')+1],'1,3,5')
        self.assertIn('--shared-gpus',actor)
        self.assertNotIn('--resume-optimizer',actor)
        self.assertEqual(json.loads((self.root/'experiment_01/status.json').read_text())['stage'],'comparison_complete')

    def test_two_gpu_budget_is_forwarded_to_actor(self):
        self.run_driver(['--actor-gpus','1,3','--memory-limit-gib','44','--run-name','experiment_02'])
        actor=next(c for c,_ in self.commands if 'panovln_dapo.run_actor' in c)
        self.assertEqual(actor[actor.index('--gpus')+1],'1,3')
        self.assertEqual(float(actor[actor.index('--memory-limit-gib')+1]),44.)
        self.assertEqual(json.loads((self.root/'experiment_02/status.json').read_text())['stage'],'comparison_complete')

    def test_low_disk_prevents_collection(self):
        self.free_gib=79
        with self.assertRaisesRegex(RuntimeError,'export reserve'):self.run_driver()
        self.assertFalse(any('panovln_dapo.collect' in c for c,_ in self.commands))

    def test_verified_gate_is_reused_and_collection_reserve_forwarded(self):
        gate=self.root/'previous/model_gate';gate.mkdir(parents=True)
        previous=dict(arguments={'checkpoint':str(self.root/'initial')},
                      manifest_sha256=hashlib.sha256(b'fixture').hexdigest(),
                      source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(driver.__file__).parent.glob('*.py')})
        (gate.parent/'config.json').write_text(json.dumps(previous))
        (gate/'complete.json').write_text(json.dumps(dict(passed=True,reports=[dict(category=c,max_error=1e-5,mean_error=1e-6) for c in ['r2r','rxr','mp3d_other','hm3d']])))
        self.run_driver(['--verified-model-gate',str(gate)])
        self.assertFalse(any('panovln_dapo.official_model_gate' in c for c,_ in self.commands))
        collect=next(c for c,_ in self.commands if 'panovln_dapo.collect' in c)
        self.assertEqual(collect[collect.index('--min-free-gib')+1],'70')

    def test_collection_adoption_checks_policy_data_and_sampling(self):
        previous=self.root/'previous';collection=previous/'collection';collection.mkdir(parents=True)
        hashes={name:'unchanged' for name in ('collect.py','backend.py','inputs.py','episode.py','sampling.py','data.py','simplevln.py')}
        (previous/'config.json').write_text(json.dumps(dict(source_sha256=hashes,manifest_sha256=hashlib.sha256(b'fixture').hexdigest())))
        config=dict(checkpoint=str(self.root/'initial'),manifest=str(self.root/'data/train.parquet'),
                    epochs=20,seed=42,rollout_n=8,target_groups=8,generation_groups=1,max_generation_batches=0,
                    max_response_tokens=64,max_env_steps=500,max_decisions=500,temperature=1.,progress_coef=0.,min_free_gib=70.)
        (collection/'config.json').write_text(json.dumps(config))
        self.assertEqual(driver.validate_adopted_collection(previous,self.root,self.root/'initial',hashes),collection)
        for field,value in [('checkpoint',str(self.root/'different')),('rollout_n',4),('manifest',str(self.root/'different.parquet'))]:
            changed={**config,field:value};(collection/'config.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):driver.validate_adopted_collection(previous,self.root,self.root/'initial',hashes)
        (collection/'config.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,'implementation'):
            driver.validate_adopted_collection(previous,self.root,self.root/'initial',{**hashes,'collect.py':'changed'})
