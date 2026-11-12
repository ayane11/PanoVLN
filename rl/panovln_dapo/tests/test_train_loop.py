"""CPU controller tests with fake stage executables, not training evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from panovln_dapo import train_loop, actor, checkpoint_audit


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.initial=self.root/'initial';self.initial.mkdir()
        self.manifest=self.root/'manifest';self.manifest.write_text('manifest')
        self.output=self.root/'run';self.commands=[];self.fail_once=False;self.empty_data=False;self.fail_eval_once=False

    def spawn(self, command, **kwargs):
        self.commands.append(command)
        def arg(key):return command[command.index(key)+1]
        out=Path(arg('--output'));out.mkdir()
        module=command[command.index('-m')+1]
        code=0
        if module=='panovln_dapo.collect':
            (out/'complete.json').write_text(json.dumps({'valid_groups':0 if self.empty_data else 1,'data_exhausted':self.empty_data}))
            (out/'selected_batch.json').write_text(json.dumps({'checkpoint':arg('--checkpoint'),'cursor':{'epoch':1,'offset':0}}))
        elif module=='panovln_dapo.run_actor':
            if self.fail_once:
                self.fail_once=False;code=1
                (out/'status.json').write_text(json.dumps({'stage':'failed'}))
            else:
                (out/'checkpoint').mkdir()
                (out/'checkpoint/vln_progress.json').write_text('{}')
                (out/'status.json').write_text(json.dumps({'stage':'update_exported'}))
        elif module=='panovln_dapo.evaluate':
            if self.fail_eval_once:
                self.fail_eval_once=False;code=1
                (out/'failure.json').write_text('{}')
            else:
                self.write_evaluation(out,checkpoint=arg('--checkpoint'))
        else:raise AssertionError(module)
        class Child:
            pid=123456
            def wait(self):return code
        return Child()

    def run_driver(self, updates, resume=False, extra=()):
        argv=['train_loop','--checkpoint',str(self.initial),'--manifest',str(self.manifest),
              '--server','http://127.0.0.1:8016','--output',str(self.output),'--updates',str(updates)]
        argv.extend(extra)
        if resume:argv.append('--resume-driver')
        with patch('sys.argv',argv), patch.object(train_loop,'active_round_processes',return_value=[]), \
             patch.object(train_loop.subprocess,'check_output',return_value='3, 20\n6, 20\n'), \
             patch.object(train_loop.subprocess,'Popen',side_effect=self.spawn), \
             patch('panovln_dapo.actor.load_batch',side_effect=lambda path:(json.loads(path.read_text()),[{}])), \
             patch('panovln_dapo.checkpoint_audit.audit',return_value={'mock_only':True}):
            train_loop.main()

    def test_policy_handoff_across_two_rounds(self):
        self.run_driver(2)
        self.assertEqual([c[c.index('-m')+1] for c in self.commands],
                         ['panovln_dapo.collect','panovln_dapo.run_actor']*2)
        first_export=str(self.output/'round_0001/update_01/checkpoint')
        for command in self.commands[2:]:
            self.assertEqual(command[command.index('--checkpoint')+1],first_export)
        self.assertIn('--resume-optimizer',self.commands[-1])
        state=json.loads((self.output/'state.json').read_text())
        self.assertEqual(state['completed_updates'],2)
        self.assertEqual(state['stage'],'training_budget_complete')

    def test_retry_keeps_collection_and_preserves_failed_attempt(self):
        self.fail_once=True
        with self.assertRaisesRegex(RuntimeError,'status 1'):self.run_driver(1)
        self.run_driver(1,resume=True)
        self.assertEqual(len(self.commands),3)
        self.assertTrue((self.output/'round_0001/update_01/status.json').exists())
        self.assertTrue((self.output/'round_0001/update_02/checkpoint/vln_progress.json').exists())
        self.assertEqual(sum('panovln_dapo.collect' in c for c in self.commands),1)

    def test_exhausted_data_finishes_without_optimizer_or_recollection(self):
        self.empty_data=True
        self.run_driver(2)
        state=json.loads((self.output/'state.json').read_text())
        self.assertEqual(state['stage'],'data_exhausted')
        self.assertEqual(state['completed_updates'],0)
        self.assertEqual(state['terminal_cursor'],{'epoch':1,'offset':0})
        self.run_driver(2,resume=True)
        self.assertEqual(len(self.commands),1)

    def test_nondefault_protocol_reaches_stages_and_is_immutable_on_resume(self):
        options={'--temperature':'0.7','--max-response-tokens':'48',
                 '--max-env-steps':'300','--max-decisions':'80','--progress-coef':'0.2',
                 '--ppo-epochs':'2','--clip-low':'0.1','--clip-high':'0.3','--dual-clip':'5.0','--entropy-coeff':'0.03'}
        extra=[item for pair in options.items() for item in pair]
        self.run_driver(2,extra=extra)
        for command in self.commands:
            is_actor='panovln_dapo.run_actor' in command
            expected={k:v for k,v in options.items() if (k in ('--ppo-epochs','--clip-low','--clip-high','--dual-clip','--entropy-coeff'))==is_actor}
            for key,value in expected.items():
                self.assertEqual(command[command.index(key)+1],value)
        before=len(self.commands)
        self.run_driver(2,resume=True,extra=extra)
        self.assertEqual(len(self.commands),before)
        changed=list(extra);changed[changed.index('--progress-coef')+1]='0.3'
        with self.assertRaisesRegex(ValueError,'configuration changed'):
            self.run_driver(2,resume=True,extra=changed)
        self.assertEqual(len(self.commands),before)

    def write_evaluation(self, path, checkpoint='initial'):
        from panovln_dapo.compare import PROTOCOL
        config={k:'fixed' for k in PROTOCOL}
        config.update(episode_keys=['dev_a','dev_b'],checkpoint=checkpoint,max_episodes=2,
                      manifest_sha256=hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        (path/'config.json').write_text(json.dumps(config))
        (path/'complete.json').write_text(json.dumps({'episodes':2,'expected_episodes':2}))
        records=[dict(episode_key=k,success=1,spl=.9,distance_to_goal=.5,oracle_success=1)
                 for k in config['episode_keys']]
        (path/'results.jsonl').write_text('\n'.join(json.dumps(r) for r in records))

    def evaluation_options(self, every=1):
        baseline=self.root/'baseline';baseline.mkdir();self.write_evaluation(baseline)
        return ['--eval-baseline',str(baseline),'--eval-manifest',str(self.manifest),
                '--eval-server','http://127.0.0.1:8017','--eval-every',str(every)]

    def test_periodic_and_final_evaluations_use_exported_checkpoint(self):
        extra=self.evaluation_options(every=2)
        self.run_driver(3,extra=extra)
        evaluations=[c for c in self.commands if 'panovln_dapo.evaluate' in c]
        self.assertEqual(len(evaluations),2)
        for number,command in zip([2,3],evaluations):
            expected=self.output/f'round_{number:04d}'
            self.assertEqual(command[command.index('--checkpoint')+1],str(expected/'update_01/checkpoint'))
            self.assertEqual(json.loads((expected/'comparison.json').read_text())['episodes'],2)
        self.assertFalse((self.output/'round_0001/comparison.json').exists())

    def test_failed_evaluation_retries_without_repeating_update(self):
        extra=self.evaluation_options();self.fail_eval_once=True
        with self.assertRaisesRegex(RuntimeError,'status 1'):self.run_driver(1,extra=extra)
        self.run_driver(1,resume=True,extra=extra)
        self.assertEqual(sum('panovln_dapo.run_actor' in c for c in self.commands),1)
        self.assertEqual(sum('panovln_dapo.collect' in c for c in self.commands),1)
        self.assertTrue((self.output/'round_0001/evaluation_01/failure.json').exists())
        self.assertTrue((self.output/'round_0001/evaluation_02/complete.json').exists())
        self.assertEqual(json.loads((self.output/'state.json').read_text())['stage'],'training_budget_complete')
