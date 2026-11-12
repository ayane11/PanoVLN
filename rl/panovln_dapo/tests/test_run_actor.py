"""Verify launcher configuration reaches the actual torchrun command, CPU only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from panovln_dapo import run_actor


class ActorLauncherTests(unittest.TestCase):
    def test_nondefault_clips_reach_actor_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch=root/'batch.json';batch.write_text('{}')
            output=root/'update';commands=[]
            def spawn(command,**kwargs):
                commands.append(command)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'3,6')
                exported=Path(command[command.index('--output')+1]);exported.mkdir()
                (exported/'vln_progress.json').write_text('{}')
                class Child:
                    pid=123456
                    def wait(self):return 0
                return Child()
            argv=['run_actor','--batch',str(batch),'--checkpoint','initial','--output',str(output),
                  '--gpus','3,6','--clip-low','0.1','--clip-high','0.3','--dual-clip','5','--entropy-coeff','0.03']
            with patch('sys.argv',argv), \
                 patch('panovln_dapo.actor.load_batch',return_value=({'trajectories':[{}]},[{}])), \
                 patch.object(run_actor.subprocess,'check_output',return_value='3, 20\n6, 20\n'), \
                 patch.object(run_actor.subprocess,'Popen',side_effect=spawn):
                run_actor.main()
            command=commands[0]
            for key,value in {'--clip-low':'0.1','--clip-high':'0.3','--dual-clip':'5.0','--entropy-coeff':'0.03'}.items():
                self.assertEqual(command[command.index(key)+1],value)
            self.assertEqual(json.loads((output/'run_manifest.json').read_text())['command'],command)
            self.assertEqual(json.loads((output/'status.json').read_text())['stage'],'update_exported')

    def test_shared_budget_gate_and_explicit_reshard_forwarding(self):
        for free,allowed in [(26000,False),(30000,True)]:
            with self.subTest(free=free),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);batch=root/'batch.json';batch.write_text('{}');output=root/'update'
                commands=[]
                def query(command,**kwargs):
                    value=free if '--query-gpu=index,memory.free' in command else 16000
                    return '\n'.join(f'{g}, {value}' for g in (0,2,3,4))
                def spawn(command,**kwargs):
                    commands.append(command)
                    self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'0,2,3,4')
                    path=Path(command[command.index('--output')+1]);path.mkdir()
                    (path/'vln_progress.json').write_text('{}')
                    class Child:
                        pid=123456
                        def wait(self):return 0
                    return Child()
                argv=['run_actor','--batch',str(batch),'--checkpoint','initial','--output',str(output),
                      '--gpus','0,2,3,4','--shared-gpus','--memory-limit-gib','24','--resume-optimizer','--allow-world-size-change']
                with patch('sys.argv',argv),patch('panovln_dapo.actor.load_batch',return_value=({'trajectories':[{}]},[{}])), \
                     patch.object(run_actor.subprocess,'check_output',side_effect=query),patch.object(run_actor.subprocess,'Popen',side_effect=spawn):
                    if allowed:run_actor.main()
                    else:
                        with self.assertRaisesRegex(RuntimeError,'headroom'):run_actor.main()
                if allowed:
                    self.assertIn('--allow-world-size-change',commands[0])
                    self.assertEqual(commands[0][commands[0].index('--memory-limit-gib')+1],'24.0')
                else:self.assertFalse(commands)
