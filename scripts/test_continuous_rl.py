"""CPU-only state machine and uneven-rank regression tests; no GPU jobs launched."""
import ast
import copy
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import io
import contextlib
from unittest.mock import patch, MagicMock

import continuous_rl as c


PROJECT = Path(__file__).resolve().parents[1]


class FakeDriver(c.Driver):
    """Exercise the real persistence/recovery loop with tiny stage artifacts."""
    fail_mode = None
    commands = []

    def capacity(self, *args, **kwargs):
        pass

    def service(self, directory):
        return None

    def stop_owned(self, child):
        pass

    def launch(self, command, log, gpu='', result=None):
        self.commands.append(command)
        module = command[1]
        get = lambda key: command[command.index(key) + 1]
        output = Path(get('--output'))
        log.write_text('fake CPU stage\n')
        if module == 'panovln_dapo.collect':
            output.mkdir()
            previous = Path(get('--checkpoint'))
            version = c.read(previous / 'vln_progress.json')['policy_version'] if (previous / 'vln_progress.json').exists() else 0
            c.atomic(output / 'selected_batch.json', {'checkpoint': str(previous), 'policy_version': version, 'cursor': {'offset': version + 1}})
            c.atomic(output / 'complete.json', {'valid_groups': 8, 'data_exhausted': False})
        elif module == 'panovln_dapo.run_actor':
            output.mkdir()
            if self.fail_mode == 'before_export':
                raise OSError('simulated write failure')
            checkpoint = output / 'checkpoint'
            checkpoint.mkdir()
            batch = Path(get('--batch'))
            payload = c.read(batch)
            (checkpoint / 'data.pt').write_bytes(b'fake cursor')
            (checkpoint / 'optimizer.pt').write_bytes(b'fake optimizer')
            c.atomic(checkpoint / 'vln_progress.json', {
                'policy_version': payload['policy_version'] + 1,
                'source_batch_sha256': c.digest(batch),
                'data_sha256': c.digest(checkpoint / 'data.pt'), 'optimizer_steps': 2,
                'lr': self.a.lr, 'ppo_epochs': 1, 'mini_batch_decisions': 128,
                'clip_low': .2, 'clip_high': .28, 'dual_clip': 10., 'entropy_coeff': 0.,
                'world_size': len(self.a.gpus)})
            if self.fail_mode == 'after_export':
                raise RuntimeError('simulated death before wrapper status')
        elif module == 'panovln_dapo.checkpoint_audit':
            c.atomic(output, {'fake_cpu_audit': True})
        else:
            raise AssertionError(module)


class ContinuousTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        initial = self.root / 'initial'
        initial.mkdir()
        (self.root / 'manifest').write_text('manifest')
        (self.root / 'environment').write_text('environment')
        c.atomic(self.root / 'assets', {'passed': True, 'episodes': 6901, 'scenes': 725})
        c.atomic(self.root / 'render', {'passed': True, 'scenes': 725})
        self.a = Namespace(project=PROJECT, output=self.root / 'run', checkpoint=initial,
                           manifest=self.root / 'manifest', environment=self.root / 'environment',
                           assets_audit=self.root / 'assets', render_audit=self.root / 'render',
                           gpus=[1, 3], port=8037, memory_limit_gib=44., max_updates=2,
                           epochs=20, seed=42, rollout_n=8, target_groups=8,
                           mini_batch_decisions=128, lr=5e-7, ppo_epochs=1,
                           keep_last=2, keep_every=0, keep_rollouts=False, wandb_project='test')
        self.wandb = MagicMock()
        self.mock = patch.dict('sys.modules', {'wandb': self.wandb})
        self.mock.start()
        FakeDriver.commands = []

    def tearDown(self):
        self.mock.stop()
        self.tmp.cleanup()

    def run_driver(self, fail=None):
        d = FakeDriver(self.a)
        d.fail_mode = fail
        try:
            d.train()
        finally:
            d.close()
        return d

    def test_resident_branch_launches_once_and_mirrors_durable_state(self):
        self.a.resident=True;self.a.save_freq=5;self.a.param_offload=True;self.a.env_pool_size=1
        self.a.memory_limits_gib=[36.,26.5]
        launches=[]
        class ResidentFake(FakeDriver):
            def launch(inner,command,log,gpu='',result=None):
                launches.append((command,gpu))
                config=c.read(Path(command[command.index('--config')+1]))
                checkpoint=Path(config['output'])/'checkpoint_v2';checkpoint.mkdir()
                c.atomic(Path(config['output'])/'complete.json',dict(policy_version=2,optimizer_steps=24,
                    checkpoint=str(checkpoint),data_exhausted=False))
        driver=ResidentFake(self.a)
        try:driver.train()
        finally:driver.close()
        self.assertEqual(len(launches),1)
        self.assertIn('panovln_dapo.resident_train',launches[0][0])
        self.assertEqual(launches[0][1],'1,3')
        self.assertEqual(driver.state['completed_updates'],2)
        self.assertEqual(driver.state['optimizer_steps'],24)
        config=c.read(self.a.output/'resident_launch.json')
        self.assertEqual(config['save_freq'],5)
        self.assertEqual(config['memory_limit_gib'],[36.,26.5])
        self.assertTrue(config['param_offload'])
        self.assertEqual(config['actor']['mini_batch_decisions'],128)
        self.assertEqual(c.read(self.a.output/'latest.json')['completed_updates'],2)
        self.wandb.init.assert_not_called()

    def test_capacity_uses_gpu_order_and_individual_budgets(self):
        self.a.gpus=[4,3];self.a.memory_limits_gib=[36.,26.5]
        driver=c.Driver(self.a)
        try:
            with patch.object(c.subprocess,'check_output',return_value='3, 18000, 30208\n4, 0, 39936\n'):
                driver.capacity([3,4])
            with patch.object(c.subprocess,'check_output',return_value='3, 18000, 30207\n4, 0, 39936\n'):
                with self.assertRaisesRegex(RuntimeError,'GPU 3'):
                    driver.capacity([4,3])
        finally:driver.close()

    def test_multi_round_resume_reshard_and_retention(self):
        first = self.run_driver()
        self.assertEqual(first.state['completed_updates'], 2)
        self.a.gpus = [1, 3, 5]
        self.a.memory_limit_gib = 40.
        self.a.max_updates = 3
        second = self.run_driver()
        self.assertEqual(second.state['completed_updates'], 3)
        self.assertEqual(second.state['optimizer_steps'], 6)
        root = self.a.output
        self.assertFalse((root / 'round_00001/update_001/checkpoint').exists())
        self.assertTrue((root / 'round_00002/update_001/checkpoint/optimizer.pt').exists())
        self.assertTrue((root / 'round_00003/update_001/checkpoint/optimizer.pt').exists())
        last = [x for x in FakeDriver.commands if x[1] == 'panovln_dapo.run_actor'][-1]
        self.assertIn('--resume-optimizer', last)
        self.assertIn('--allow-world-size-change', last)
        self.assertEqual(last[last.index('--gpus')+1], '1,3,5')
        self.assertEqual(c.read(root / 'latest.json')['completed_updates'], 3)
        self.assertEqual(self.wandb.init.call_args.kwargs['mode'], 'offline')

    def test_recover_export_without_wrapper_status(self):
        with self.assertRaises(RuntimeError):
            self.run_driver('after_export')
        before = len([x for x in FakeDriver.commands if x[1] == 'panovln_dapo.run_actor'])
        self.a.max_updates = 1
        recovered = self.run_driver()
        self.assertEqual(recovered.state['completed_updates'], 1)
        after = len([x for x in FakeDriver.commands if x[1] == 'panovln_dapo.run_actor'])
        self.assertEqual(before, after)

    def test_failed_update_reuses_completed_collection(self):
        with self.assertRaises(OSError):
            self.run_driver('before_export')
        self.assertEqual(c.read(self.a.output / 'state.json')['completed_updates'], 0)
        self.a.max_updates = 1
        recovered = self.run_driver()
        self.assertEqual(recovered.state['completed_updates'], 1)
        self.assertEqual(len([x for x in FakeDriver.commands if x[1] == 'panovln_dapo.collect']), 1)
        self.assertTrue((self.a.output / 'round_00001/update_002/checkpoint').exists())

    def test_lock_and_science_changes_rejected(self):
        driver = FakeDriver(self.a)
        try:
            with self.assertRaises(BlockingIOError):
                FakeDriver(self.a)
        finally:
            driver.close()
        self.a.max_updates = 1
        self.run_driver()
        self.a.seed = 8
        with self.assertRaisesRegex(ValueError, 'Scientific configuration'):
            self.run_driver()

    def test_live_owned_process_blocks_restart(self):
        self.a.output.mkdir()
        c.atomic(self.a.output / 'process_stage.json', {'pid': 123, 'identity': ['start', 'command']})
        with patch.object(c, 'process_identity', return_value=['start', 'command']):
            with self.assertRaisesRegex(RuntimeError, 'still alive'):
                FakeDriver(self.a)
        c.atomic(self.a.output / 'process_stage.json', {'pid': 123, 'identity': None})
        with patch.object(c, 'process_identity', return_value=None):
            driver = FakeDriver(self.a)
            driver.close()

    def test_runtime_isolation_and_no_capacity_threshold(self):
        original = PROJECT / 'rl/panovln_dapo/actor.py'
        before = c.digest(original)
        hashes = c.make_runtime(PROJECT, self.root / 'runtime')
        text = (self.root / 'runtime/panovln_dapo/actor.py').read_text()
        self.assertEqual(before, c.digest(original))
        self.assertNotIn('shutil.disk_usage', text)
        self.assertNotIn('mini_batch_decisions%world', text)
        tree = ast.parse(Path(c.__file__).read_text())
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                             and node.func.attr == 'disk_usage' for node in ast.walk(tree)))
        for path in (self.root / 'runtime/panovln_dapo').glob('*.py'):
            compile(path.read_text(), str(path), 'exec')
        self.assertIn('actor.py', hashes)

    def test_foreground_logs_are_forwarded_without_repeats(self):
        path = self.root / 'stage.log'
        offsets = {}
        terminal = io.StringIO()
        with contextlib.redirect_stdout(terminal):
            c.forward_log(path, offsets)
            path.write_text('采样开始\n')
            c.forward_log(path, offsets)
            c.forward_log(path, offsets)
            with path.open('a') as stream:
                stream.write('optimizer_step=1\n')
            c.forward_log(path, offsets)
        self.assertEqual(terminal.getvalue(), '采样开始\noptimizer_step=1\n')

    def test_stop_only_owned_process_and_record_identity(self):
        driver = c.Driver(self.a)
        foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        owned = None
        try:
            driver.setup()
            owned = driver.spawn(['-c', 'import time; time.sleep(60)'], self.root / 'owned.log', '', 'stage')
            record = c.read(self.a.output / 'process_stage.json')
            self.assertEqual(record['pid'], owned.pid)
            self.assertIsNotNone(record['identity'])
            driver.stop_owned(owned)
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(foreign.poll())
        finally:
            driver.close()
            foreign.terminate()
            foreign.wait(timeout=10)

    def test_export_checksum_mismatch_is_not_adopted(self):
        with self.assertRaises(RuntimeError):
            self.run_driver('after_export')
        exported = self.a.output / 'round_00001/update_001/checkpoint'
        (exported / 'data.pt').write_bytes(b'corrupted')
        self.a.max_updates = 1
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.run_driver()
        self.assertEqual(c.read(self.a.output / 'state.json')['completed_updates'], 0)

    def test_actual_patched_assignments_cover_every_rank_minibatch(self):
        text = c.patched_actor((PROJECT / 'rl/panovln_dapo/actor.py').read_text())
        block = text.split('        assigned=[]', 1)[1].split('        old_values=[]', 1)[0]
        code = 'assigned=[]\n' + '\n'.join(line[8:] for line in block.splitlines() if line.strip())
        for world in (2, 3, 4):
            for count in (1, 127, 128, 129, 256, 1762):
                all_real = []
                totals = []
                for rank in range(world):
                    scope = dict(rows=[None]*count, args=Namespace(mini_batch_decisions=128), world=world, rank=rank)
                    exec(code, scope)
                    assigned = scope['assigned']
                    self.assertEqual(assigned, c.assignments(count, 128, world, rank))
                    all_real.extend(i for i, real in assigned if real)
                    totals.append(len(assigned))
                    old_values = {i for i, _ in assigned}
                    # Every training row, including zero-weight padding, has a replayed old probability.
                    for start in range(0, count, 128):
                        end = min(start+128, count)
                        chunk = list(range(start, end))
                        chunk += [0] * ((-len(chunk)) % world)
                        self.assertTrue(set(chunk[rank::world]).issubset(old_values))
                self.assertEqual(sorted(all_real), list(range(count)))
                self.assertEqual(len(set(totals)), 1)

    def test_uneven_rank_padding_preserves_token_mean_gradient(self):
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            self.skipTest('Run in the container training environment for torch CPU validation')
        import torch
        spec = importlib.util.spec_from_file_location('tested_algorithms', PROJECT / 'rl/panovln_dapo/algorithms.py')
        algorithms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(algorithms)
        torch.manual_seed(7)
        count = 257
        features = torch.randn(count, 3, dtype=torch.float64)
        advantages = torch.randn(count, 3, dtype=torch.float64)
        masks = (torch.rand(count, 3) > .25).to(torch.float64)
        masks[:, 0] = 1
        old = torch.zeros(count, 3, dtype=torch.float64)
        for start in range(0, count, 128):
            end = min(count, start+128)
            denominator = masks[start:end].sum()
            parameter = torch.tensor(.13, dtype=torch.float64, requires_grad=True)
            numerator, _ = algorithms.policy_loss_sum(old[start:end], features[start:end]*parameter,
                                                     advantages[start:end], masks[start:end], .2, .28, 10.)
            (numerator/denominator).backward()
            reference = parameter.grad.item()
            for world in (2, 3, 4):
                gradients = []
                chunk = list(range(start, end))
                chunk += [0]*((-len(chunk)) % world)
                for rank in range(world):
                    parameter = torch.tensor(.13, dtype=torch.float64, requires_grad=True)
                    for local in range(rank, len(chunk), world):
                        index = chunk[local]
                        numerator, _ = algorithms.policy_loss_sum(old[index], features[index]*parameter,
                                                                 advantages[index], masks[index]*int(local<end-start),
                                                                 .2, .28, 10.)
                        (numerator*world/denominator).backward()
                    gradients.append(parameter.grad.item())
                self.assertAlmostEqual(sum(gradients)/world, reference, places=12)


if __name__ == '__main__':
    unittest.main()


class AlignedDefaultsTests(unittest.TestCase):
    def arguments(self, *extra):
        values=['continuous_rl.py']
        for key in ('project','output','checkpoint','manifest','environment','assets-audit','render-audit'):
            values += ['--'+key, '/tmp/unused-'+key]
        return values+['--gpus','1,3']+list(extra)

    def test_aligned_defaults_and_generation_batch_forwarding(self):
        with patch.object(sys,'argv',self.arguments()):
            args=c.parse_args()
        self.assertEqual((args.rollout_n,args.target_groups,args.generation_groups,args.mini_batch_decisions,args.epochs,args.max_updates),(8,128,192,128*8,20,0))
        driver=object.__new__(c.Driver);driver.a=args;driver.server='http://unused'
        command=driver.collect(Path('/tmp/model'),Path('/tmp/collection'))
        self.assertEqual(command[command.index('--generation-groups')+1],'192')

    def test_zero_generation_batch_rejected(self):
        with patch.object(sys,'argv',self.arguments('--generation-groups','0')),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):c.parse_args()
