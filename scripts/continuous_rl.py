"""Durable, offline-W&B multi-round orchestration; never manages foreign jobs."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def forward_log(path, offsets):
    """Mirror newly appended stage output to the foreground terminal once."""
    path = Path(path)
    if not path.exists():
        return
    with path.open(errors='replace') as stream:
        stream.seek(offsets.get(str(path), 0))
        text = stream.read()
        offsets[str(path)] = stream.tell()
    if text:
        print(text, end='', flush=True)


def process_identity(pid):
    try:
        path = Path('/proc') / str(pid)
        stat = (path / 'stat').read_text().split(') ', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return [stat[19], (path / 'cmdline').read_bytes().decode().replace('\0', ' ')]
    except (FileNotFoundError, ProcessLookupError):
        return None


def assignments(count, minibatch, world, rank):
    """Match actor minibatch padding, including when world does not divide batch."""
    result = []
    for start in range(0, count, minibatch):
        end = min(start + minibatch, count)
        positions = list(range(start, end))
        positions += [0] * ((-len(positions)) % world)
        result.extend((positions[i], i < end - start) for i in range(rank, len(positions), world))
    return result


def patched_actor(text):
    # Public actor now contains uneven-rank assignment and no storage gate.
    return text


def make_runtime(project, destination):
    """Use an isolated package: active experiment source files remain untouched."""
    package = destination / 'panovln_dapo'
    package.mkdir(parents=True)
    for source in (project / 'rl/panovln_dapo').glob('*.py'):
        text = source.read_text()
        if source.name == 'actor.py':
            text = patched_actor(text)
        elif source.name == 'simplevln.py':
            old = "Path(__file__).resolve().parents[3] / 'SimpleVLN' / 'RL'"
            if old not in text:
                raise ValueError('SimpleVLN source locator changed')
            text = text.replace(old, f"Path({str(project.parent / 'SimpleVLN/RL')!r})")
        elif source.name == 'run_actor.py':
            text = text.replace("Path('rl/panovln_dapo').glob('*.py')", "Path(__file__).parent.glob('*.py')")
        (package / source.name).write_text(text)
    return {p.name: digest(p) for p in package.glob('*.py')}


def complete_collection(round_dir):
    found = [p for p in sorted(round_dir.glob('collection_*')) if p.is_dir()
             and (p / 'complete.json').is_file() and not (p / 'failure.json').exists()]
    if len(found) > 1:
        raise ValueError('Ambiguous completed collection')
    return found[0] if found else None


def exported_update(round_dir):
    # A checkpoint directory is atomically renamed only after every file is written.
    # Recovery can adopt it even if the wrapper died before writing update_exported.
    found = [p for p in sorted(round_dir.glob('update_*')) if p.is_dir()
             and (p / 'checkpoint/vln_progress.json').is_file()]
    if len(found) > 1:
        raise ValueError('Ambiguous exported update')
    return found[0] if found else None


def next_attempt(round_dir, prefix):
    numbers = [int(p.name.rsplit('_', 1)[1]) for p in round_dir.glob(prefix + '_*') if p.is_dir()]
    return round_dir / f'{prefix}_{max(numbers, default=0)+1:03d}'


def verify_export_binding(checkpoint, previous, batch):
    progress = read(checkpoint / 'vln_progress.json')
    prior = read(previous / 'vln_progress.json')['policy_version'] if (previous / 'vln_progress.json').exists() else 0
    if progress['policy_version'] != prior + 1 or progress['source_batch_sha256'] != digest(batch):
        raise ValueError('Export does not belong to this policy/batch')
    if progress['data_sha256'] != digest(checkpoint / 'data.pt'):
        raise ValueError('Export cursor checksum mismatch')
    if not (checkpoint / 'optimizer.pt').is_file():
        raise ValueError('Export is missing optimizer state')
    return progress


class Driver:
    def __init__(self, args):
        self.a = args
        self.root = args.output.resolve()
        self.project = args.project.resolve()
        self.children = []
        self.wandb = None
        self.seen_steps = set()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / 'driver.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.check_orphans()
        except BaseException:
            self.lock.close()
            raise
        self.env = os.environ.copy()
        self.env.update(PYTHONPATH=f'{self.root}/runtime:{self.project}',
                        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                        HF_HUB_OFFLINE='1', WANDB_MODE='offline')
        self.server = f'http://127.0.0.1:{args.port}'

    def check_orphans(self):
        for path in self.root.glob('process_*.json'):
            record = read(path)
            live = process_identity(record['pid'])
            if live is not None and live == record['identity']:
                raise RuntimeError(f"Previous owned process still alive: {record['pid']}; do not start a second driver")
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                parts = path.read_bytes().decode().split('\0')
            except (OSError, UnicodeDecodeError):
                continue
            if any(p in ('panovln_dapo.collect', 'panovln_dapo.run_actor', 'panovln_dapo.actor',
                         'panovln_dapo.env_server','panovln_dapo.resident_train') for p in parts):
                if any(p.startswith(str(self.root) + '/') for p in parts):
                    raise RuntimeError(f'Previous stage still alive: {path.parent.name}')

    def setup(self):
        a = self.a
        science = dict(attention_backend=getattr(a,'attention_backend','auto'),resident=getattr(a,'resident',False),initial_checkpoint=str(a.checkpoint.resolve()), manifest=str(a.manifest.resolve()),
                       manifest_sha256=digest(a.manifest), environment_sha256=digest(a.environment),
                       epochs=a.epochs, seed=a.seed, rollout_n=a.rollout_n, target_groups=a.target_groups, generation_groups=getattr(a,'generation_groups',192),
                       lr=a.lr, mini_batch_decisions=a.mini_batch_decisions,
                       ppo_epochs=a.ppo_epochs, clip_low=.2, clip_high=.28, dual_clip=10.,
                       precision=getattr(a,'precision','fp32'), entropy_coeff=0., max_env_steps=getattr(a,'max_env_steps',200), max_decisions=getattr(a,'max_decisions',64),
                       max_response_tokens=getattr(a,'max_response_tokens',256), temperature=1., progress_coef=0.)
        config_path = self.root / 'config.json'
        if config_path.exists():
            config = read(config_path)
            if config['science'] != science:
                raise ValueError('Scientific configuration changed; restore it or choose a new RUN_DIR')
            if config['driver_sha256'] != digest(Path(__file__)):
                raise ValueError('Driver implementation changed; review before resuming')
            self.state = read(self.root / 'state.json')
            self.config = config
            self.check_runtime()
        else:
            # Refuse to disguise incomplete initialization as a new training run.
            if (self.root / 'runtime').exists():
                raise ValueError('Incomplete initialization; choose a new RUN_DIR')
            assets, render = read(a.assets_audit), read(a.render_audit)
            if not assets['passed'] or assets['episodes'] != 6901 or assets['scenes'] != 725:
                raise ValueError('Official asset gate not complete')
            if not render['passed'] or render['scenes'] != 725:
                raise ValueError('Official render gate not complete')
            package_hashes = make_runtime(self.project, self.root / 'runtime')
            shutil.copyfile(a.environment, self.root / 'environment.yaml')
            initial = a.checkpoint.resolve()
            if not initial.is_dir():
                raise ValueError('Initial checkpoint does not exist')
            if (initial / 'vln_progress.json').exists():
                progress = read(initial / 'vln_progress.json')
                if progress.get('precision','fp32')!=science['precision']:
                    raise ValueError('Initial trained checkpoint precision mismatch')
                for key in ('lr', 'ppo_epochs', 'mini_batch_decisions', 'clip_low', 'clip_high', 'dual_clip', 'entropy_coeff'):
                    if progress.get(key, 0.) != science[key]:
                        raise ValueError(f'Initial trained checkpoint hyperparameter mismatch: {key}')
                if not (initial / 'optimizer.pt').is_file() or progress['data_sha256'] != digest(initial / 'data.pt'):
                    raise ValueError('Initial trained checkpoint is incomplete')
            self.state = dict(completed_updates=0, optimizer_steps=0, current_checkpoint=str(initial),
                              stage='ready', data_exhausted=False, wandb_id=uuid.uuid4().hex[:16])
            self.config = dict(science=science, runtime_sha256=package_hashes,
                               driver_sha256=digest(Path(__file__)))
            # State first: a config file always implies initialized state exists.
            atomic(self.root / 'state.json', self.state)
            atomic(config_path, self.config)
        if not Path(self.state['current_checkpoint']).is_dir():
            raise ValueError('Latest checkpoint is missing')
        atomic(self.root / 'resources.json', dict(gpus=a.gpus, port=a.port, memory_limit_gib=a.memory_limit_gib,
                                                memory_limits_gib=getattr(a,'memory_limits_gib',None),
                                                max_updates=a.max_updates, keep_last=a.keep_last,
                                                keep_every=a.keep_every, time=time.time()))
        if not self.finished() and not getattr(a,'resident',False):
            import wandb
            (self.root / 'wandb_offline').mkdir(exist_ok=True)
            # Offline segments use the same logical id; explicit global steps survive restarts.
            self.wandb = wandb.init(project=a.wandb_project, mode='offline', id=self.state['wandb_id'],
                                    name=self.root.name, dir=str(self.root / 'wandb_offline'),
                                    config={**science, 'gpus': a.gpus, 'resume_round': self.state['completed_updates']})

    def check_runtime(self):
        actual = {p.name: digest(p) for p in (self.root / 'runtime/panovln_dapo').glob('*.py')}
        if actual != self.config['runtime_sha256']:
            raise ValueError('Isolated training runtime changed')
        if digest(self.root / 'environment.yaml') != self.config['science']['environment_sha256']:
            raise ValueError('Isolated environment configuration changed')

    def finished(self):
        return self.state.get('data_exhausted', False) or (
            self.a.max_updates > 0 and self.state['completed_updates'] >= self.a.max_updates)

    def status(self, stage, **extra):
        self.state.update(stage=stage, time=time.time(), **extra)
        atomic(self.root / 'state.json', self.state)
        print(json.dumps(self.state), flush=True)
        if self.wandb:
            self.wandb.summary['stage'] = stage
            self.wandb.summary['completed_updates'] = self.state['completed_updates']

    def capacity(self, gpus, idle=False):
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free',
                                       '--format=csv,noheader,nounits'], text=True)
        memory = {int(i): (int(u), int(f)) for i, u, f in (r.split(',') for r in raw.splitlines())}
        budgets=getattr(self.a,'memory_limits_gib',None)
        limits=dict(zip(self.a.gpus,budgets or [self.a.memory_limit_gib]*len(self.a.gpus)))
        for gpu in gpus:
            if gpu not in memory or (idle and memory[gpu][0] > 1000) or (
                    not idle and memory[gpu][1] < (limits[gpu] + 3) * 1024):
                raise RuntimeError(f'GPU {gpu} busy or insufficient memory; no other jobs were stopped')

    def spawn(self, command, log, gpu, label):
        self.check_runtime()
        env = self.env.copy()
        if gpu is None:
            env.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            env['CUDA_VISIBLE_DEVICES'] = gpu
        stream = log.open('x')
        try:
            child = subprocess.Popen([sys.executable, '-u', *command], env=env, cwd=self.project,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            stream.close()
        self.children.append(child)
        record = dict(pid=child.pid, identity=process_identity(child.pid), command=command)
        atomic(self.root / f'process_{label}.json', record)
        return child

    def stop_owned(self, child):
        # Only a Popen object created in this invocation can reach this function.
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                child.wait(timeout=10)
                return
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)

    def observe(self, result):
        path = result / 'actor.log'
        if not self.wandb or not path.exists():
            return
        for epoch, start, end, norm in re.findall(r'rank=0 PPO epoch=(\d+) rows=(\d+):(\d+) norm=([^\s]+)', path.read_text(errors='replace')):
            key = (str(result), epoch, start, end)
            if key not in self.seen_steps:
                self.seen_steps.add(key)
                self.wandb.log({'train/round': self.state['completed_updates'] + 1,
                                'train/attempt_optimizer_step': len(self.seen_steps),
                                'train/committed_optimizer_steps': self.state['optimizer_steps'],
                                'train/gradient_norm': float(norm), 'train/decision_end': int(end)})

    def launch(self, command, log, gpu='', result=None):
        child = self.spawn(command, log, gpu, 'stage')
        offsets = {}
        while child.poll() is None:
            forward_log(log, offsets)
            if result:
                forward_log(result / 'actor.log', offsets)
                self.observe(result)
            time.sleep(5)
        forward_log(log, offsets)
        if result:
            forward_log(result / 'actor.log', offsets)
            self.observe(result)
        if child.returncode:
            raise RuntimeError(f'Stage failed ({child.returncode}); see {log}')

    def service(self, round_dir):
        self.capacity(self.a.gpus[:2], idle=not getattr(self.a,'resident',False))
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', self.a.port))
        log = round_dir / f'environment_{time.time_ns()}.log'
        child = self.spawn(['-m', 'panovln_dapo.env_server', '--config', str(self.root / 'environment.yaml'),
                            '--gpu-ids', str(self.a.gpus[0] if getattr(self.a,'resident',False) else self.a.gpus[1]), '--pool-size', str(getattr(self.a,'env_pool_size',1)), '--port', str(self.a.port)],
                           log, None, 'environment')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f'Environment failed; see {log}')
            try:
                with opener.open(self.server + '/health', timeout=2) as response:
                    if response.status == 200:
                        return child
            except OSError:
                pass
            time.sleep(1)
        raise RuntimeError('Owned environment did not become ready')

    def collect(self, checkpoint, destination):
        return ['-m', 'panovln_dapo.collect', '--attention-backend',getattr(self.a,'attention_backend','auto'),'--precision',getattr(self.a,'precision','fp32'), '--checkpoint', str(checkpoint),
                '--manifest', str(self.a.manifest.resolve()), '--server', self.server, '--output', str(destination),
                '--epochs', str(self.a.epochs), '--seed', str(self.a.seed), '--rollout-n', str(self.a.rollout_n),
                '--target-groups', str(self.a.target_groups), '--generation-groups', str(getattr(self.a,'generation_groups',192)),
                '--inference-batch-size', str(getattr(self.a,'inference_batch_size',1)),
                '--inference-wait-ms', str(getattr(self.a,'inference_wait_ms',2.)),
                '--rollout-window', str(getattr(self.a,'rollout_window',1)),
                '--rollout-scheduler', getattr(self.a,'rollout_scheduler','sliding'),
                '--max-generation-batches', '0', '--max-response-tokens', str(getattr(self.a,'max_response_tokens',256)), '--max-env-steps', str(getattr(self.a,'max_env_steps',200)),
                '--max-decisions', str(getattr(self.a,'max_decisions',64)), '--temperature', '1', '--progress-coef', '0', '--min-free-gib', '0']

    def actor(self, checkpoint, batch, result):
        args = ['-m', 'panovln_dapo.run_actor', '--attention-backend',getattr(self.a,'attention_backend','auto'),'--precision',getattr(self.a,'precision','fp32'), '--checkpoint', str(checkpoint), '--batch', str(batch),
                '--output', str(result), '--gpus', ','.join(map(str, self.a.gpus)), '--shared-gpus',
                '--memory-limit-gib', str(self.a.memory_limit_gib), '--mini-batch-decisions', str(self.a.mini_batch_decisions),
                '--ppo-micro-batch-size', str(getattr(self.a,'ppo_micro_batch_size',1)),
                '--log-prob-micro-batch-size', str(getattr(self.a,'log_prob_micro_batch_size',1)),
                '--optimizer-offload-mode',getattr(self.a,'optimizer_offload_mode','bulk'),
                '--fsdp-size', str(getattr(self.a,'fsdp_size',-1)),
                '--use-torch-compile' if getattr(self.a,'use_torch_compile',True) else '--no-use-torch-compile',
                '--gradient-checkpointing' if getattr(self.a,'gradient_checkpointing',True) else '--no-gradient-checkpointing',
                '--forward-prefetch' if getattr(self.a,'forward_prefetch',False) else '--no-forward-prefetch',
                '--optimizer-offload' if getattr(self.a,'optimizer_offload',True) else '--no-optimizer-offload',
                '--lr', str(self.a.lr), '--ppo-epochs', str(self.a.ppo_epochs),
                '--clip-low', '.2', '--clip-high', '.28', '--dual-clip', '10', '--entropy-coeff', '0']
        if (checkpoint / 'vln_progress.json').exists():
            args += ['--resume-optimizer', '--allow-world-size-change', '--resume-seed', str(self.a.seed)]
        return args

    def prune(self):
        """Delete only this run's committed old data, after latest.json is durable."""
        n = self.state['completed_updates']
        latest = Path(self.state['current_checkpoint']).resolve()
        for number in range(1, n + 1):
            round_dir = self.root / f'round_{number:05d}'
            marker = round_dir / 'commit.json'
            if not marker.exists():
                continue
            commit = read(marker)
            checkpoint = Path(commit['checkpoint'])
            if not checkpoint.resolve().is_relative_to(round_dir.resolve()) or checkpoint.is_symlink():
                raise ValueError('Refusing retention outside this run')
            keep = self.a.keep_last == 0 or number > n - self.a.keep_last or (self.a.keep_every > 0 and number % self.a.keep_every == 0)
            if not keep and checkpoint.exists() and checkpoint.resolve() != latest:
                shutil.rmtree(checkpoint)
                atomic(round_dir / 'pruned.json', {'checkpoint': str(checkpoint), 'time': time.time()})
            if not self.a.keep_rollouts:
                for collection in round_dir.glob('collection_*'):
                    if not collection.is_dir() or collection.is_symlink():
                        continue
                    for path in collection.glob('*.json'):
                        if path.name == 'selected_batch.json' or re.fullmatch(r'v\d+_\d+_\d+_r\d+\.json', path.name):
                            path.unlink()

    def train_resident(self):
        """Keep a torchrun actor alive across collection/update/save boundaries."""
        if self.finished():return
        a=self.a
        resident_root=self.root/'resident';resident_root.mkdir(exist_ok=True)
        initial=Path(self.config['science']['initial_checkpoint'])
        initial_progress=initial/'vln_progress.json'
        base_version=read(initial_progress)['policy_version'] if initial_progress.exists() else 0
        actor=dict(attention_backend=getattr(a,'attention_backend','auto'),precision=getattr(a,'precision','fp32'),lr=a.lr,ppo_epochs=a.ppo_epochs,
                   mini_batch_decisions=a.mini_batch_decisions,
                   ppo_micro_batch_size=getattr(a,'ppo_micro_batch_size',1),
                   log_prob_micro_batch_size=getattr(a,'log_prob_micro_batch_size',1),
                   optimizer_offload=getattr(a,'optimizer_offload',True),
                   optimizer_offload_mode=getattr(a,'optimizer_offload_mode','bulk'),
                   gradient_checkpointing=getattr(a,'gradient_checkpointing',True),
                   use_torch_compile=getattr(a,'use_torch_compile',True),
                   forward_prefetch=getattr(a,'forward_prefetch',False),fsdp_size=getattr(a,'fsdp_size',-1),
                   allow_world_size_change=True,resume_seed=a.seed)
        config=dict(checkpoint=str(initial),output=str(resident_root),manifest=str(a.manifest.resolve()),
                    server=self.server,seed=a.seed,epochs=a.epochs,memory_limit_gib=getattr(a,'memory_limits_gib',None) or a.memory_limit_gib,
                    max_updates=base_version+a.max_updates if a.max_updates else 0,
                    save_freq=a.save_freq,param_offload=a.param_offload,
                    wandb_project=a.wandb_project,wandb_id=self.state['wandb_id'],actor=actor,
                    keep_last=a.keep_last,keep_every=a.keep_every,keep_rollouts=a.keep_rollouts,
                    collection=dict(max_response_tokens=getattr(a,'max_response_tokens',256),
                        max_env_steps=getattr(a,'max_env_steps',200),max_decisions=getattr(a,'max_decisions',64),rollout_n=a.rollout_n,target_groups=a.target_groups,
                        generation_groups=getattr(a,'generation_groups',192),
                        rollout_window=getattr(a,'rollout_window',1),
                        rollout_scheduler=getattr(a,'rollout_scheduler','sliding'),
                        inference_batch_size=getattr(a,'inference_batch_size',1),
                        inference_wait_ms=getattr(a,'inference_wait_ms',2.)))
        path=self.root/'resident_launch.json';atomic(path,config)
        self.capacity(a.gpus)
        self.status('resident_running',resident_output=str(resident_root))
        service=self.service(resident_root)
        try:
            command=['-m','torch.distributed.run','--standalone',f'--nproc-per-node={len(a.gpus)}',
                     '-m','panovln_dapo.resident_train','--config',str(path)]
            self.launch(command,resident_root/f'worker_{time.time_ns()}.log',','.join(map(str,a.gpus)))
        finally:
            self.stop_owned(service)
        complete=read(resident_root/'complete.json')
        self.status('data_exhausted' if complete['data_exhausted'] else 'training_complete',
                    completed_updates=complete['policy_version']-base_version,
                    optimizer_steps=complete['optimizer_steps'],current_checkpoint=complete['checkpoint'],
                    data_exhausted=complete['data_exhausted'])
        atomic(self.root/'latest.json',dict(checkpoint=complete['checkpoint'],
                                           completed_updates=self.state['completed_updates']))

    def train(self):
        self.setup()
        if getattr(self.a,'resident',False):
            return self.train_resident()
        self.state.pop('error', None)
        atomic(self.root / 'latest.json', dict(checkpoint=self.state['current_checkpoint'],
                                               completed_updates=self.state['completed_updates']))
        self.prune()
        while not self.finished():
            number = self.state['completed_updates'] + 1
            round_dir = self.root / f'round_{number:05d}'
            round_dir.mkdir(exist_ok=True)
            previous = Path(self.state['current_checkpoint'])
            collection = complete_collection(round_dir)
            result = exported_update(round_dir)
            if result and collection is None:
                raise ValueError('Export lacks completed on-policy collection')
            if collection is None:
                collection = next_attempt(round_dir, 'collection')
                self.status('collecting', round=number, collection=str(collection))
                service = self.service(round_dir)
                try:
                    self.launch(self.collect(previous, collection), round_dir / (collection.name + '.log'), str(self.a.gpus[0]))
                finally:
                    self.stop_owned(service)
            metrics = read(collection / 'complete.json')
            batch = collection / 'selected_batch.json'
            if metrics['valid_groups'] == 0:
                if not metrics['data_exhausted']:
                    raise ValueError('No valid groups without dataset exhaustion')
                self.status('data_exhausted', data_exhausted=True, terminal_cursor=read(batch)['cursor'])
                break
            if result is None:
                result = next_attempt(round_dir, 'update')
                self.status('updating', round=number, actor=str(result))
                self.capacity(self.a.gpus)
                self.launch(self.actor(previous, batch, result), round_dir / (result.name + '.log'), result=result)
            checkpoint = result / 'checkpoint'
            progress = verify_export_binding(checkpoint, previous, batch)
            self.status('auditing_export', round=number)
            audit_path = round_dir / f'audit_{time.time_ns()}.json'
            self.launch(['-m', 'panovln_dapo.checkpoint_audit', '--initial', str(previous),
                         '--candidate', str(checkpoint), '--output', str(audit_path)],
                        round_dir / (audit_path.stem + '.log'))
            commit = dict(round=number, checkpoint=str(checkpoint), previous=str(previous),
                          collection=str(collection), audit=str(audit_path), metrics=progress,
                          collection_metrics=metrics, time=time.time())
            atomic(round_dir / 'commit.json', commit)
            # State is authoritative; latest.json is a convenience pointer repaired on resume.
            self.status('ready', completed_updates=number, current_checkpoint=str(checkpoint),
                        optimizer_steps=self.state['optimizer_steps'] + progress['optimizer_steps'],
                        data_exhausted=metrics.get('data_exhausted', False))
            atomic(self.root / 'latest.json', dict(checkpoint=str(checkpoint), completed_updates=number))
            if self.wandb:
                self.wandb.log({**{f'train_final/{k}': v for k, v in progress.items()
                                  if isinstance(v, (int, float)) and math.isfinite(v)},
                                'train/round': number, 'train/committed_optimizer_steps': self.state['optimizer_steps']})
                self.wandb.summary['latest_checkpoint'] = str(checkpoint)
            self.prune()
        atomic(self.root / 'latest.json', dict(checkpoint=self.state['current_checkpoint'],
                                               completed_updates=self.state['completed_updates']))
        self.status('data_exhausted' if self.state['data_exhausted'] else 'training_complete')

    def close(self):
        for child in reversed(self.children):
            self.stop_owned(child)
        if self.wandb:
            self.wandb.finish(exit_code=0 if self.state['stage'] in ('training_complete', 'data_exhausted') else 1)
        self.lock.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inference-batch-size',type=int,default=1)
    p.add_argument('--inference-wait-ms',type=float,default=2.)
    p.add_argument('--rollout-window',type=int,default=1)
    p.add_argument('--rollout-scheduler',choices=['sliding','window'],default='sliding')
    p.add_argument('--resident',action='store_true')
    p.add_argument('--save-freq',type=int,default=5)
    p.add_argument('--param-offload',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--env-pool-size',type=int,default=1)
    p.add_argument('--attention-backend',choices=['auto','sdpa','flash_attention_2'],default='auto')
    p.add_argument('--precision',choices=['fp32','bf16'],default='bf16')
    p.add_argument('--project', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--environment', required=True, type=Path)
    p.add_argument('--assets-audit', required=True, type=Path)
    p.add_argument('--render-audit', required=True, type=Path)
    p.add_argument('--gpus', required=True)
    p.add_argument('--port', type=int, default=8037)
    p.add_argument('--memory-limit-gib', type=float, default=40.)
    p.add_argument('--memory-limits-gib',type=float,nargs='+',help='Resident per-rank limits in --gpus order; overrides scalar limit')
    p.add_argument('--max-updates', type=int, default=0)
    p.add_argument('--max-response-tokens',type=int,default=256)
    p.add_argument('--max-env-steps',type=int,default=200)
    p.add_argument('--max-decisions',type=int,default=64)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rollout-n', type=int, default=8)
    p.add_argument('--target-groups', type=int, default=128)
    p.add_argument('--generation-groups', type=int, default=192)
    p.add_argument('--ppo-micro-batch-size',type=int,default=4)
    p.add_argument('--log-prob-micro-batch-size',type=int,default=8)
    p.add_argument('--use-torch-compile',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--fsdp-size',type=int,default=-1)
    p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--forward-prefetch',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--optimizer-offload-mode',choices=['bulk','parameter'],default='bulk')
    p.add_argument('--optimizer-offload',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--mini-batch-decisions', type=int, default=1024)
    p.add_argument('--lr', type=float, default=5e-7)
    p.add_argument('--ppo-epochs', type=int, default=1)
    p.add_argument('--keep-last', type=int, default=2)
    p.add_argument('--keep-every', type=int, default=0)
    p.add_argument('--keep-rollouts', action='store_true')
    p.add_argument('--wandb-project', default='panovln-dapo')
    a = p.parse_args()
    a.gpus = [int(g) for g in a.gpus.split(',')]
    if a.save_freq<1 or a.env_pool_size<1:p.error('Invalid resident save/environment configuration')
    if len(a.gpus)>4:p.error('At most four GPUs are authorized')
    if len(a.gpus) < 2 or len(set(a.gpus)) != len(a.gpus) or min(a.gpus) < 0:
        p.error('Need at least two distinct nonnegative GPUs')
    if a.memory_limits_gib is not None:
        if not a.resident:p.error('Per-rank memory limits require --resident')
        if len(a.memory_limits_gib)!=len(a.gpus) or any(not math.isfinite(v) or v<=0 for v in a.memory_limits_gib):
            p.error('Need one positive finite memory limit per GPU')
    if min(a.max_response_tokens,a.max_env_steps,a.max_decisions)<1:p.error('Training length limits must be positive')
    if a.max_updates < 0 or min(a.epochs, a.target_groups, a.generation_groups, a.ppo_epochs) < 1 or min(a.keep_last,a.keep_every) < 0:
        p.error('Invalid training/retention budget')
    if min(a.ppo_micro_batch_size,a.log_prob_micro_batch_size)<1:
        p.error('Microbatch sizes must be positive')
    if a.fsdp_size!=-1 and (a.fsdp_size<1 or len(a.gpus)%a.fsdp_size):
        p.error('FSDP size must divide selected GPU count')
    if a.inference_batch_size<1 or not math.isfinite(a.inference_wait_ms) or a.inference_wait_ms<0:
        p.error('Invalid inference batching configuration')
    if a.rollout_window < 1:
        p.error('Rollout window must be positive')
    if a.rollout_n < 2 or a.mini_batch_decisions < len(a.gpus):
        p.error('Invalid group/minibatch size')
    if not 0 < a.memory_limit_gib or not math.isfinite(a.memory_limit_gib) or not 0 < a.lr or not math.isfinite(a.lr):
        p.error('Invalid memory/lr')
    return a


def main():
    args = parse_args()
    driver = Driver(args)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        driver.train()
    except BaseException as exc:
        if hasattr(driver, 'state'):
            try:
                driver.status('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
            except OSError:
                print('Could not write status; previous durable checkpoint remains authoritative.', file=sys.stderr)
        raise
    finally:
        driver.close()


if __name__ == '__main__':
    main()
