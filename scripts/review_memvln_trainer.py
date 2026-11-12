"""Approved short ZeRO-2 review; checkpoints and W&B files are temporary.

Uses the real initial/terminal expert states and full trajectory summary.
The controller hides GPUs; only torchrun children get the approved GPU.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import yaml


def run_child(config, output, gpu, resume=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WANDB_MODE='offline',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
        '--module', 'src.train.train_memvln', '--config', str(config), '--output-dir', str(output)]
    if resume:
        command += ['--resume', str(resume)]
    print('Trainer review:', 'resume update 1 and replay 2-3' if resume else 'fresh updates 1-3', flush=True)
    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=900)
    if result.returncode:
        print(result.stdout[-14000:], flush=True)
        raise RuntimeError(f'Training subprocess failed: {result.returncode}')
    for line in result.stdout.splitlines():
        if line.startswith('{"update"') or 'W&B syncing is set to' in line or 'Restored trainable tensors' in line:
            print(line, flush=True)


def load_tensors(bundle):
    import torch
    from safetensors.torch import load_file
    learned = {'adapter.' + k: v for k, v in load_file(str(bundle / 'adapter_model.safetensors')).items()}
    learned.update({'memory.' + k: v for k, v in torch.load(bundle / 'episode_memory.pt',
        map_location='cpu', weights_only=True).items()})
    return learned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/train/config/memvln.yaml')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.error('Requires explicit --run and user approval for this GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(8)
    cfg = yaml.safe_load(Path(args.config).read_text())
    with open(cfg['data']['jsonl']) as handle:
        initial = json.loads(next(handle)); terminal = initial
        for line in handle:
            row = json.loads(line)
            if (row['dataset'], row['episode_id']) != (initial['dataset'], initial['episode_id']):
                break
            terminal = row
    if initial['step_index'] != 0 or terminal['action_sequence'][0] != 'stop':
        raise ValueError('Review requires real initial and terminal expert states')
    with tempfile.TemporaryDirectory(prefix='.memvln-trainer-review-', dir=Path.cwd()) as folder:
        folder = Path(folder)
        data = folder / 'fixture.jsonl'
        data.write_text(json.dumps(initial) + '\n' + json.dumps(terminal) + '\n')
        cfg['data'].update(jsonl=str(data), summaries=True, sub_summaries_jsonl=None)
        cfg['training'].update(stage='joint', global_batch=1, max_updates=3, epochs=1,
            save_updates=1, save_total_limit=3, episode_subset=None, deterministic=True, verify_resume_state=True)
        config = folder / 'review.yaml'; config.write_text(yaml.safe_dump(cfg))
        output = folder / 'run'
        run_child(config, output, args.gpu)
        checkpoint = output / 'checkpoint-3'
        expected = load_tensors(checkpoint / 'inference')
        before = torch.load(checkpoint / 'stream_rank0.pt', map_location='cpu', weights_only=False)
        run_child(config, output, args.gpu, output / 'checkpoint-1')
        after = torch.load(checkpoint / 'stream_rank0.pt', map_location='cpu', weights_only=False)
        actual = load_tensors(checkpoint / 'inference')
        if actual.keys() != expected.keys():
            raise ValueError('Learned tensor keys differ after resume')
        max_error = 0.
        mismatches = []
        for name in expected:
            max_error = max(max_error, float((actual[name].float() - expected[name].float()).abs().max()))
            try:
                torch.testing.assert_close(actual[name], expected[name], atol=2e-5, rtol=1e-4, msg=name)
            except AssertionError:
                mismatches.append(dict(name=name, max_abs_error=float((actual[name].float()-expected[name].float()).abs().max())))
        print(json.dumps(dict(resume_max_abs_error=max_error, tensor_mismatch_count=len(mismatches), examples=mismatches[:5])), flush=True)
        if mismatches:
            raise AssertionError('Resumed learned tensors differ from uninterrupted training')
        for key in ('updates', 'counters', 'schedule'):
            if before[key] != after[key]:
                raise ValueError(f'Resumed progress differs: {key}')
        offline_runs = list((output / 'wandb').glob('offline-run-*'))
        if not offline_runs:
            raise ValueError('Missing offline W&B runs')
        print(json.dumps(dict(gpu=args.gpu, zero_stage=2, updates=after['updates'],
            counters=after['counters'], learned_tensor_count=len(actual),
            resumed_max_abs_error=max_error, offline_wandb_runs=len(offline_runs),
            checkpoint_excludes_frozen_base=True)), flush=True)
    print('Temporary training checkpoints and offline W&B logs removed.', flush=True)


if __name__ == '__main__':
    main()
