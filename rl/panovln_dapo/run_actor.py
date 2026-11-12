"""Launch one verified batch update on explicitly selected idle GPUs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',required=True,type=Path)
    p.add_argument('--attention-backend',choices=['auto','sdpa','flash_attention_2'],default='auto')
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--gpus',required=True)
    p.add_argument('--resume-optimizer',action='store_true')
    p.add_argument('--allow-world-size-change',action='store_true')
    p.add_argument('--resume-seed',type=int,default=42)
    p.add_argument('--shared-gpus',action='store_true')
    p.add_argument('--memory-limit-gib',type=float)
    p.add_argument('--lr',type=float,default=5e-7)
    p.add_argument('--entropy-coeff',type=float,default=0.)
    p.add_argument('--clip-low',type=float,default=.2)
    p.add_argument('--clip-high',type=float,default=.28)
    p.add_argument('--dual-clip',type=float,default=10.)
    p.add_argument('--ppo-epochs',type=int,default=1)
    p.add_argument('--ppo-micro-batch-size',type=int,default=1)
    p.add_argument('--log-prob-micro-batch-size',type=int,default=1)
    p.add_argument('--use-torch-compile',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--fsdp-size',type=int,default=-1)
    p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--forward-prefetch',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--optimizer-offload-mode',choices=['bulk','parameter'],default='bulk')
    p.add_argument('--optimizer-offload',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--mini-batch-decisions',type=int,default=128)
    args=p.parse_args();gpus=[int(g) for g in args.gpus.split(',')]
    if len(set(gpus))!=len(gpus) or not gpus or any(g<0 for g in gpus):raise ValueError('Invalid GPU list')
    if args.fsdp_size!=-1 and (args.fsdp_size<1 or len(gpus)%args.fsdp_size):
        raise ValueError('FSDP size must divide selected GPU count')
    args.output.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):
        record=dict(stage=stage,time=time.time(),**kw)
        (args.output/'status.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    try:
        from .actor import load_batch
        payload,rows=load_batch(args.batch)
        status('batch_validated',decisions=len(rows),trajectories=len(payload['trajectories']))
        del payload,rows
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
        if args.shared_gpus:
            if args.memory_limit_gib is None or not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib<=0:
                raise ValueError('Shared GPUs require an explicit positive allocator memory limit')
            free_raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(line.split(',')[0]):int(line.split(',')[1]) for line in free_raw.splitlines()}
            if any(free.get(g,0)<(args.memory_limit_gib+3)*1024 for g in gpus):raise RuntimeError(f'Shared GPUs lack requested budget plus 3 GiB headroom: {free}')
        elif any(g not in memory or memory[g]>1000 for g in gpus):raise RuntimeError(f'Requested GPUs not idle: {memory}')
        snapshot=args.output/'source_snapshot';snapshot.mkdir();hashes={}
        for source in Path('rl/panovln_dapo').glob('*.py'):
            data=source.read_bytes();(snapshot/source.name).write_bytes(data);hashes[source.name]=hashlib.sha256(data).hexdigest()
        command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={len(gpus)}',
            '-m','panovln_dapo.actor','--batch',str(args.batch),'--checkpoint',args.checkpoint,
            '--attention-backend',args.attention_backend,'--precision',args.precision,'--output',str(args.output/'checkpoint'),'--lr',str(args.lr),'--ppo-epochs',str(args.ppo_epochs),
            '--ppo-micro-batch-size',str(args.ppo_micro_batch_size),'--log-prob-micro-batch-size',str(args.log_prob_micro_batch_size),
            '--mini-batch-decisions',str(args.mini_batch_decisions),
            '--optimizer-offload-mode',args.optimizer_offload_mode,
            '--fsdp-size',str(args.fsdp_size),
            '--use-torch-compile' if args.use_torch_compile else '--no-use-torch-compile',
            '--gradient-checkpointing' if args.gradient_checkpointing else '--no-gradient-checkpointing',
            '--forward-prefetch' if args.forward_prefetch else '--no-forward-prefetch',
            '--optimizer-offload' if args.optimizer_offload else '--no-optimizer-offload',
            '--clip-low',str(args.clip_low),'--clip-high',str(args.clip_high),'--dual-clip',str(args.dual_clip),'--entropy-coeff',str(args.entropy_coeff)]
        if args.resume_optimizer:command.append('--resume-optimizer')
        if args.allow_world_size_change:command.append('--allow-world-size-change')
        if args.memory_limit_gib is not None:command.extend(['--memory-limit-gib',str(args.memory_limit_gib)])
        command.extend(['--resume-seed',str(args.resume_seed)])
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpus,OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        (args.output/'run_manifest.json').write_text(json.dumps(dict(command=command,gpus=gpus,source_sha256=hashes,
            batch_sha256=hashlib.sha256(args.batch.read_bytes()).hexdigest()),indent=2)+'\n')
        with (args.output/'actor.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            status('actor_running',pid=child.pid,gpus=gpus);code=child.wait()
        if code:raise RuntimeError(f'Actor exited with status {code}; inspect actor.log')
        if not (args.output/'checkpoint/vln_progress.json').exists():raise RuntimeError('Missing checkpoint metadata')
        status('update_exported',checkpoint=str(args.output/'checkpoint'))
    except BaseException as exc:
        status('failed',error=str(exc));raise


if __name__=='__main__':main()
