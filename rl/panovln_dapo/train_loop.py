"""Sequential multi-update DAPO driver with durable stage records.

The training environment is an already-running dedicated service on a GPU that
is not used by the actor. Each collection process exits before the actor starts.
Resume reuses completed on-policy collections and never overwrites attempts.
Checkpoints are retained; actor export refuses insufficient disk space.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic_json(path, payload):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2)+'\n')
    temporary.replace(path)


def active_round_processes(directory):
    """Detect surviving stage processes even after their driver was interrupted."""
    prefix=str(directory.resolve())+'/'
    found=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:parts=path.read_bytes().decode().split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(part in ('panovln_dapo.collect','panovln_dapo.run_actor','panovln_dapo.actor','panovln_dapo.evaluate') for part in parts):
            if any(part.startswith(prefix) for part in parts):found.append(int(path.parent.name))
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--server',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--actor-gpus',default='3,6')
    p.add_argument('--rollout-gpu',type=int,default=3)
    p.add_argument('--updates',type=int,default=3)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rollout-n',type=int,default=4)
    p.add_argument('--target-groups',type=int,default=8)
    p.add_argument('--generation-groups',type=int,default=1)
    p.add_argument('--max-generation-batches',type=int,default=0)
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--max-response-tokens',type=int,default=64)
    p.add_argument('--max-env-steps',type=int,default=500)
    p.add_argument('--max-decisions',type=int,default=500)
    p.add_argument('--progress-coef',type=float,default=0.)
    p.add_argument('--entropy-coeff',type=float,default=0.)
    p.add_argument('--clip-low',type=float,default=.2)
    p.add_argument('--clip-high',type=float,default=.28)
    p.add_argument('--dual-clip',type=float,default=10.)
    p.add_argument('--ppo-epochs',type=int,default=1)
    p.add_argument('--lr',type=float,default=5e-7)
    p.add_argument('--mini-batch-decisions',type=int,default=128)
    p.add_argument('--eval-baseline',type=Path)
    p.add_argument('--eval-manifest',type=Path)
    p.add_argument('--eval-server')
    p.add_argument('--eval-every',type=int,default=1)
    p.add_argument('--resume-driver',action='store_true')
    args=p.parse_args()
    if args.updates<1:raise ValueError('Need a positive update budget')
    args.output=args.output.resolve();args.checkpoint=args.checkpoint.resolve();args.manifest=args.manifest.resolve()
    evaluation_enabled=any((args.eval_baseline,args.eval_manifest,args.eval_server))
    if evaluation_enabled:
        if not all((args.eval_baseline,args.eval_manifest,args.eval_server)) or args.eval_every<1:
            raise ValueError('Periodic evaluation needs baseline, manifest, server and positive frequency')
        args.eval_baseline=args.eval_baseline.resolve();args.eval_manifest=args.eval_manifest.resolve()
        from .compare import read_run
        baseline_config,_=read_run(args.eval_baseline)
        if hashlib.sha256(args.eval_manifest.read_bytes()).hexdigest()!=baseline_config['manifest_sha256']:
            raise ValueError('Evaluation manifest differs from baseline')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume_driver'}
    config['manifest_sha256']=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if evaluation_enabled:
        config['eval_baseline_sha256']={name:hashlib.sha256((args.eval_baseline/name).read_bytes()).hexdigest()
                                       for name in ('config.json','complete.json','results.jsonl')}
    sources={p.name:p.read_bytes() for p in Path(__file__).parent.glob('*.py')}
    config['implementation_sha256']={name:hashlib.sha256(data).hexdigest() for name,data in sources.items()}
    if args.resume_driver:
        if json.loads((args.output/'config.json').read_text())!=config:raise ValueError('Driver resume configuration changed')
        state=json.loads((args.output/'state.json').read_text())
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        atomic_json(args.output/'config.json',config)
        snapshot=args.output/'source_snapshot';snapshot.mkdir()
        for name,data in sources.items():(snapshot/name).write_bytes(data)
        state=dict(completed_updates=0,current_checkpoint=str(args.checkpoint),stage='ready')
        atomic_json(args.output/'state.json',state)
    # File lock is held for the entire driver lifetime; stage PIDs are also
    # inspected because an orphaned subprocess may outlive a crashed driver.
    import fcntl
    lock=(args.output/'driver.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads((args.output/'state.json').read_text())
    def update(stage,**kw):
        state.update(stage=stage,time=time.time(),**kw)
        atomic_json(args.output/'state.json',state);print(json.dumps(state),flush=True)
    def launch(command, log, gpu):
        current={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
        if current!=config['implementation_sha256']:raise RuntimeError('Source changed during driver execution')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        with log.open('x') as stream:
            child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            update(state['stage'],child_pid=child.pid,command=command)
            code=child.wait()
        if code:raise RuntimeError(f'Stage exited with status {code}; inspect {log}')
        state.pop('child_pid',None)
    try:
        while state['completed_updates']<args.updates and not state.get('data_exhausted',False):
            number=state['completed_updates']+1
            round_dir=args.output/f'round_{number:04d}';round_dir.mkdir(exist_ok=True)
            active=active_round_processes(round_dir)
            if active:raise RuntimeError(f'Previous stage processes still active: {active}')
            checkpoint=Path(state['current_checkpoint'])
            # An interrupted driver may have a fully exported update. Adopt it
            # only after the same full tensor/checksum audit as a fresh stage.
            updates=sorted(d for d in round_dir.glob('update_*') if d.is_dir())
            successful=[d for d in updates if (d/'checkpoint/vln_progress.json').exists()
                        and (d/'status.json').exists() and json.loads((d/'status.json').read_text())['stage']=='update_exported']
            if len(successful)>1:raise ValueError('Multiple committed updates for one round')
            if successful:
                result=successful[0]
                completed=[d for d in round_dir.glob('collection_*') if d.is_dir() and (d/'complete.json').exists() and not (d/'failure.json').exists()]
                if len(completed)!=1:raise ValueError('Committed update lacks its unique completed collection')
                collection=completed[0]
            else:
                collections=sorted(d for d in round_dir.glob('collection_*') if d.is_dir())
                completed=[d for d in collections if (d/'complete.json').exists() and not (d/'failure.json').exists()]
                if len(completed)>1:raise ValueError('Multiple completed collections for one round')
                if completed:collection=completed[0]
                else:
                    collection=round_dir/f'collection_{len(collections)+1:02d}'
                    update('collecting',round=number)
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
                    memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
                    if memory.get(args.rollout_gpu,100000)>1000:raise RuntimeError('Rollout GPU not idle')
                    command=[sys.executable,'-u','-m','panovln_dapo.collect','--checkpoint',str(checkpoint),
                        '--manifest',str(args.manifest),'--server',args.server,'--output',str(collection),
                        '--epochs',str(args.epochs),'--seed',str(args.seed),'--rollout-n',str(args.rollout_n),
                        '--target-groups',str(args.target_groups),'--generation-groups',str(args.generation_groups),
                        '--max-generation-batches',str(args.max_generation_batches),
                        '--temperature',str(args.temperature),'--max-response-tokens',str(args.max_response_tokens),
                        '--max-env-steps',str(args.max_env_steps),'--max-decisions',str(args.max_decisions),
                        '--progress-coef',str(args.progress_coef)]
                    launch(command,round_dir/(collection.name+'.log'),str(args.rollout_gpu))
                collection_metrics=json.loads((collection/'complete.json').read_text())
                if collection_metrics.get('valid_groups')==0 and collection_metrics.get('data_exhausted'):
                    terminal=json.loads((collection/'selected_batch.json').read_text())
                    update('data_exhausted',data_exhausted=True,terminal_cursor=terminal['cursor'],last_collection=str(collection))
                    break
                from .actor import load_batch
                payload,rows=load_batch(collection/'selected_batch.json')
                if Path(payload['checkpoint']).resolve()!=checkpoint.resolve():raise ValueError('Collection policy mismatch')
                del payload,rows
                result=round_dir/f'update_{len(updates)+1:02d}'
                update('updating',round=number)
                command=[sys.executable,'-u','-m','panovln_dapo.run_actor','--batch',str(collection/'selected_batch.json'),
                    '--checkpoint',str(checkpoint),'--output',str(result),'--gpus',args.actor_gpus,
                    '--lr',str(args.lr),'--mini-batch-decisions',str(args.mini_batch_decisions),
                    '--ppo-epochs',str(args.ppo_epochs),'--clip-low',str(args.clip_low),
                    '--clip-high',str(args.clip_high),'--dual-clip',str(args.dual_clip),'--entropy-coeff',str(args.entropy_coeff)]
                if (checkpoint/'vln_progress.json').exists():command.append('--resume-optimizer')
                launch(command,round_dir/(result.name+'.log'),'')
            update('auditing_export',round=number)
            from .checkpoint_audit import audit
            import torch
            torch.set_num_threads(2)
            report=audit(checkpoint,result/'checkpoint')
            atomic_json(round_dir/'checkpoint_audit.json',report)
            collection_metrics=json.loads((collection/'complete.json').read_text())
            if evaluation_enabled and (number%args.eval_every==0 or number==args.updates or collection_metrics.get('data_exhausted')):
                from .compare import compare, read_run
                attempts=sorted(d for d in round_dir.glob('evaluation_*') if d.is_dir())
                completed_evaluations=[d for d in attempts if (d/'complete.json').exists() and not (d/'failure.json').exists()]
                if len(completed_evaluations)>1:raise ValueError('Multiple completed evaluations for one round')
                if completed_evaluations:
                    evaluation=completed_evaluations[0]
                    evaluation_config,_=read_run(evaluation)
                    if Path(evaluation_config['checkpoint']).resolve()!=(result/'checkpoint').resolve():
                        raise ValueError('Evaluation checkpoint mismatch')
                else:
                    evaluation=round_dir/f'evaluation_{len(attempts)+1:02d}'
                    update('evaluating',round=number)
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
                    memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
                    if memory.get(args.rollout_gpu,100000)>1000:raise RuntimeError('Evaluation GPU not idle')
                    command=[sys.executable,'-u','-m','panovln_dapo.evaluate','--checkpoint',str(result/'checkpoint'),
                             '--manifest',str(args.eval_manifest),'--server',args.eval_server,
                             '--max-episodes',str(baseline_config['max_episodes']),'--output',str(evaluation)]
                    launch(command,round_dir/(evaluation.name+'.log'),str(args.rollout_gpu))
                current_baseline_hashes={name:hashlib.sha256((args.eval_baseline/name).read_bytes()).hexdigest()
                                         for name in config['eval_baseline_sha256']}
                if current_baseline_hashes!=config['eval_baseline_sha256']:
                    raise ValueError('Evaluation baseline changed during training')
                report=compare(args.eval_baseline,evaluation)
                atomic_json(round_dir/'comparison.json',report)
                update('evaluation_complete',last_evaluation=str(evaluation))
            update('ready',completed_updates=number,current_checkpoint=str(result/'checkpoint'),data_exhausted=collection_metrics.get('data_exhausted',False))
        update('data_exhausted' if state.get('data_exhausted') else 'training_budget_complete')
    except BaseException as exc:
        update('failed',error=str(exc));raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
