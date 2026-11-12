"""Persistent FSDP actor + native sampler, configured by JSON, launched via torchrun.

Requires an already-running Habitat server. This entry owns no other jobs.
The legacy controller is intentionally kept available until GPU integration passes.
"""
import argparse
import asyncio
from contextlib import nullcontext
from datetime import timedelta
import fcntl
import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
import time
import traceback
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig
import pyarrow.parquet as pq
from .attention import resolve_attention
from .actor import ActorSession,actor_parser,load_batch
from .backend import NativePolicy
from .collect import run as collect_run,restore_cursor
from .sampling import EpisodeCursor
from .resident_loop import ResidentLoop
from .phase_offload import move_actor_between_phases
from .optimizer import move_adamw_moments
from .checkpoint_audit import audit
from .resume import restore_rng


def atomic_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def recover_export(config,root):
    """Adopt a complete atomic export after interruption before latest publication."""
    latest=root/'latest.json'
    previous=Path(json.loads(latest.read_text())['checkpoint']) if latest.exists() else Path(config['checkpoint']).resolve()
    prior_path=previous/'vln_progress.json'
    prior=json.loads(prior_path.read_text())['policy_version'] if prior_path.exists() else 0
    candidates=[]
    for candidate in root.glob('checkpoint_v*'):
        if re.fullmatch(r'checkpoint_v[0-9]+',candidate.name) is None:continue
        progress_path=candidate/'vln_progress.json'
        if not progress_path.is_file():continue
        progress=json.loads(progress_path.read_text())
        if progress['policy_version']>prior:candidates.append((candidate,progress))
    if len(candidates)>1:raise ValueError('Ambiguous uncommitted resident exports')
    if not candidates:return
    candidate,progress=candidates[0]
    if progress.get('previous_durable_policy_version')!=prior:
        raise ValueError('Uncommitted export has wrong previous durable version')
    source=Path(progress['resident_source_batch'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=progress['source_batch_sha256']:
        raise ValueError('Uncommitted export source batch changed')
    report=audit(previous,candidate,expected_updates=progress['policy_version']-prior)
    atomic_json(root/f'audit_v{progress["policy_version"]}.json',report)
    atomic_json(latest,dict(checkpoint=str(candidate),policy_version=progress['policy_version'],
                           optimizer_steps=progress['cumulative_optimizer_steps']))


def prune_committed(root,current,*,keep_last=2,keep_every=0,keep_rollouts=False):
    """Prune only audited resident artifacts after latest.json is committed."""
    root=Path(root);current=Path(current)
    if keep_last<0 or keep_every<0:raise ValueError('Invalid checkpoint retention')
    latest=json.loads((root/'latest.json').read_text())
    if Path(latest['checkpoint']).resolve()!=current.resolve():raise ValueError('Cannot prune before durable publication')
    if current.resolve().parent!=root.resolve():raise ValueError('Checkpoint is outside resident output')
    version=latest['policy_version'];checkpoints=[]
    for path in root.glob('checkpoint_v*'):
        match=re.fullmatch(r'checkpoint_v([0-9]+)',path.name)
        if match is None or path.is_symlink() or not path.is_dir():continue
        number=int(match.group(1))
        if number<=version and (root/f'audit_v{number}.json').is_file():checkpoints.append((number,path))
    checkpoints.sort();retained={path for _,path in (checkpoints[-keep_last:] if keep_last else checkpoints)}
    for number,path in checkpoints:
        if path!=current and path not in retained and not (keep_every and number%keep_every==0):
            shutil.rmtree(path)
    if not keep_rollouts:
        for directory in root.glob('collection_v*'):
            match=re.fullmatch(r'collection_v([0-9]+)_[0-9]+',directory.name)
            if match is None or int(match.group(1))>=version or directory.is_symlink():continue
            if not (directory/'complete.json').is_file():continue
            for path in directory.glob('*.json'):
                if path.name=='selected_batch.json' or re.fullmatch(r'v[0-9]+_[0-9]+_[0-9]+_r[0-9]+\.json',path.name):
                    path.unlink()


def prune_consumed_rollouts(root,current_batch):
    """Drop consumed cache copies; retain the pending export's source batch.

    Unexported live updates replay from the last durable checkpoint on failure,
    so older consumed payloads cannot be resumed independently.
    """
    root=Path(root);current_batch=Path(current_batch)
    match=re.fullmatch(r'collection_v([0-9]+)_[0-9]+',current_batch.parent.name)
    if match is None or current_batch.parent.parent.resolve()!=root.resolve():
        raise ValueError('Pending batch is outside this resident run')
    current_version=int(match.group(1))
    if not current_batch.is_file() or current_batch.is_symlink():raise ValueError('Pending source batch missing')
    for directory in root.glob('collection_v*'):
        match=re.fullmatch(r'collection_v([0-9]+)_[0-9]+',directory.name)
        if match is None or directory.is_symlink() or not directory.is_dir():continue
        if int(match.group(1))>current_version or not (directory/'complete.json').is_file():continue
        for path in directory.glob('*.json'):
            if path.is_symlink():continue
            if path.name=='selected_batch.json':
                if path.resolve()!=current_batch.resolve():path.unlink()
            elif re.fullmatch(r'v[0-9]+_[0-9]+_[0-9]+_r[0-9]+\.json',path.name):path.unlink()


class ResidentAdapter:
    def __init__(self,config,root,rank,world,device,control_group=None):
        self.config,self.root,self.rank,self.world,self.device=config,root,rank,world,device
        self.control_group=control_group
        self.actor=None;self.policy=None;self.batch_path=None
        self.logger=None
        self.exhausted=False
        self.previous=Path(config['checkpoint']).resolve()
        latest=root/'latest.json'
        if latest.exists():self.previous=Path(json.loads(latest.read_text())['checkpoint'])
        self.args=actor_parser().parse_args(['--batch','pending','--checkpoint',str(self.previous),'--output',str(root/'pending')])
        for key,value in config.get('actor',{}).items():
            if key not in vars(self.args) or key in ('checkpoint','batch','output'):raise ValueError(f'Invalid actor option {key}')
            setattr(self.args,key,value)
        budgets=config['memory_limit_gib']
        self.args.memory_limit_gib=budgets[rank] if isinstance(budgets,list) else budgets
        self.args.resume_optimizer=(self.previous/'vln_progress.json').exists()
        self.cursor=EpisodeCursor(pq.read_table(config['manifest']).to_pylist(),seed=config.get('seed',42),epochs=config.get('epochs',20))
        self.version=restore_cursor(self.cursor,self.previous)
        progress=json.loads((self.previous/'vln_progress.json').read_text()) if self.args.resume_optimizer else {}
        self.steps=progress.get('cumulative_optimizer_steps',progress.get('optimizer_steps',0))
        torch.manual_seed(config.get('seed',42));random.seed(config.get('seed',42))
        if self.args.resume_optimizer:
            saved=torch.load(self.previous/'data.pt',map_location='cpu',weights_only=False)
            restore_rng(saved,world,rank,allow_world_size_change=self.args.allow_world_size_change,seed=self.args.resume_seed)

    def broadcast_result(self,result):
        envelope=[result];dist.broadcast_object_list(envelope,src=0,group=self.control_group)
        if envelope[0].get('error'):raise RuntimeError(envelope[0]['error'])
        return envelope[0]

    def record_phase(self,phase,started,version,**extra):
        seconds=time.monotonic()-started
        if self.rank==0:
            record=dict(phase=phase,policy_version=version,wall_seconds=seconds,time=time.time(),**extra)
            with (self.root/'phase_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print('PHASE '+json.dumps(record),flush=True)
        return seconds

    def collect(self,version):
        started=time.monotonic()
        state=None
        if self.actor is not None:
            with FSDP.state_dict_type(self.actor.model,StateDictType.FULL_STATE_DICT,
                                     FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
                state=self.actor.model.state_dict()
            self.actor.model.zero_grad(set_to_none=True)
            if self.args.optimizer_offload:move_adamw_moments(self.actor.optimizer,'cpu')
            if self.config.get('param_offload',True):move_actor_between_phases(self.actor.model,'cpu')
        result={}
        if self.rank==0:
            try:
                if self.policy is None:
                    with torch.random.fork_rng(devices=[self.device.index]):
                        self.policy=NativePolicy.load(str(self.previous),device='cpu',
                            dtype=torch.bfloat16 if self.args.precision=='bf16' else torch.float32,attention=resolve_attention(self.args.attention_backend,self.args.precision))
                if state is not None:self.policy.model.load_state_dict(state,strict=True)
                del state
                self.policy.model.to(self.device).eval()
                options=dict(checkpoint=str(self.previous),manifest=self.config['manifest'],server=self.config['server'],
                    attention_backend=self.args.attention_backend,seed=self.config.get('seed',42),epochs=self.config.get('epochs',20),precision=self.args.precision,
                    rollout_n=8,target_groups=128,generation_groups=192,max_generation_batches=0,
                    temperature=1.,max_response_tokens=256,max_env_steps=200,max_decisions=64,progress_coef=0.,min_free_gib=0.)
                collection_options=self.config.get('collection',{})
                if set(collection_options)&{'checkpoint','manifest','output','precision','server'}:
                    raise ValueError('Collection overrides cannot change resident policy/data binding')
                options.update(collection_options)
                options['output']=self.root/f'collection_v{version}_{time.time_ns()}'
                args=SimpleNamespace(**options)
                self.policy.temperature=args.temperature;self.policy.max_response_tokens=args.max_response_tokens
                payload=asyncio.run(collect_run(args,policy=self.policy,cursor=self.cursor,policy_version=version))
                result=dict(batch=str(args.output/'selected_batch.json'),empty=not payload['trajectories'],
                            data_exhausted=payload['metrics']['data_exhausted'])
            except BaseException as error:result=dict(error=repr(error))
            finally:
                if self.policy is not None:self.policy.model.to('cpu')
                torch.cuda.empty_cache()
        result=self.broadcast_result(result)
        if self.actor is not None and self.config.get('param_offload',True):
            move_actor_between_phases(self.actor.model,self.device)
        self.exhausted=result['data_exhausted']
        if result['empty']:
            self.record_phase('collection_empty',started,version)
            return None
        self.batch_path=Path(result['batch'])
        payload,rows=load_batch(self.batch_path)
        self.rows=rows;self.args.batch=self.batch_path
        self.collection_seconds=self.record_phase('collection',started,version,decisions=len(rows),
            valid_groups=payload['metrics']['valid_groups'])
        return payload

    def update(self,payload):
        started=time.monotonic()
        if self.actor is None:
            rng=(torch.get_rng_state(),torch.cuda.get_rng_state(self.device),random.getstate())
            self.actor=ActorSession(self.args,payload,self.rows,self.device,self.rank,self.world)
            self.actor.control_group=self.control_group
            torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1],self.device);random.setstate(rng[2])
        metrics=self.actor.update(payload,self.rows,self.args)
        seconds=self.record_phase('update',started,metrics['policy_version'],
            decisions=metrics['decisions'],response_tokens=metrics['response_tokens'],
            optimizer_steps=metrics['optimizer_steps'])
        metrics.update(update_wall_seconds=seconds,collection_wall_seconds=self.collection_seconds,
            update_response_tokens_per_second=metrics['response_tokens']/seconds)
        if self.rank==0:
            atomic_json(self.root/f'update_v{metrics["policy_version"]}.json',metrics)
        if self.rank==0 and self.logger is not None:
            values={f'train/{key}':value for key,value in metrics.items() if isinstance(value,(int,float))}
            values.update({f'collection/{key}':value for key,value in payload['metrics'].items() if isinstance(value,(int,float))})
            self.logger.log(values,step=metrics['policy_version'])
        if self.rank==0 and not self.config.get('keep_rollouts',False):
            prune_consumed_rollouts(self.root,self.batch_path)
        return metrics

    def export(self,payload,metrics):
        started=time.monotonic()
        self.args.output=self.root/f'checkpoint_v{metrics["policy_version"]}'
        if self.rank==0:
            incomplete=self.args.output.with_name(self.args.output.name+'.incomplete')
            if incomplete.exists():
                incomplete.rename(self.root/f'.failed_export_v{metrics["policy_version"]}_{time.time_ns()}')
        dist.barrier()
        metrics=dict(metrics,resident_source_batch=str(self.batch_path))
        self.actor.export(self.args,payload,metrics)
        result={}
        if self.rank==0:
            try:
                report=audit(self.previous,self.args.output,expected_updates=metrics['policy_version']-metrics['previous_durable_policy_version'])
                atomic_json(self.root/f'audit_v{metrics["policy_version"]}.json',report)
                atomic_json(self.root/'latest.json',dict(checkpoint=str(self.args.output),
                    policy_version=metrics['policy_version'],optimizer_steps=metrics['cumulative_optimizer_steps']))
                prune_committed(self.root,self.args.output,keep_last=self.config.get('keep_last',2),
                    keep_every=self.config.get('keep_every',0),keep_rollouts=self.config.get('keep_rollouts',False))
            except BaseException as error:result=dict(error=repr(error))
        self.broadcast_result(result)
        self.previous=self.args.output
        self.record_phase('export_and_audit',started,metrics['policy_version'])
        if self.rank==0 and self.logger is not None:
            self.logger.summary['latest_checkpoint']=str(self.previous)
            self.logger.summary['durable_policy_version']=metrics['policy_version']
            self.logger.summary['durable_optimizer_steps']=metrics['cumulative_optimizer_steps']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    a=parser.parse_args();config=json.loads(a.config.read_text())
    root=Path(config['output']).resolve();root.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
    budgets=config['memory_limit_gib']
    if isinstance(budgets,list) and len(budgets)!=int(os.environ['WORLD_SIZE']):raise ValueError('One budget is required per rank')
    budget=budgets[rank] if isinstance(budgets,list) else budgets
    total=torch.cuda.get_device_properties(rank).total_memory
    if not isinstance(budget,(int,float)) or not 0<budget*1024**3<=total:raise ValueError('Invalid resident memory budget')
    free,_=torch.cuda.mem_get_info(rank)
    if free<(budget+3)*1024**3:raise RuntimeError('Insufficient resident budget plus 3 GiB headroom')
    if int(os.environ['WORLD_SIZE'])>4:raise ValueError('At most four GPUs are authorized')
    torch.cuda.set_per_process_memory_fraction(budget*1024**3/total,rank)
    timeout=timedelta(hours=config.get('collective_timeout_hours',48))
    dist.init_process_group('nccl',timeout=timeout)
    control_group=dist.new_group(backend='gloo',timeout=timeout)
    lock=None;logger=None
    try:
        if rank==0:
            lock=(root/'resident.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            scientific={k:v for k,v in config.items() if k not in ('memory_limit_gib','max_updates','save_freq')}
            scientific['runtime_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob('*.py'))}
            scientific['manifest_sha256']=hashlib.sha256(Path(config['manifest']).read_bytes()).hexdigest()
            manifest=root/'resident_config.json'
            if manifest.exists() and json.loads(manifest.read_text())!=scientific:raise ValueError('Resident configuration changed')
            atomic_json(manifest,scientific)
            recover_export(config,root)
        dist.barrier()
        adapter=ResidentAdapter(config,root,rank,dist.get_world_size(),torch.device('cuda',rank),control_group)
        if rank==0 and config.get('logger','wandb')=='wandb':
            import wandb
            run_path=root/'resident_run.json'
            if run_path.exists():run_id=json.loads(run_path.read_text())['id']
            else:
                run_id=config.get('wandb_id') or wandb.util.generate_id();atomic_json(run_path,{'id':run_id})
            log_dir=root/'wandb_offline';log_dir.mkdir(exist_ok=True)
            logger=wandb.init(project=config.get('wandb_project','panovln-dapo'),name=root.name,id=run_id,
                mode='offline',dir=str(log_dir),config=config)
            adapter.logger=logger
        loop=ResidentLoop(policy_version=adapter.version,optimizer_steps=adapter.steps,save_freq=config.get('save_freq',5))
        progress=loop.run(adapter.collect,adapter.update,adapter.export,max_updates=config.get('max_updates',0))
        if rank==0:
            atomic_json(root/'complete.json',dict(policy_version=progress.durable_version,
                optimizer_steps=progress.durable_optimizer_steps,checkpoint=str(adapter.previous),data_exhausted=adapter.exhausted))
    except BaseException:
        traceback.print_exc()
        if logger is not None:logger.finish(exit_code=1)
        if lock is not None:lock.close()
        # Exit this rank immediately so torchrun terminates peers blocked in a
        # collective. NCCL destruction on only one failed rank may deadlock.
        os._exit(1)
    else:
        if logger is not None:logger.finish()
        if lock is not None:lock.close()
        dist.destroy_process_group()


if __name__=='__main__':main()
