"""One real-reward DAPO rollout-batch update, followed by atomic HF/optimizer export.

Launch with torchrun. The source batch must be a complete collect_round result.
All ranks do equal numbers of forward/backward calls; padding contributes zero.
PPO epochs/minibatches are configurable; this is not a complete experiment driver.
"""
import argparse
from collections import defaultdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import traceback
import faulthandler
import signal
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, StateDictType,
                                  FullStateDictConfig, FullOptimStateDictConfig)
from transformers import AutoProcessor
from .algorithms import policy_loss_sum
from .backend import decode_frames,response_logprobs,batch_response_logprobs
from .batch import decision_rows,token_count
from .distributed import load_actor
from .episode import Generation
from .inputs import build_inputs,ObservationConfig
from .optimizer import move_adamw_moments,step_adamw_with_parameter_offload,load_adamw_state_on_cpu
from .resume import validate_continuation,restore_rng



class ReplayFrames:
    """Decode only selected frames while preserving absolute history indices."""
    def __init__(self,buffer,count):
        self.buffer,self.count,self.cache=buffer,count,{}
    def __len__(self):return self.count
    def __getitem__(self,index):
        if index<0:index+=self.count
        if not 0<=index<self.count:raise IndexError(index)
        if index not in self.cache:self.cache[index]=decode_frames([self.buffer[index]])[0]
        return self.cache[index]


def replay(processor,row,device):
    gen=row['generation']
    observation=ObservationConfig(**gen.replay_metadata['observation'])
    prepared=build_inputs(processor,row['instruction'],ReplayFrames(row['image_buffer'],row['frame_count']),observation)
    if prepared['inputs']['input_ids'][0].tolist()!=gen.prompt_ids or prepared['image_indices']!=gen.image_indices:
        raise ValueError('Replay prompt/image selection changed')
    if prepared['prompt']!=gen.replay_metadata['raw_prompt']:
        raise ValueError('Replay template changed')
    return prepared['inputs'].to(device)


def load_batch(path):
    return parse_batch(json.loads(Path(path).read_text()))


def parse_batch(payload):
    """Validate an on-policy batch from disk or the resident collector."""
    if not payload.get('trajectories'):raise ValueError('No useful trajectories; must not update')
    groups=defaultdict(list)
    for trajectory in payload['trajectories']:
        groups[trajectory['group_uid']].append(trajectory)
    for group in groups.values():
        if len(group)<2:
            raise ValueError('Incomplete contrast group: at least two trajectories required')
        identities={(t.get('scene_id'),t.get('episode_id'),t['instruction']) for t in group}
        if len(identities)!=1:
            raise ValueError('Contrast group mixes different episodes')
    if 'metrics' in payload and payload['metrics']['valid_groups']!=len(groups):
        raise ValueError('Batch group count differs from collection metrics')
    for trajectory in payload['trajectories']:
        for decision in trajectory['decisions']:
            decision['generation']=Generation(**decision['generation'])
    rows=decision_rows(payload['trajectories'])
    if not any(abs(row['advantage'])>1e-8 for row in rows):raise ValueError('All advantages are zero')
    return payload,rows


def replay_microbatch(model, processor, rows, device, *, return_entropy=False):
    prompts=[replay(processor,row,device) for row in rows]
    generations=[row['generation'] for row in rows]
    if len(rows)==1:
        gen=generations[0]
        return [response_logprobs(model,prompts[0],gen.response_ids,
                gen.replay_metadata['temperature'],return_entropy=return_entropy)]
    return batch_response_logprobs(model,prompts,[g.response_ids for g in generations],
            [g.replay_metadata['temperature'] for g in generations],
            processor.tokenizer.pad_token_id,return_entropy=return_entropy)



class ActorSession:
    """Own FSDP parameters and Adam state across on-policy update batches.

    Construction restores optimizer/RNG once. update() preserves state in memory;
    exporting or ending a sampling round must not reconstruct this object.
    """
    def __init__(self,args,payload,rows,device,rank,world):
        self.completed_optimizer_steps=0
        self.policy_version=payload['policy_version']
        print(f'rank={rank} load actor, decisions={len(rows)}',flush=True)
        model=load_actor(args.checkpoint,device,compute_dtype=torch.bfloat16 if args.precision=='bf16' else torch.float32,
                         fsdp_size=args.fsdp_size,gradient_checkpointing=args.gradient_checkpointing,
                         forward_prefetch=args.forward_prefetch,use_torch_compile=args.use_torch_compile,
                         attention_backend=getattr(args,'attention_backend','auto'))
        processor=AutoProcessor.from_pretrained(args.checkpoint,local_files_only=True)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,foreach=False)
        rng_resume={'mode':'fresh'}
        if args.resume_optimizer:
            progress=json.loads((Path(args.checkpoint)/'vln_progress.json').read_text())
            if progress.get('precision','fp32')!=args.precision: raise ValueError('Resume precision changed')
            for key in ('lr','ppo_epochs','mini_batch_decisions','clip_low','clip_high','dual_clip'):
                if progress[key]!=getattr(args,key):raise ValueError(f'Resume configuration changed: {key}')
            if progress.get('entropy_coeff',0.)!=args.entropy_coeff:raise ValueError('Resume configuration changed: entropy_coeff')
            data_bytes=(Path(args.checkpoint)/'data.pt').read_bytes()
            if hashlib.sha256(data_bytes).hexdigest()!=progress['data_sha256']:
                raise ValueError('Checkpoint data cursor checksum mismatch')
            if progress['policy_version']!=payload['policy_version']:
                raise ValueError('Checkpoint policy version differs from rollout')
            if args.optimizer_offload and getattr(args,'optimizer_offload_mode','bulk')=='parameter':
                # CPU mmap is shared by the OS page cache; each rank selects its
                # own shards without aggregating all Adam moments on CUDA.
                full=torch.load(Path(args.checkpoint)/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
                sharded=FSDP.shard_full_optim_state_dict(full,model,optim=optimizer)
                load_adamw_state_on_cpu(optimizer,sharded)
            else:
                full=torch.load(Path(args.checkpoint)/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True) if rank==0 else None
                sharded=FSDP.scatter_full_optim_state_dict(full,model,optim=optimizer)
                optimizer.load_state_dict(sharded)
            del full,sharded
            saved=torch.load(Path(args.checkpoint)/'data.pt',map_location='cpu',weights_only=False)
            validate_continuation(saved,payload,world,allow_world_size_change=args.allow_world_size_change)
            rng_resume=restore_rng(saved,world,rank,allow_world_size_change=args.allow_world_size_change,seed=args.resume_seed)
        if args.optimizer_offload:move_adamw_moments(optimizer,'cpu')
        self.model,self.processor,self.optimizer=model,processor,optimizer
        self.rng_resume=rng_resume
        self.rank,self.world,self.device=rank,world,device

    def update(self,payload,rows,args):
        if payload['policy_version']!=self.policy_version:
            raise ValueError('Session received a stale or future policy batch')
        model,processor,optimizer=self.model,self.processor,self.optimizer
        rank,world,device=self.rank,self.world,self.device
        # Padding duplicates a real row only to match FSDP collective counts.
        assigned=[]
        for start in range(0,len(rows),args.mini_batch_decisions):
            end=min(start+args.mini_batch_decisions,len(rows))
            positions=list(range(start,end))
            positions += [0]*((-len(positions))%world)
            assigned.extend((positions[i],i<end-start) for i in range(rank,len(positions),world))
        old_values=[];maximum=total_error=0.;count=0
        model.eval()
        phase='probability_check'
        with torch.no_grad():
            for offset in range(0,len(assigned),args.log_prob_micro_batch_size):
                block=assigned[offset:offset+args.log_prob_micro_batch_size]
                block_rows=[rows[index] for index,_ in block]
                old_block=replay_microbatch(model,processor,block_rows,device)
                for (index,real),old in zip(block,old_block):
                    current_row=index;gen=rows[index]['generation']
                    if not torch.isfinite(old).all():
                        raise ValueError('Nonfinite recomputed old log probabilities')
                    error=(old-torch.tensor(gen.response_logprobs,device=device)).abs()
                    if real:
                        maximum=max(maximum,error.max().item());total_error+=error.sum().item();count+=error.numel()
                    old_values.append(old.cpu())
                del old_block,old
        errors=torch.tensor([maximum,total_error,count],dtype=torch.float64,device=device)
        max_error=errors[0].clone();dist.all_reduce(max_error,op=dist.ReduceOp.MAX)
        dist.all_reduce(errors[1:])
        if not torch.isfinite(errors).all() or not torch.isfinite(max_error) or (args.precision=='fp32' and (max_error.item()>1e-3 or errors[1].item()/max(errors[2].item(),1)>1e-4)):
            raise ValueError(f'Rollout/actor mismatch: max={max_error.item()}, mean={errors[1].item()/max(errors[2].item(),1)}')
        print(f'rank={rank} actor_replay_passed rollout_comparison={args.precision} max={max_error.item()} mean={errors[1].item()/max(errors[2].item(),1)}',flush=True)
        old_by_index={index:value for (index,_),value in zip(assigned,old_values)}
        del old_values
        model.train();loss_sum=0.;max_norm=0.;update_count=0;local_changes=0
        clip_sum=dual_sum=kl_sum=observed_tokens=entropy_sum=pg_sum=0.
        for epoch in range(args.ppo_epochs):
            for start in range(0,len(rows),args.mini_batch_decisions):
                end=min(start+args.mini_batch_decisions,len(rows))
                chunk=list(range(start,end));chunk += [0]*((-len(chunk))%world)
                denominator=token_count(rows[start:end])
                optimizer.zero_grad(set_to_none=True)
                local_positions=list(range(rank,len(chunk),world))
                for offset in range(0,len(local_positions),args.ppo_micro_batch_size):
                    positions=local_positions[offset:offset+args.ppo_micro_batch_size]
                    phase='training_forward'
                    values=replay_microbatch(model,processor,[rows[chunk[i]] for i in positions],device,
                                             return_entropy=bool(args.entropy_coeff))
                    micro_loss=None
                    for local,value in zip(positions,values):
                        index=chunk[local];real=local<(end-start)
                        current_row=index;row=rows[index];gen=row['generation']
                        logp,entropy=value if args.entropy_coeff else (value,None)
                        mask=torch.tensor(gen.response_mask,device=device,dtype=logp.dtype)*int(real)
                        advantage=torch.full_like(logp,row['advantage'])
                        old=old_by_index[index].to(device)
                        numerator,_=policy_loss_sum(old,logp,advantage,mask,args.clip_low,args.clip_high,args.dual_clip)
                        pg_sum+=float(numerator.detach())
                        if entropy is not None:
                            # Source verl applies response-token masks, not trajectory
                            # advantage weights, to the entropy regularizer.
                            entropy_numerator=(entropy*mask).sum()
                            entropy_sum+=float(entropy_numerator.detach())
                            numerator=numerator-args.entropy_coeff*entropy_numerator
                            del entropy_numerator
                        with torch.no_grad():
                            delta=(logp-old).clamp(-20,20);ratio=delta.exp()
                            raw=-advantage*ratio;clipped=-advantage*ratio.clamp(1-args.clip_low,1+args.clip_high)
                            upper=torch.maximum(raw,clipped)
                            clip_sum+=float(((clipped>raw)*mask).sum())
                            dual_sum+=float(((upper > -advantage*args.dual_clip)*(advantage<0)*mask).sum())
                            kl_sum+=float((-delta*mask).sum());observed_tokens+=float(mask.sum())
                        micro_loss=numerator if micro_loss is None else micro_loss+numerator
                        loss_sum+=float(numerator.detach())
                    phase='training_backward'
                    (micro_loss*world/denominator).backward()
                    del values,micro_loss,numerator,logp,entropy,value
                phase='optimizer_step'
                norm=model.clip_grad_norm_(1.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradient norm')
                max_norm=max(max_norm,float(norm))
                watched=[(p,p.detach().flatten()[:1024].clone()) for p in model.parameters() if p.requires_grad and p.numel() and p.grad is not None]
                if args.optimizer_offload and getattr(args,'optimizer_offload_mode','bulk')=='parameter':
                    step_adamw_with_parameter_offload(optimizer)
                else:
                    if args.optimizer_offload:move_adamw_moments(optimizer,device)
                    optimizer.step()
                local_changes+=sum(int(not torch.equal(p.detach().flatten()[:1024],old)) for p,old in watched)
                del watched
                if args.optimizer_offload:move_adamw_moments(optimizer,'cpu')
                update_count+=1
                print(f'rank={rank} PPO epoch={epoch} rows={start}:{end} norm={float(norm)}',flush=True)
        changes=torch.tensor(local_changes,device=device);dist.all_reduce(changes)
        if not changes.item():raise ValueError('Optimizer did not change sampled parameters')
        optimizer.zero_grad(set_to_none=True)
        stats=torch.tensor([loss_sum,clip_sum,dual_sum,kl_sum,observed_tokens,pg_sum,entropy_sum],dtype=torch.float64,device=device)
        dist.all_reduce(stats)
        metrics=dict(actor_loss=float(stats[0]/stats[4]),pg_loss=float(stats[5]/stats[4]),
                     entropy=float(stats[6]/stats[4]) if args.entropy_coeff else None,clip_fraction=float(stats[1]/stats[4]),
                     dual_clip_fraction=float(stats[2]/stats[4]),ppo_kl=float(stats[3]/stats[4]),policy_version=payload['policy_version']+1,decisions=len(rows),response_tokens=token_count(rows),
                     max_logprob_error=max_error.item(),mean_logprob_error=errors[1].item()/max(errors[2].item(),1),
                     max_gradient_norm=max_norm,optimizer_steps=update_count,changed_parameter_samples=int(changes.item()))
        self.policy_version=metrics['policy_version']
        self.completed_optimizer_steps+=update_count
        return metrics


    def export(self,args,payload,metrics):
        model,processor,optimizer=self.model,self.processor,self.optimizer
        rank,world,device=self.rank,self.world,self.device
        rng_resume=self.rng_resume
        args.output.parent.mkdir(parents=True,exist_ok=True)
        phase='checkpoint_export'
        print(f'rank={rank} export',flush=True)
        if args.optimizer_offload and getattr(args,'optimizer_offload_mode','bulk')=='bulk':
            move_adamw_moments(optimizer,device)
        with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True,rank0_only=True),
                                 FullOptimStateDictConfig(offload_to_cpu=True,rank0_only=True)):
            state=model.state_dict();optim_state=FSDP.optim_state_dict(model,optimizer)
        rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'python':random.getstate()}
        rngs=[None]*world;dist.all_gather_object(rngs,rng)
        if rank==0:
            temporary=args.output.with_name(args.output.name+'.incomplete')
            temporary.mkdir(parents=True,exist_ok=False)
            model.module.save_pretrained(temporary,state_dict=state,safe_serialization=True,max_shard_size='5GB')
            processor.save_pretrained(temporary)
            torch.save(optim_state,temporary/'optimizer.pt')
            torch.save({'cursor':payload['cursor'],'rng_by_rank':rngs,'policy_version':metrics['policy_version']},temporary/'data.pt')
            progress={**metrics,'data_sha256':hashlib.sha256((temporary/'data.pt').read_bytes()).hexdigest(),
                      'precision':args.precision,'probability_validation':'actor-replay-strict_rollout-diagnostic' if args.precision=='bf16' else 'fp32-strict','source_batch_sha256':hashlib.sha256(args.batch.read_bytes()).hexdigest(),
                      'lr':args.lr,'world_size':world,'rng_resume':rng_resume,'memory_limit_gib':args.memory_limit_gib,
                      'peak_allocated_bytes_rank0':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes_rank0':torch.cuda.max_memory_reserved(device),'ppo_epochs':args.ppo_epochs,'optimizer_offload':args.optimizer_offload,
                      'optimizer_offload_mode':getattr(args,'optimizer_offload_mode','bulk'),
                      'attention_backend':getattr(args,'attention_backend','auto'),
                      'use_torch_compile':args.use_torch_compile,'fsdp_size':args.fsdp_size,'gradient_checkpointing':args.gradient_checkpointing,
                      'forward_prefetch':args.forward_prefetch,'ppo_micro_batch_size':args.ppo_micro_batch_size,'log_prob_micro_batch_size':args.log_prob_micro_batch_size,
                      'mini_batch_decisions':args.mini_batch_decisions,'clip_low':args.clip_low,
                      'clip_high':args.clip_high,'dual_clip':args.dual_clip,'entropy_coeff':args.entropy_coeff,'scope':getattr(args,'export_scope','real-reward DAPO rollout-batch update')}
            (temporary/'vln_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            temporary.rename(args.output)
            print(json.dumps(progress,indent=2),flush=True)
        dist.barrier(group=getattr(self,'control_group',None))
        if args.optimizer_offload:move_adamw_moments(optimizer,'cpu')


def actor_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',required=True,type=Path)
    parser.add_argument('--attention-backend',choices=['auto','sdpa','flash_attention_2'],default='auto')
    parser.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--resume-optimizer',action='store_true')
    parser.add_argument('--allow-world-size-change',action='store_true')
    parser.add_argument('--resume-seed',type=int,default=42)
    parser.add_argument('--memory-limit-gib',type=float)
    parser.add_argument('--lr',type=float,default=5e-7)
    parser.add_argument('--optimizer-offload-mode',choices=['bulk','parameter'],default='bulk')
    parser.add_argument('--optimizer-offload',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--entropy-coeff',type=float,default=0.)
    parser.add_argument('--clip-low',type=float,default=.2)
    parser.add_argument('--clip-high',type=float,default=.28)
    parser.add_argument('--dual-clip',type=float,default=10.)
    parser.add_argument('--ppo-epochs',type=int,default=1)
    parser.add_argument('--use-torch-compile',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--fsdp-size',type=int,default=-1)
    parser.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--forward-prefetch',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--mini-batch-decisions',type=int,default=128)
    parser.add_argument('--ppo-micro-batch-size',type=int,default=1)
    parser.add_argument('--log-prob-micro-batch-size',type=int,default=1)
    return parser


def main():
    parser=actor_parser()
    args=parser.parse_args()
    if min(args.ppo_micro_batch_size,args.log_prob_micro_batch_size)<1:
        parser.error('Microbatch sizes must be positive')
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1,all_threads=True)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
    if args.memory_limit_gib is not None:
        total=torch.cuda.get_device_properties(rank).total_memory
        if not math.isfinite(args.memory_limit_gib) or not 0<args.memory_limit_gib*1024**3<=total:raise ValueError('Invalid memory limit')
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib*1024**3/total,rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    world=dist.get_world_size();device=torch.device('cuda',rank)
    phase='setup';current_row=None
    try:
        if args.output.exists():raise FileExistsError(args.output)
        if (Path(args.checkpoint)/'vln_progress.json').exists() and not args.resume_optimizer:
            raise ValueError('Trained checkpoint requires --resume-optimizer to preserve optimizer history')
        if args.ppo_epochs<1 or args.mini_batch_decisions<world:
            raise ValueError('Mini-batch decisions must be at least world size')
        if not all(math.isfinite(x) for x in (args.lr,args.clip_low,args.clip_high,args.dual_clip,args.entropy_coeff)) or args.lr<=0 or not 0<=args.clip_low<1 or args.clip_high<0 or args.dual_clip<=1 or args.entropy_coeff<0:
            raise ValueError('Invalid optimization hyperparameters')
        payload,rows=load_batch(args.batch)
        if payload.get('precision',args.precision)!=args.precision: raise ValueError('Rollout/actor precision configuration mismatch')
        if Path(payload['checkpoint']).resolve()!=Path(args.checkpoint).resolve():
            raise ValueError('Batch policy checkpoint differs from actor initialization')
        session=ActorSession(args,payload,rows,device,rank,world)
        metrics=session.update(payload,rows,args)
        model,processor,optimizer=session.model,session.processor,session.optimizer
        rng_resume=session.rng_resume
        session.export(args,payload,metrics)
    except BaseException:
        detail=traceback.format_exc()
        print(f'FATAL rank={rank} phase={phase} row={current_row}\n{detail}',file=sys.stderr,flush=True)
        try:
            failure={'rank':rank,'phase':phase,'row':current_row,'traceback':detail}
            (args.output.parent/f'failure_rank{rank}.json').write_text(json.dumps(failure,indent=2)+'\n')
        except Exception:
            traceback.print_exc()
        # NCCL shutdown can wait forever after another rank diverges. Let the
        # torchrun supervisor terminate peers; never hide the original error
        # behind destroy_process_group in an exception-finally block.
        os._exit(1)
    else:
        dist.destroy_process_group()


if __name__=='__main__':main()
