"""Configurable-rank restoration and real long-input training memory probe.

No checkpoint is exported. Probe optimizer changes are discarded on process exit.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig, FullOptimStateDictConfig
from transformers import AutoProcessor
from .actor import load_batch, replay
from .algorithms import policy_loss_sum
from .backend import response_logprobs
from .distributed import load_actor
from .optimizer import move_adamw_moments
from .resume import validate_continuation,restore_rng


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--batch',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--memory-limit-gib',type=float,default=24.)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--resume-seed',type=int,default=42)
    args=p.parse_args();rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if args.steps<2 or args.memory_limit_gib<=0:raise ValueError('Need two steps and a positive memory budget')
    torch.cuda.set_device(rank);torch.set_num_threads(2);device=torch.device('cuda',rank)
    total=torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib*1024**3/total,device)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    phase='setup';samples=[]
    def memory(label):
        torch.cuda.synchronize(device)
        free,total=torch.cuda.mem_get_info(device)
        entry=dict(phase=label,allocated=torch.cuda.memory_allocated(device),reserved=torch.cuda.memory_reserved(device),
                   peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device),free_bytes=free,total_bytes=total)
        samples.append(entry)
        (args.output/f'memory_rank{rank}.json').write_text(json.dumps(samples,indent=2)+'\n')
        print(f'rank={rank} '+json.dumps(entry),flush=True)
    try:
        if rank==0:args.output.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        payload,rows=load_batch(args.batch)
        if Path(payload['checkpoint']).resolve()!=args.checkpoint.resolve():raise ValueError('Batch checkpoint mismatch')
        progress=json.loads((args.checkpoint/'vln_progress.json').read_text())
        if hashlib.sha256((args.checkpoint/'data.pt').read_bytes()).hexdigest()!=progress['data_sha256']:raise ValueError('Data checksum mismatch')
        saved=torch.load(args.checkpoint/'data.pt',map_location='cpu',weights_only=False)
        validate_continuation(saved,payload,world,allow_world_size_change=True)
        phase='load_actor';memory(phase)
        model=load_actor(str(args.checkpoint),device)
        processor=AutoProcessor.from_pretrained(args.checkpoint,local_files_only=True)
        memory('model_loaded')
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=progress['lr'],foreach=False)
        phase='optimizer_scatter'
        full=torch.load(args.checkpoint/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True) if rank==0 else None
        sharded=FSDP.scatter_full_optim_state_dict(full,model,optim=optimizer)
        optimizer.load_state_dict(sharded);del sharded
        memory('optimizer_restored')
        phase='optimizer_roundtrip'
        with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True,rank0_only=True),
                                 FullOptimStateDictConfig(offload_to_cpu=True,rank0_only=True)):
            recovered=FSDP.optim_state_dict(model,optimizer)
        if rank==0:
            if full['param_groups']!=recovered['param_groups'] or full['state'].keys()!=recovered['state'].keys():raise ValueError('Restored optimizer mapping differs')
            for name,values in full['state'].items():
                if values.keys()!=recovered['state'][name].keys():raise ValueError('Restored optimizer fields differ')
                for key,value in values.items():
                    actual=recovered['state'][name][key]
                    if torch.is_tensor(value):
                        if value.dtype!=actual.dtype or not torch.equal(value,actual) or not torch.isfinite(actual).all():raise ValueError(f'Optimizer mismatch/nonfinite: {name}/{key}')
                    elif value!=actual:raise ValueError(f'Optimizer mismatch: {name}/{key}')
            (args.output/'optimizer_roundtrip.json').write_text(json.dumps(dict(all_states_bitwise_equal=True,state_parameters=len(full['state']),old_world_size=len(saved['rng_by_rank']),world_size=world),indent=2)+'\n')
        del full,recovered
        dist.barrier();move_adamw_moments(optimizer,'cpu');torch.cuda.empty_cache()
        rng=restore_rng(saved,world,rank,allow_world_size_change=True,seed=args.resume_seed)
        (args.output/f'rng_rank{rank}.json').write_text(json.dumps(rng,indent=2)+'\n')
        # Longest real prompt/response examples exercise the recorded batch's
        # largest sequence shapes; each rank receives one decision per step.
        indices=sorted(range(len(rows)),key=lambda i:len(rows[i]['generation'].prompt_ids)+len(rows[i]['generation'].response_ids),reverse=True)[:world*args.steps]
        if len(indices)!=world*args.steps:raise ValueError('Insufficient probe decisions')
        local=indices[rank::world];old_values={};max_error=0.;error_sum=0.;error_count=0
        phase='probability_check';model.eval()
        with torch.no_grad():
            for index in local:
                row=rows[index];gen=row['generation'];inputs=replay(processor,row,device)
                logp=response_logprobs(model,inputs,gen.response_ids,gen.replay_metadata['temperature'])
                error=(logp-torch.tensor(gen.response_logprobs,device=device)).abs()
                max_error=max(max_error,float(error.max()));error_sum+=float(error.sum());error_count+=error.numel();old_values[index]=logp.cpu()
                del inputs,logp
        error_tensor=torch.tensor(max_error,device=device);dist.all_reduce(error_tensor,op=dist.ReduceOp.MAX)
        error_totals=torch.tensor([error_sum,error_count],dtype=torch.float64,device=device);dist.all_reduce(error_totals)
        mean_error=float(error_totals[0]/error_totals[1])
        if not torch.isfinite(error_tensor) or not torch.isfinite(error_totals).all() or error_tensor.item()>1e-3 or mean_error>1e-4:raise ValueError(f'Probability error max={error_tensor.item()} mean={mean_error}')
        memory('probability_check_complete');model.train();norms=[]
        for step,index in enumerate(local):
            optimizer.zero_grad(set_to_none=True);phase=f'train_step_{step+1}'
            row=rows[index];gen=row['generation'];inputs=replay(processor,row,device)
            logp=response_logprobs(model,inputs,gen.response_ids,gen.replay_metadata['temperature'])
            mask=torch.tensor(gen.response_mask,device=device,dtype=logp.dtype)
            denominator=mask.sum().clone();dist.all_reduce(denominator)
            numerator,_=policy_loss_sum(old_values[index].to(device),logp,torch.full_like(logp,row['advantage']),mask,progress['clip_low'],progress['clip_high'],progress['dual_clip'])
            (numerator*world/denominator).backward();del inputs,logp,numerator
            memory(f'backward_{step+1}')
            norm=model.clip_grad_norm_(1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite probe gradient')
            norms.append(float(norm))
            move_adamw_moments(optimizer,device);memory(f'moments_on_gpu_{step+1}')
            optimizer.step();move_adamw_moments(optimizer,'cpu')
            memory(f'optimizer_step_{step+1}')
        optimizer.zero_grad(set_to_none=True)
        report=dict(rank=rank,steps=args.steps,decision_indices=local,prompt_lengths=[len(rows[i]['generation'].prompt_ids) for i in local],
                    gradient_norms=norms,max_logprob_error=float(error_tensor),mean_logprob_error=mean_error,memory=samples,rng=rng,
                    scope='Diagnostic updates discarded; not a committed DAPO training checkpoint')
        (args.output/f'result_rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
        dist.barrier()
        if rank==0:
            reports=[json.loads((args.output/f'result_rank{r}.json').read_text()) for r in range(world)]
            passed=all(min(s['free_bytes'] for s in r['memory'])>=2*1024**3 for r in reports)
            (args.output/'complete.json').write_text(json.dumps(dict(world_size=world,memory_limit_gib=args.memory_limit_gib,steps=args.steps,
                optimizer_roundtrip_passed=True,training_steps_passed=True,headroom_gate_passed=passed,
                min_observed_free_bytes=min(s['free_bytes'] for r in reports for s in r['memory'])),indent=2)+'\n')
            if not passed:raise RuntimeError('Observed shared memory headroom fell below 2 GiB')
        dist.barrier()
    except BaseException:
        detail=traceback.format_exc();print(detail,file=sys.stderr,flush=True)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/f'failure_rank{rank}.json').write_text(json.dumps(dict(rank=rank,phase=phase,traceback=detail),indent=2)+'\n')
        os._exit(1)
    else:dist.destroy_process_group()


if __name__=='__main__':main()
