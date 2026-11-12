"""Full-parameter FSDP serial/microbatch gradient parity on synthetic images."""
import argparse
import base64
from datetime import timedelta
import io
import json
import math
import os
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import AutoProcessor
from PIL import Image
from .distributed import load_actor
from .backend import response_logprobs,batch_response_logprobs
from .inputs import build_inputs,ObservationConfig


def gradient_statistics(reference, actual, *, chunk_elements=1_048_576):
    """Bounded-memory diagnostics; these do not change pass/fail tolerances."""
    if reference.shape!=actual.shape or chunk_elements<1:raise ValueError('Invalid gradient comparison')
    left,right=reference.reshape(-1),actual.reshape(-1)
    ref2=actual2=error2=dot=0.
    sign_disagreements=0
    for start in range(0,left.numel(),chunk_elements):
        x=left[start:start+chunk_elements].double()
        y=right[start:start+chunk_elements].double()
        ref2+=x.square().sum().item();actual2+=y.square().sum().item()
        error2+=(y-x).square().sum().item();dot+=(x*y).sum().item()
        sign_disagreements+=int((x.sign()!=y.sign()).sum().item())
    return dict(reference_squared_norm=ref2,actual_squared_norm=actual2,
                error_squared_norm=error2,dot=dot,elements=left.numel(),
                sign_disagreements=sign_disagreements)


def summarize_gradient_statistics(stats):
    ref2=stats['reference_squared_norm'];actual2=stats['actual_squared_norm']
    return dict(**stats,relative_l2=math.sqrt(stats['error_squared_norm']/ref2) if ref2 else None,
                cosine=stats['dot']/math.sqrt(ref2*actual2) if ref2 and actual2 else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--memory-limit-gib',type=float,default=28.)
    p.add_argument('--memory-limits-gib',type=float,nargs='+')
    p.add_argument('--pad-serial',action='store_true',help='Pad serial full sequences to batch width to isolate padding effects')
    p.add_argument('--repeat-serial',action='store_true',help='Measure serial repeat variability before batch comparison')
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--ppo-micro-batch-size',type=int,default=2)
    p.add_argument('--log-prob-micro-batch-size',type=int,default=2)
    p.add_argument('--atol',type=float,default=.001)
    p.add_argument('--rtol',type=float,default=.0001)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    world=int(os.environ['WORLD_SIZE'])
    if world>4:p.error('At most four GPUs are authorized')
    budgets=a.memory_limits_gib or [a.memory_limit_gib]*world
    if len(budgets)!=world or any(not math.isfinite(v) or v<=0 for v in budgets):p.error('Invalid memory budgets')
    if min(a.ppo_micro_batch_size,a.log_prob_micro_batch_size)<1:p.error('Invalid microbatch size')
    if any(not math.isfinite(v) or v<0 for v in (a.atol,a.rtol)):p.error('Invalid tolerances')
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
    device=torch.device('cuda',rank)
    budget=budgets[rank]
    free,total=torch.cuda.mem_get_info(device)
    if free<(budget+3)*1024**3:raise RuntimeError('Insufficient memory budget plus 3 GiB headroom')
    torch.cuda.set_per_process_memory_fraction(budget*1024**3/total,rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=20));torch.manual_seed(42)
    report=dict(scope=__doc__,world_size=dist.get_world_size(),atol=a.atol,rtol=a.rtol,rank=rank,memory_limit_gib=budget,precision=a.precision,ppo_micro_batch_size=a.ppo_micro_batch_size,
                log_prob_micro_batch_size=a.log_prob_micro_batch_size)
    def stage(name):
        report['stage']=name
        print(f'rank={rank} stage={name}',flush=True)
    try:
        stage('load_model')
        model=load_actor(a.checkpoint,device,compute_dtype=torch.bfloat16 if a.precision=='bf16' else torch.float32)
        processor=AutoProcessor.from_pretrained(a.checkpoint,local_files_only=True)
        cfg=model.module.config
        observation=ObservationConfig(top_crop_degrees=cfg.erp_top_crop_degrees,
            bottom_crop_degrees=cfg.erp_bottom_crop_degrees,panovggt_enabled=cfg.panovggt_enabled)
        image=Image.new('RGB',(1280,640),(80,100,120))
        count=max(a.ppo_micro_batch_size,a.log_prob_micro_batch_size)
        histories=[(1,3)[i%2] for i in range(count)]
        texts=[('FORWARD','STOP')[i%2] for i in range(count)]
        prompts=[build_inputs(processor,'Walk forward and stop near the doorway.',[image]*count,observation)['inputs'].to(device) for count in histories]
        responses=[processor.tokenizer.encode(text,add_special_tokens=False) for text in texts]
        def batch(start,size):
            stop=min(start+size,count)
            return batch_response_logprobs(model,prompts[start:stop],responses[start:stop],
                [1.]*(stop-start),processor.tokenizer.pad_token_id)
        serial_prompts=prompts
        if a.pad_serial:
            width=max(prompt['input_ids'].shape[1]+len(ids) for prompt,ids in zip(prompts,responses))
            serial_prompts=[]
            for prompt,ids in zip(prompts,responses):
                amount=width-prompt['input_ids'].shape[1]-len(ids)
                row=dict(prompt)
                for key in ('input_ids','attention_mask','mm_token_type_ids'):
                    if key in row:
                        fill=processor.tokenizer.pad_token_id if key=='input_ids' else 0
                        row[key]=torch.nn.functional.pad(row[key],(amount,0),value=fill)
                serial_prompts.append(row)
        report['serial_padded_to_batch_width']=a.pad_serial
        report['history_counts']=histories
        model.eval();stage('probability_parity')
        with torch.no_grad():
            reference=[response_logprobs(model,prompt,ids) for prompt,ids in zip(serial_prompts,responses)]
            actual=[value for start in range(0,count,a.log_prob_micro_batch_size)
                    for value in batch(start,a.log_prob_micro_batch_size)]
        report['max_logprob_error']=max((x-y).abs().max().item() for x,y in zip(reference,actual))
        probability_errors=[]
        for index,(x,y) in enumerate(zip(reference,actual)):
            try:torch.testing.assert_close(x,y,atol=a.atol,rtol=a.rtol)
            except AssertionError as error:probability_errors.append(dict(row=index,error=str(error)))
        report['probability_errors']=probability_errors
        del reference,actual
        model.train();model.zero_grad(set_to_none=True);denominator=sum(map(len,responses))
        for index,(prompt,ids) in enumerate(zip(serial_prompts,responses)):
            stage(f'serial_backward_{index}')
            (-response_logprobs(model,prompt,ids).sum()/denominator).backward()
        reference={name:param.grad.detach().cpu().clone() for name,param in model.named_parameters() if param.grad is not None}
        if a.repeat_serial:
            model.zero_grad(set_to_none=True);stage('repeat_serial_backward')
            for prompt,ids in zip(serial_prompts,responses):
                (-response_logprobs(model,prompt,ids).sum()/denominator).backward()
            repeated={name:param.grad for name,param in model.named_parameters() if param.grad is not None}
            if reference.keys()!=repeated.keys():raise ValueError('Repeat gradient coverage differs')
            repeat_stats={}
            for name,expected in reference.items():
                value=repeated[name].detach().cpu()
                group='vision' if '.visual.' in name else 'language_or_fusion'
                stats=gradient_statistics(expected,value)
                total=repeat_stats.setdefault(group,{key:0 for key in stats})
                for key,number in stats.items():total[key]+=number
            report['serial_repeat_statistics']={key:summarize_gradient_statistics(value) for key,value in repeat_stats.items()}
            del repeated,value
        model.zero_grad(set_to_none=True);stage('batch_backward')
        for start in range(0,count,a.ppo_micro_batch_size):
            (-sum(value.sum() for value in batch(start,a.ppo_micro_batch_size))/denominator).backward()
        actual={name:param.grad for name,param in model.named_parameters() if param.grad is not None}
        if reference.keys()!=actual.keys() or not reference:raise ValueError('Gradient coverage differs or is empty')
        errors=[];elements=0;maximum=0.;statistics={}
        for name,expected in reference.items():
            value=actual[name].detach().cpu();elements+=value.numel()
            group='vision' if '.visual.' in name else 'language_or_fusion'
            stats=gradient_statistics(expected,value)
            total=statistics.setdefault(group,{key:0 for key in stats})
            for key,number in stats.items():total[key]+=number
            if value.numel():maximum=max(maximum,(value-expected).abs().max().item())
            try:torch.testing.assert_close(value,expected,atol=a.atol,rtol=a.rtol)
            except AssertionError as error:errors.append(dict(parameter=name,error=str(error)))
        report.update(gradient_statistics={key:summarize_gradient_statistics(value) for key,value in statistics.items()},
                      compared_local_gradient_elements=elements,max_gradient_error=maximum,gradient_errors=errors,
                      passed=not errors and not probability_errors,peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        reports=[None]*dist.get_world_size();dist.all_gather_object(reports,report)
        if rank==0:
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(dict(passed=all(r['passed'] for r in reports),ranks=reports),indent=2)+'\n')
        if errors or probability_errors:
            raise ValueError(f'{len(errors)} local gradient tensors and {len(probability_errors)} probability rows failed')
        stage('passed');dist.barrier()
    except BaseException as error:
        report.update(passed=False,error=repr(error),peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.with_name(a.output.stem+f'_rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
        os._exit(1)
    else:
        dist.destroy_process_group()


if __name__=='__main__':main()
