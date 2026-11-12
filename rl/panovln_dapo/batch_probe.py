"""Real native Pano batch parity; synthetic observations, not navigation/FSDP validation."""
import argparse
from contextlib import nullcontext
import base64
import io
import json
import math
from pathlib import Path
import subprocess
import time
import torch
from PIL import Image
from .backend import NativePolicy,response_logprobs,batch_response_logprobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--shared-gpu',action='store_true')
    p.add_argument('--memory-limit-gib',type=float)
    p.add_argument('--history-counts',type=int,nargs='+',default=[1,3])
    p.add_argument('--attention',default='sdpa')
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='bfloat16')
    p.add_argument('--gradients',action='store_true')
    p.add_argument('--activation-offload',action='store_true',help='CPU saved-tensor storage during parity backward')
    p.add_argument('--atol',type=float,default=.05)
    p.add_argument('--rtol',type=float,default=.01)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if min(a.history_counts)<1 or len(a.history_counts)<2:p.error('Need at least two nonempty histories')
    if not all(math.isfinite(value) for value in (a.atol,a.rtol)) or a.atol<0 or a.rtol<0:p.error('Tolerances must be nonnegative')
    # Physical GPU IDs, not CUDA_VISIBLE_DEVICES-relative IDs. Do not touch an
    # occupied GPU or change another job to make room for this probe.
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
    if a.gpu not in used:raise ValueError('Unknown GPU')
    if a.shared_gpu:
        if a.memory_limit_gib is None or not math.isfinite(a.memory_limit_gib) or a.memory_limit_gib<=0:
            p.error('Shared GPU requires a positive memory limit')
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
        if free[a.gpu]<(a.memory_limit_gib+3)*1024:raise RuntimeError('Insufficient shared GPU budget plus 3 GiB headroom')
    elif used[a.gpu]>1000:raise RuntimeError(f'Requested GPU is not idle: {used}')
    torch.cuda.set_device(a.gpu);torch.set_num_threads(2);torch.manual_seed(42)
    if a.memory_limit_gib is not None:
        total=torch.cuda.get_device_properties(a.gpu).total_memory
        torch.cuda.set_per_process_memory_fraction(a.memory_limit_gib*1024**3/total,a.gpu)
    print('stage=load_model',flush=True)
    policy=NativePolicy.load(a.checkpoint,device=f'cuda:{a.gpu}',dtype=getattr(torch,a.dtype),attention=a.attention)
    image=Image.new('RGB',(1280,640),(80,100,120));buffer=io.BytesIO();image.save(buffer,format='JPEG')
    frame=base64.b64encode(buffer.getvalue()).decode()
    print('stage=prepare_inputs',flush=True)
    prompts=[policy.prepare('Walk forward and stop near the doorway.',[frame]*count)['inputs'] for count in a.history_counts]
    responses=[policy.processor.tokenizer.encode('STOP' if i%2 else 'FORWARD',add_special_tokens=False) for i in range(len(prompts))]
    temperatures=[1.]*len(prompts)
    def serial():return [response_logprobs(policy.model,prompt,ids) for prompt,ids in zip(prompts,responses)]
    def batched():return batch_response_logprobs(policy.model,prompts,responses,temperatures,policy.processor.tokenizer.pad_token_id)
    def timed(fn):
        torch.cuda.synchronize();start=time.perf_counter();values=fn();torch.cuda.synchronize()
        return values,time.perf_counter()-start
    report=dict(scope=__doc__,checkpoint=a.checkpoint,gpu=a.gpu,dtype=a.dtype,attention=a.attention,
                history_counts=a.history_counts,prompt_lengths=[row['input_ids'].shape[1] for row in prompts],
                atol=a.atol,rtol=a.rtol,gradients=a.gradients,activation_offload=a.activation_offload)
    try:
        print('stage=probability_parity',flush=True)
        with torch.no_grad():
            serial();batched()  # warm both paths, then measure with synchronization
            single,report['serial_seconds']=timed(serial)
            batch,report['batch_seconds']=timed(batched)
        report['max_logprob_error']=max((left-right).abs().max().item() for left,right in zip(single,batch))
        for left,right in zip(single,batch):torch.testing.assert_close(left,right,atol=a.atol,rtol=a.rtol)
        if a.gradients:
            print('stage=gradient_parity',flush=True)
            policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            policy.model.train()
            denominator=sum(map(len,responses))
            policy.model.zero_grad(set_to_none=True)
            def saved_context():
                return torch.autograd.graph.save_on_cpu(pin_memory=True) if a.activation_offload else nullcontext()
            for index,(prompt,ids) in enumerate(zip(prompts,responses)):
                print(f'stage=serial_backward sample={index}',flush=True)
                with saved_context():
                    (-response_logprobs(policy.model,prompt,ids).sum()/denominator).backward()
            reference={name:param.grad.detach().cpu().clone() for name,param in policy.model.named_parameters() if param.grad is not None}
            policy.model.zero_grad(set_to_none=True)
            print('stage=batch_backward',flush=True)
            with saved_context():
                (-sum(value.sum() for value in batched())/denominator).backward()
            print('stage=compare_gradients',flush=True)
            actual={name:param.grad for name,param in policy.model.named_parameters() if param.grad is not None}
            if reference.keys()!=actual.keys():raise ValueError('Gradient parameter coverage differs')
            if not reference:raise ValueError('No trainable gradient tensors were compared')
            report['compared_gradient_tensors']=len(reference)
            for name,expected in reference.items():
                torch.testing.assert_close(actual[name].detach().cpu(),expected,atol=a.atol,rtol=a.rtol,msg=lambda msg:f'{name}: {msg}')
        report['passed']=True
    except BaseException as error:
        report.update(passed=False,error=str(error))
        raise
    finally:
        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
