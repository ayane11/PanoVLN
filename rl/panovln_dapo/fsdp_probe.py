"""Two-or-more GPU optimizer mechanics probe, NOT a DAPO reward experiment."""
import argparse
from datetime import timedelta
from functools import partial
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoProcessor
from src.qwen_vl import Qwen3_5ForConditionalGenerationForPanoVLN
from .backend import decode_frames,response_logprobs
from .inputs import build_inputs
from .algorithms import policy_loss_sum


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--checkpoint',default='checkpoints/PanoVLN')
    args=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    try:
        if args.output.exists():raise FileExistsError(args.output)
        print(f'rank={rank} load',flush=True)
        model=Qwen3_5ForConditionalGenerationForPanoVLN.from_pretrained(args.checkpoint,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa')
        device=torch.device('cuda',rank)
        model._ensure_panovggt_model(device,torch.float32)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train()
        # Replicate frozen geometry: its native hook changes device/dtype, so do
        # not shard those parameters. All trainable policy parameters are sharded.
        block_types={type(m) for m in model.modules() if type(m).__name__ in ('Qwen3_5DecoderLayer','Qwen3_5VisionBlock')}
        if not block_types: raise ValueError('No expected transformer blocks found')
        print(f'rank={rank} wrap {[c.__name__ for c in block_types]}',flush=True)
        model=FSDP(model,device_id=device,use_orig_params=True,
                   ignored_modules=[model.panovggt],
                   auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls=block_types))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-7,foreach=False)
        processor=AutoProcessor.from_pretrained(args.checkpoint,local_files_only=True)
        trajectory=json.loads(Path(args.trajectory).read_text())
        dec=trajectory['decisions'][0]; gen=dec['generation']
        inputs=build_inputs(processor,trajectory['instruction'],decode_frames(trajectory['image_buffer'][:dec['env_step_before']+1]))['inputs'].to(device)
        print(f'rank={rank} forward/backward',flush=True)
        logp=response_logprobs(model,inputs,gen['response_ids'])
        old=torch.tensor(gen['response_logprobs'],device=device)
        error=(logp.detach()-old).abs()
        if error.max()>1e-3 or error.mean()>1e-4:
            raise ValueError(f'FSDP probability mismatch: {error.max().item()} / {error.mean().item()}')
        # Deliberate unit advantage tests optimizer mechanics only. It must never
        # be reported as an online DAPO update or saved as a trained checkpoint.
        loss,n=policy_loss_sum(old,logp,torch.ones_like(logp),torch.ones_like(logp))
        (loss/n).backward()
        norm=model.clip_grad_norm_(1.0)
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradient norm')
        watched=[(name,param,param.detach().flatten()[:1024].clone()) for name,param in model.named_parameters() if param.requires_grad and param.numel() and param.grad is not None]
        print(f'rank={rank} optimizer step',flush=True)
        optimizer.step()
        changed=sum(int(not torch.equal(param.detach().flatten()[:1024],before)) for _,param,before in watched)
        local=dict(rank=rank,max_logprob_error=error.max().item(),mean_logprob_error=error.mean().item(),gradient_norm=float(norm),changed_parameter_samples=changed,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        reports=[None]*dist.get_world_size();dist.all_gather_object(reports,local)
        if not any(r['changed_parameter_samples'] for r in reports):raise ValueError('No parameter update')
        if rank==0:
            result={'scope':'optimizer mechanics with unit advantages; NOT DAPO training','ranks':reports}
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
