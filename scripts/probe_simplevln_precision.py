"""Two-GPU precision experiment; no checkpoint export or training restart."""
import argparse
import json
import os
import time
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['fp32', 'bf16_fla'], required=True)
    p.add_argument('--trajectory', required=True, type=Path)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--reference', type=Path, help='Reuse exact generated tokens and probabilities')
    a = p.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    dist.init_process_group('nccl')
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    if a.mode == 'fp32':
        # Restore the exact optional-kernel fallback used before installation.
        for key in ['causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule',
                    'fused_recurrent_gated_delta_rule', 'FusedRMSNormGated']:
            setattr(qwen, key, None)
        qwen.is_fast_path_available = False
    elif not qwen.is_fast_path_available:
        raise RuntimeError('FLA/causal-conv1d fast path unavailable')
    from panovln_dapo.backend import NativePolicy, response_logprobs
    from panovln_dapo.distributed import load_actor
    from panovln_dapo.inputs import build_inputs, ObservationConfig
    from panovln_dapo.actor import ReplayFrames
    from panovln_dapo.algorithms import policy_loss_sum
    from transformers import AutoProcessor
    row = json.loads(a.trajectory.read_text())
    frames = row['image_buffer'][:1]
    if rank == 0 and a.reference:
        a.output.mkdir(parents=True, exist_ok=False)
        (a.output/'generation.json').write_text(a.reference.read_text())
    if rank == 0 and not a.reference:
        a.output.mkdir(parents=True, exist_ok=False)
        policy = NativePolicy.load(a.checkpoint, device='cuda:0', attention='sdpa',
                                   dtype=torch.float32 if a.mode == 'fp32' else torch.bfloat16,
                                   max_response_tokens=64)
        torch.cuda.synchronize(); start = time.monotonic()
        generation = policy.generate(row['instruction'], frames)
        torch.cuda.synchronize(); elapsed = time.monotonic()-start
        from dataclasses import asdict
        (a.output/'generation.json').write_text(json.dumps(asdict(generation)))
        (a.output/'rollout.json').write_text(json.dumps({'seconds_including_first_kernel_compile': elapsed,
                                                       'tokens': len(generation.response_ids)}))
        del policy, generation
        import gc
        gc.collect(); torch.cuda.empty_cache()
    dist.barrier()
    generation = json.loads((a.output/'generation.json').read_text())
    model = load_actor(a.checkpoint, torch.device('cuda',rank),
                       compute_dtype=torch.float32 if a.mode=='fp32' else torch.bfloat16)
    processor = AutoProcessor.from_pretrained(a.checkpoint,local_files_only=True)
    prepared=build_inputs(processor,row['instruction'],ReplayFrames(frames,1),
                          ObservationConfig(**generation['replay_metadata']['observation']))
    assert prepared['inputs']['input_ids'][0].tolist()==generation['prompt_ids']
    inputs=prepared['inputs'].to(torch.device('cuda',rank))
    context=lambda: torch.autocast('cuda',dtype=torch.bfloat16) if a.mode=='bf16_fla' else nullcontext()
    expected_geometry=inputs['image_erp_geometry'].detach().clone()
    geometry_checks=[]
    def check_geometry(module, args, kwargs):
        geometry=kwargs['image_erp_geometry']
        assert geometry.dtype==torch.float32, 'FSDP changed geometry dtype'
        assert torch.equal(geometry,expected_geometry), 'FSDP changed geometry values'
        geometry_checks.append(True)
    # Runs inside the FSDP wrapper, after its root input-casting step.
    geometry_hook=model.module.register_forward_pre_hook(check_geometry,with_kwargs=True)
    model.eval()
    with torch.no_grad(),context():
        logp=response_logprobs(model,inputs,generation['response_ids'])
    error=(logp-torch.tensor(generation['response_logprobs'],device=rank)).abs()
    report={'mode':a.mode,'rank':rank,'max_logprob_error':error.max().item(),
            'mean_logprob_error':error.mean().item(),
            'ratio_min':torch.exp(logp-torch.tensor(generation['response_logprobs'],device=rank)).min().item(),
            'ratio_max':torch.exp(logp-torch.tensor(generation['response_logprobs'],device=rank)).max().item()}
    actor_old=logp.detach().clone()
    report['eval_logprobs']=logp.tolist()
    report['parameter_dtypes']={n:str(v.dtype) for n,v in model.named_parameters() if not 'panovggt.' in n}
    model.train(); times=[]
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); start=time.monotonic()
        with context():
            logp=response_logprobs(model,inputs,generation['response_ids'])
            report['train_eval_max_error']=(logp.detach()-actor_old).abs().max().item()
            numerator,count=policy_loss_sum(actor_old,
                                            logp,torch.ones_like(logp),torch.ones_like(logp),.2,.28,10.)
            loss=numerator/count
        loss.backward()
        norm=model.clip_grad_norm_(1.)
        torch.cuda.synchronize(); times.append(time.monotonic()-start)
        assert torch.isfinite(norm) and torch.isfinite(logp).all()
        assert all(p.grad is None for p in model.module.panovggt.parameters())
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-7,foreach=False)
    watched=[(p,p.detach().flatten()[:16].clone()) for p in model.parameters() if p.requires_grad and p.numel() and p.grad is not None]
    optimizer.step()
    changed=sum(not torch.equal(p.detach().flatten()[:16],value) for p,value in watched)
    assert geometry_checks
    geometry_hook.remove()
    report.update(geometry_input_exact=True,geometry_checks=len(geometry_checks),gradient_norm=float(norm),changed_parameter_samples=changed,forward_backward_seconds=times,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),geometry_frozen=True)
    (a.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='parameter_dtypes'}),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    main()
