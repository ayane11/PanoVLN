"""Read-only native Pano probability decomposition; never exports checkpoints."""
import argparse, gc, json, time
from pathlib import Path
from contextlib import nullcontext
import torch

p=argparse.ArgumentParser()
p.add_argument('--kernels',choices=['fla','fallback'],required=True)
p.add_argument('--output',required=True)
p.add_argument('--frames',type=int,default=1)
p.add_argument('--modes',nargs='+',default=['bf16','fp32_autocast'])
a=p.parse_args()
torch.set_num_threads(2); torch.manual_seed(42)
import transformers.models.qwen3_5.modeling_qwen3_5 as q
if a.kernels=='fallback':
    for key in ['causal_conv1d_fn','causal_conv1d_update','chunk_gated_delta_rule','fused_recurrent_gated_delta_rule','FusedRMSNormGated']:
        setattr(q,key,None)
    q.is_fast_path_available=False
else: assert q.is_fast_path_available
from panovln_dapo.backend import NativePolicy,response_logprobs
from transformers import LogitsProcessor,LogitsProcessorList
root=Path('outputs/dapo_acceleration')
g=json.loads((root/'bf16_fla_01/generation.json').read_text())
r=json.loads(Path('outputs/dapo_continuous/continuous_main/round_00001/collection_001/v0_0_0_r0.json').read_text())
ids=g['response_ids']; out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
class ForceResponse(LogitsProcessor):
    def __call__(self,input_ids,scores):
        i=input_ids.shape[1]-prompt_length
        result=torch.full_like(scores,-float('inf')); result[:,ids[i]]=0
        return result
for mode in a.modes:
    policy=NativePolicy.load('checkpoints/PanoVLN',attention='sdpa',dtype=torch.float32 if mode in ('fp32','fp32_autocast') else torch.bfloat16)
    policy.model._ensure_panovggt_model(torch.device('cuda',0),torch.float32 if mode=='fp32' else torch.bfloat16)
    inputs=policy.prepare(r['instruction'],r['image_buffer'][:a.frames])['inputs']
    prompt_length=inputs['input_ids'].shape[1]
    if a.frames==1: assert inputs['input_ids'][0].tolist()==g['prompt_ids']
    if mode.startswith('cast_'):
        key=mode.removeprefix('cast_')
        inputs[key]=inputs[key].to(torch.bfloat16)
    if mode=='bf16_autocast_castinputs':
        inputs={k:v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v for k,v in inputs.items()}
    ctx=lambda: torch.autocast('cuda',dtype=torch.bfloat16) if mode not in ('bf16','fp32') else nullcontext()
    result={'mode':mode,'kernels':a.kernels,'frames':a.frames,'prompt_length':prompt_length,'parameters':{n:str(v.dtype) for n,v in policy.model.named_parameters() if not n.startswith('panovggt.')}}
    head_records=[]
    def capture_head(module,args,output):
        # Diagnostic only: recompute the same output projection in FP32.
        # Do not change returned activations or model generation.
        with torch.autocast('cuda',enabled=False):
            raw=torch.nn.functional.linear(args[0].float(),module.weight.float(),
                module.bias.float() if module.bias is not None else None)
        head_records.append((output.detach().float(),raw))
    head_hook=policy.model.lm_head.register_forward_hook(capture_head)
    with torch.inference_mode(),ctx():
        full=response_logprobs(policy.model,inputs,ids)
        generated=policy.model.generate(**inputs,do_sample=False,max_new_tokens=len(ids),
            logits_processor=LogitsProcessorList([ForceResponse()]),return_dict_in_generate=True,output_logits=True,
            eos_token_id=None,pad_token_id=policy.processor.tokenizer.pad_token_id)
        assert generated.sequences[0,-len(ids):].tolist()==ids
        cached=torch.stack([score[0].float().log_softmax(-1)[token] for score,token in zip(generated.logits,ids)])
        repeat=response_logprobs(policy.model,inputs,ids)
    head_hook.remove()
    full_raw,full_fp32=head_records[0]
    cache_raw=torch.cat([v[0] for v in head_records[1:-1]],dim=1)
    cache_fp32=torch.cat([v[1] for v in head_records[1:-1]],dim=1)
    targets=torch.tensor(ids,device=full_raw.device).view(1,-1,1)
    def chosen(values):
        return values.log_softmax(-1).gather(-1,targets).flatten()
    full_fp32_lp=chosen(full_fp32[:,:-1])
    cache_fp32_lp=chosen(cache_fp32)
    result['fp32_projection_cached_full_max']=(full_fp32_lp-cache_fp32_lp).abs().max().item()
    result['bf16_selected_logit_deltas']=(cache_raw.gather(-1,targets)-full_raw[:,:-1].gather(-1,targets)).flatten().tolist()
    del head_records,full_raw,full_fp32,cache_raw,cache_fp32
    result.update(full=full.tolist(),cached=cached.tolist(),repeat=repeat.tolist(),
                  cached_full_max=(full-cached).abs().max().item(),repeat_error=(full-repeat).abs().max().item())
    (out/f'{mode}.json').write_text(json.dumps(result,indent=2))
    print(mode,a.kernels,'cached_full_max',result['cached_full_max'],'repeat_error',result['repeat_error'],flush=True)
    del policy,inputs,full,cached,generated,repeat
    gc.collect();torch.cuda.empty_cache()
