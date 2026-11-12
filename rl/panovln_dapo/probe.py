"""Small real-model probability probe; synthetic images, no navigation claims."""
import argparse
import base64
import io
import json
import time
from pathlib import Path
import torch
from PIL import Image
from .backend import NativePolicy,response_logprobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='checkpoints/PanoVLN')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--max-tokens',type=int,default=8)
    parser.add_argument('--image',type=Path)
    parser.add_argument('--history-count',type=int,default=1)
    parser.add_argument('--instruction',default='Walk forward and stop near the doorway.')
    parser.add_argument('--dtype',choices=['bfloat16','float32'],default='bfloat16')
    parser.add_argument('--attention',default='flash_attention_2')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.manual_seed(42)
    start=time.monotonic()
    print('Loading full Pano model',flush=True)
    policy=NativePolicy.load(args.checkpoint,max_response_tokens=args.max_tokens,dtype=getattr(torch,args.dtype),attention=args.attention)
    image=Image.open(args.image).convert('RGB') if args.image else Image.new('RGB',(1280,640),(80,100,120))
    buffer=io.BytesIO();image.save(buffer,format='JPEG')
    frames=[base64.b64encode(buffer.getvalue()).decode()]*args.history_count
    print('Generating response',flush=True)
    gen=policy.generate(args.instruction,frames)
    print('Recomputing response likelihood',flush=True)
    prepared=policy.prepare(args.instruction,frames)
    with torch.no_grad():
        recomputed=response_logprobs(policy.model,prepared['inputs'],gen.response_ids,policy.temperature)
    original=torch.tensor(gen.response_logprobs,device=recomputed.device)
    error=(recomputed-original).abs()
    result=dict(scope='probability probe, not navigation/training',image=str(args.image),history_count=args.history_count,dtype=args.dtype,attention=args.attention,
                response=gen.action_text,response_ids=gen.response_ids,
                rollout_logprobs=gen.response_logprobs,recomputed_logprobs=recomputed.tolist(),
                max_error=error.max().item(),mean_error=error.mean().item(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-start,
                panovggt_enabled=bool(policy.model.config.panovggt_enabled))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
