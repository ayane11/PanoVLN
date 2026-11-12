"""New native-model/environment rollout smoke test, not a benchmark."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import torch
import pyarrow.parquet as pq
from .backend import NativePolicy
from .episode import run_episode
from .simplevln import add_source_path


async def main_async(args):
    if args.output.exists(): raise FileExistsError(args.output)
    rows=pq.read_table(args.manifest).to_pylist()
    row=rows[args.episode_index]
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    policy=NativePolicy.load(args.checkpoint,dtype=torch.float32,attention='sdpa',max_response_tokens=64)
    add_source_path()
    from recipe.vln_simplevln.env_pool import VLNEnv
    decisions=0
    async def decide(instruction,frames):
        nonlocal decisions
        gen=policy.generate(instruction,frames)
        decisions+=1
        print(f'decision={decisions} frames={len(frames)} output={gen.action_text}',flush=True)
        return gen
    result=await run_episode(VLNEnv(args.server),row['extra_info'],decide,
                             group_uid='smoke',trajectory_uid='smoke:0',
                             max_env_steps=args.max_steps,actions_per_replan=4)
    for decision in result['decisions']: decision['generation']=asdict(decision['generation'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ('reward','metrics','done_reason','truncated')},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='checkpoints/PanoVLN')
    p.add_argument('--manifest',default='outputs/dapo_migration_v1/manifests/train.parquet')
    p.add_argument('--episode-index',type=int,default=0)
    p.add_argument('--server',default='http://127.0.0.1:8013')
    p.add_argument('--max-steps',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,required=True)
    asyncio.run(main_async(p.parse_args()))


if __name__=='__main__':main()
