"""Full-episode on-policy sampling pilot. No optimizer updates or score claims."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
import pyarrow.parquet as pq
from .backend import NativePolicy
from .episode import run_episode
from .sampling import EpisodeCursor,SamplingConfig,collect_round
from .simplevln import add_source_path


async def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    config=vars(args).copy();config['output']=str(args.output)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows=pq.read_table(args.manifest).to_pylist()
    cursor=EpisodeCursor(rows,seed=args.seed,epochs=1)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    policy=NativePolicy.load(args.checkpoint,dtype=torch.float32,attention='sdpa',max_response_tokens=64)
    add_source_path()
    from recipe.vln_simplevln.env_pool import VLNEnv
    start=time.monotonic();completed=0
    async def rollout(row,group_uid,trajectory_uid,version):
        nonlocal completed
        episode_start=time.monotonic();decisions=0
        async def decide(instruction,frames):
            nonlocal decisions
            gen=policy.generate(instruction,frames);decisions+=1
            print(f'{trajectory_uid} decision={decisions} frame={len(frames)-1}',flush=True)
            return gen
        result=await run_episode(VLNEnv(args.server),row['extra_info'],decide,
                                 group_uid=group_uid,trajectory_uid=trajectory_uid,
                                 max_env_steps=500,max_decisions=500,actions_per_replan=4)
        serialized={**result,'decisions':[{**d,'generation':asdict(d['generation'])} for d in result['decisions']]}
        filename=trajectory_uid.replace(':','_')+'.json'
        (args.output/filename).write_text(json.dumps(serialized,ensure_ascii=False)+'\n')
        completed+=1
        metric=dict(completed=completed,group_uid=group_uid,trajectory_uid=trajectory_uid,
                    episode_id=result['episode_id'],scene_id=result['scene_id'],reward=result['reward'],
                    metrics=result['metrics'],done_reason=result['done_reason'],truncated=result['truncated'],
                    elapsed_seconds=time.monotonic()-episode_start,total_elapsed_seconds=time.monotonic()-start)
        with (args.output/'progress.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        print(json.dumps(metric),flush=True)
        return result
    sampling=SamplingConfig(rollout_n=4,target_groups=8,generation_groups=1,max_generation_batches=16,concurrency=1)
    selected,metrics=await collect_round(cursor,rollout,sampling,policy_version=0)
    payload=dict(policy_version=0,checkpoint=args.checkpoint,cursor=cursor.state_dict(),metrics=metrics,
                 trajectories=[{**t,'decisions':[{**d,'generation':asdict(d['generation'])} for d in t['decisions']]} for t in selected])
    (args.output/'selected_batch.json').write_text(json.dumps(payload,ensure_ascii=False)+'\n')
    (args.output/'complete.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('PILOT COMPLETE '+json.dumps(metrics),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='checkpoints/PanoVLN')
    p.add_argument('--manifest',default='outputs/dapo_migration_v1/manifests/train.parquet')
    p.add_argument('--server',default='http://127.0.0.1:8013')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:asyncio.run(run(args))
    except BaseException as exc:
        if args.output.exists():
            import traceback
            (args.output/'failure.json').write_text(json.dumps({'error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n')
        raise


if __name__=='__main__': main()
