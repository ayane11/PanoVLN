"""Configurable single-policy sampling round with checkpoint cursor continuation."""
from .attention import resolve_attention
import argparse
import asyncio
from dataclasses import asdict
import json
import hashlib
import math
import shutil
from pathlib import Path
import time
import torch
import pyarrow.parquet as pq
from .backend import NativePolicy
from .episode import run_episode
from .sampling import EpisodeCursor,SamplingConfig,collect_round
from .simplevln import add_source_path


def write_collection_json(path, payload, *, min_free_gib=0):
    """Reserve export capacity before each potentially large trajectory write."""
    if not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError('Invalid disk reserve')
    data=(json.dumps(payload,ensure_ascii=False)+'\n').encode()
    if min_free_gib and shutil.disk_usage(path.parent).free < len(data)+min_free_gib*1024**3:
        raise RuntimeError('Collection disk reserve would be violated; no trajectory/batch was written')
    with path.open('xb') as stream:
        stream.write(data)


def restore_cursor(cursor, checkpoint):
    checkpoint = Path(checkpoint)
    progress_path = checkpoint/'vln_progress.json'
    if not progress_path.exists():
        if (checkpoint/'data.pt').exists() or (checkpoint/'optimizer.pt').exists():
            raise ValueError('Incomplete training checkpoint metadata')
        return 0
    progress = json.loads(progress_path.read_text())
    data = (checkpoint/'data.pt').read_bytes()
    if hashlib.sha256(data).hexdigest() != progress['data_sha256']:
        raise ValueError('Checkpoint data checksum mismatch')
    saved = torch.load(checkpoint/'data.pt', map_location='cpu', weights_only=False)
    if saved['policy_version'] != progress['policy_version']:
        raise ValueError('Checkpoint policy version mismatch')
    cursor.load_state_dict(saved['cursor'])
    return progress['policy_version']


def save_round(args,version,cursor,before,sampling,selected,metrics):
    payload=dict(precision=getattr(args,'precision','fp32'),policy_version=version,checkpoint=args.checkpoint,cursor_before=before,
                 cursor=cursor.state_dict(),sampling_config=asdict(sampling),metrics=metrics,
                 trajectories=[{**t,'decisions':[{**d,'generation':asdict(d['generation'])} for d in t['decisions']]} for t in selected])
    write_collection_json(args.output/'selected_batch.json',payload,min_free_gib=getattr(args,'min_free_gib',0))
    (args.output/'complete.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('ROUND COMPLETE '+json.dumps(metrics),flush=True)
    return payload


async def run(args, *, policy=None, cursor=None, policy_version=None):
    args.precision=getattr(args,'precision','fp32')
    args.output.mkdir(parents=True,exist_ok=False)
    config=vars(args).copy();config['output']=str(args.output)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if (cursor is None)!=(policy_version is None):
        raise ValueError('Resident cursor and policy version must be supplied together')
    if cursor is None:
        rows=pq.read_table(args.manifest).to_pylist()
        cursor=EpisodeCursor(rows,seed=args.seed,epochs=args.epochs)
        version=restore_cursor(cursor,args.checkpoint)
    else:
        if policy is None:raise ValueError('Resident collection requires its in-memory policy')
        version=policy_version
        if not isinstance(version,int) or version<0:raise ValueError('Invalid resident policy version')
    config.update(policy_version=version,cursor_before=cursor.state_dict(),
                  manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  dtype=args.precision,attention=resolve_attention(getattr(args,'attention_backend','auto'),args.precision),collision_recovery_steps=0)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    sampling=SamplingConfig(rollout_n=args.rollout_n,target_groups=args.target_groups,
                            generation_groups=args.generation_groups,max_generation_batches=args.max_generation_batches,
                            concurrency=getattr(args,"rollout_window",1),
                            scheduler=getattr(args,"rollout_scheduler","sliding"))
    if cursor.exhausted:
        selected,metrics=await collect_round(cursor,None,sampling,policy_version=version)
        return save_round(args,version,cursor,config['cursor_before'],sampling,selected,metrics)
    if policy is None:
        if args.precision=='bf16':
            from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
            if not qwen.is_fast_path_available: raise RuntimeError('BF16 RL requires FLA and causal-conv1d fast path')
        torch.set_num_threads(2);torch.manual_seed(args.seed+version)
        policy=NativePolicy.load(args.checkpoint,dtype=torch.bfloat16 if args.precision=='bf16' else torch.float32,attention=resolve_attention(getattr(args,'attention_backend','auto'),args.precision),max_response_tokens=args.max_response_tokens,temperature=args.temperature)
    add_source_path()
    from recipe.vln_simplevln.env_pool import VLNEnv
    start=time.monotonic();completed=0
    async def rollout(row,group_uid,trajectory_uid,version):
        nonlocal completed
        episode_start=time.monotonic();decisions=0
        async def decide(instruction,frames):
            nonlocal decisions
            gen=await inference.generate(instruction,frames);decisions+=1
            print(f'{trajectory_uid} decision={decisions} frame={len(frames)-1}',flush=True)
            return gen
        result=await run_episode(VLNEnv(args.server),row['extra_info'],decide,
                                 group_uid=group_uid,trajectory_uid=trajectory_uid,
                                 max_env_steps=args.max_env_steps,max_decisions=args.max_decisions,actions_per_replan=4,
                                 progress_coef=args.progress_coef)
        serialized={**result,'decisions':[{**d,'generation':asdict(d['generation'])} for d in result['decisions']]}
        filename=trajectory_uid.replace(':','_')+'.json'
        write_collection_json(args.output/filename,serialized,min_free_gib=args.min_free_gib)
        completed+=1
        metric=dict(completed=completed,group_uid=group_uid,trajectory_uid=trajectory_uid,
                    episode_id=result['episode_id'],scene_id=result['scene_id'],reward=result['reward'],
                    metrics=result['metrics'],done_reason=result['done_reason'],truncated=result['truncated'],
                    elapsed_seconds=time.monotonic()-episode_start,total_elapsed_seconds=time.monotonic()-start)
        with (args.output/'progress.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        print(json.dumps(metric),flush=True)
        return result
    def report_rollout_error(record):
        with (args.output/'failures.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print('ROLLOUT_FAILED '+json.dumps({k:v for k,v in record.items() if k!='traceback'}),flush=True)
    from .inference_queue import InferenceQueue
    async with InferenceQueue(policy,getattr(args,'inference_batch_size',1),
                              getattr(args,'inference_wait_ms',2.)) as inference:
        selected,metrics=await collect_round(cursor,rollout,sampling,policy_version=version,on_rollout_error=report_rollout_error)
        metrics.update(inference.metrics())
    return save_round(args,version,cursor,config['cursor_before'],sampling,selected,metrics)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attention-backend',choices=['auto','sdpa','flash_attention_2'],default='auto')
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--checkpoint',default='checkpoints/PanoVLN')
    p.add_argument('--manifest',default='outputs/dapo_migration_v1/manifests/train.parquet')
    p.add_argument('--server',default='http://127.0.0.1:8013')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--rollout-n',type=int,default=8)
    p.add_argument('--target-groups',type=int,default=128)
    p.add_argument('--generation-groups',type=int,default=192)
    p.add_argument('--max-generation-batches',type=int,default=16)
    p.add_argument('--inference-batch-size',type=int,default=1)
    p.add_argument('--inference-wait-ms',type=float,default=2.)
    p.add_argument('--rollout-window',type=int,default=1)
    p.add_argument('--rollout-scheduler',choices=['sliding','window'],default='sliding')
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--max-response-tokens',type=int,default=256)
    p.add_argument('--max-env-steps',type=int,default=200)
    p.add_argument('--max-decisions',type=int,default=64)
    p.add_argument('--progress-coef',type=float,default=0.)
    p.add_argument('--min-free-gib',type=float,default=0.,help='Reserve free disk space after each trajectory/batch write')
    args=p.parse_args()
    if not math.isfinite(args.min_free_gib) or args.min_free_gib<0:raise ValueError('Invalid disk reserve')
    try:asyncio.run(run(args))
    except BaseException as exc:
        if args.output.exists():
            import traceback
            (args.output/'failure.json').write_text(json.dumps({'error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n')
        raise


if __name__=='__main__': main()
