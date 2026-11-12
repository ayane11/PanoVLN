"""Fresh deterministic evaluation using the same Pano input/execution contract.

Fail on any episode error. Results are never silently dropped. The HTTP service
must load the matching split. Primary metrics: SR, SPL, NE and oracle success.
"""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import random
from pathlib import Path
import torch
import pyarrow.parquet as pq
from .backend import NativePolicy
from .episode import Generation,run_episode
from .simplevln import add_source_path


class GreedyPolicy(NativePolicy):
    def generate(self,instruction,frames):
        prepared=self.prepare(instruction,frames);inputs=prepared['inputs']
        with torch.inference_mode():
            output=self.model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=64,
                                       eos_token_id=self.processor.tokenizer.eos_token_id,
                                       pad_token_id=self.processor.tokenizer.pad_token_id)
        ids=output[0,inputs['input_ids'].shape[-1]:].tolist()
        return Generation(self.processor.tokenizer.decode(ids,skip_special_tokens=True),
                          response_ids=ids,replay_metadata={'generation_capped':len(ids)==64})


async def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    rows=pq.read_table(args.manifest).to_pylist()
    if args.max_episodes: rows=random.Random(42).sample(rows,min(args.max_episodes,len(rows)))
    if not rows:raise ValueError('No evaluation episodes')
    splits={r['extra_info']['split'] for r in rows}
    if 'train' in splits:raise ValueError('Evaluation requires a validation split')
    config=vars(args).copy();config['output']=str(args.output)
    config.update(dtype='float32',attention='sdpa',do_sample=False,actions_per_replan=4,
                  stop_commit_max_actions=10,max_env_steps=500,max_decisions=500,
                  manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  episode_keys=[r['extra_info']['episode_key'] for r in rows])
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    torch.set_num_threads(2);torch.manual_seed(42)
    policy=GreedyPolicy.load(args.checkpoint,dtype=torch.float32,attention='sdpa')
    add_source_path()
    from recipe.vln_simplevln.env_pool import VLNEnv
    results=[]
    for index,row in enumerate(rows):
        async def decide(instruction,frames):return policy.generate(instruction,frames)
        trajectory=await run_episode(VLNEnv(args.server),row['extra_info'],decide,
                                     group_uid='evaluation',trajectory_uid=row['extra_info']['episode_key'])
        record={**row['extra_info'],**trajectory['metrics'],
                'done_reason':trajectory['done_reason'],'truncated':trajectory['truncated'],
                'invalid_outputs':sum(d['invalid_output'] for d in trajectory['decisions']),
                'generated_actions':[d['generation'].action_text for d in trajectory['decisions']],
                'executed_actions':[a for d in trajectory['decisions'] for a in d['executed_actions']]}
        with (args.output/'results.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        results.append(record)
        print(f'episode={index+1}/{len(rows)} success={record["success"]} spl={record["spl"]}',flush=True)
    summary={'episodes':len(results),'expected_episodes':len(rows),
             **{k:sum(r[k] for r in results)/len(results) for k in ['success','spl','distance_to_goal','oracle_success','env_steps','num_decisions']}}
    (args.output/'complete.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='checkpoints/PanoVLN')
    p.add_argument('--manifest',default='outputs/dapo_migration_v1/manifests/val_seen.parquet')
    p.add_argument('--server',default='http://127.0.0.1:8014')
    p.add_argument('--max-episodes',type=int,default=64)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.max_episodes<0:raise ValueError('Invalid episode limit')
    try:asyncio.run(run(args))
    except BaseException as exc:
        if args.output.exists():
            import traceback
            (args.output/'failure.json').write_text(json.dumps({'error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n')
        raise


if __name__=='__main__':main()
