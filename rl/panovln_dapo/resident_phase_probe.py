"""Real-model continuous-update/phase migration probe; synthetic decisions, no reward claims."""
import argparse
import base64
from datetime import timedelta
import io
import gc
from dataclasses import asdict
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig
from PIL import Image
from .actor import ActorSession,actor_parser,replay_microbatch
from .backend import NativePolicy
from .episode import Generation
from .inputs import build_inputs,ObservationConfig
from .phase_offload import move_actor_between_phases
from .sampling import EpisodeCursor
from .collect import restore_cursor
from .checkpoint_audit import audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--optimizer-offload-mode',choices=['bulk','parameter'],default='bulk')
    p.add_argument('--checkpoint-test-root',type=Path)
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--save-freq',type=int,default=5)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--memory-limits-gib',nargs='+',type=float,required=True)
    a=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world>4 or len(a.memory_limits_gib)!=world:raise ValueError('Invalid GPU count/budgets')
    torch.cuda.set_device(rank);torch.set_num_threads(2);torch.manual_seed(42)
    device=torch.device('cuda',rank);budget=a.memory_limits_gib[rank]
    free,total=torch.cuda.mem_get_info(device)
    if not math.isfinite(budget) or budget<=0 or free<(budget+3)*1024**3:
        raise RuntimeError('Insufficient GPU budget plus 3 GiB headroom')
    torch.cuda.set_per_process_memory_fraction(budget*1024**3/total,rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    control_group=dist.new_group(backend='gloo',timeout=timedelta(minutes=30))
    args=actor_parser().parse_args(['--batch','synthetic','--checkpoint',a.checkpoint,'--output','unused',
        '--precision','bf16','--mini-batch-decisions',str(world)])
    args.memory_limit_gib=budget
    args.optimizer_offload_mode=a.optimizer_offload_mode
    args.resume_optimizer=(Path(a.checkpoint)/'vln_progress.json').exists()
    args.export_scope=__doc__
    if a.updates<1 or a.save_freq<1:raise ValueError('Invalid update/save interval')
    report=dict(scope=__doc__,rank=rank,updates=[],passed=False)
    stage='load_model'
    def mark(value):
        nonlocal stage
        stage=value;print(f'rank={rank} stage={stage} allocated={torch.cuda.memory_allocated(device)} reserved={torch.cuda.memory_reserved(device)}',flush=True)
    try:
        mark('load_model')
        cursor=EpisodeCursor(list(range(100)),seed=42,epochs=2,shuffle=False)
        initial_version=restore_cursor(cursor,a.checkpoint)
        before=cursor.state_dict();cursor.take(1)
        payload={'policy_version':initial_version,'precision':'bf16','checkpoint':a.checkpoint,
                 'cursor_before':before,'cursor':cursor.state_dict()}
        previous=Path(a.checkpoint)
        durable_version=initial_version
        interval_steps=0
        report['exports']=[]
        session=ActorSession(args,payload,[],device,rank,world)
        session.control_group=control_group
        report['initial_policy_version']=initial_version
        report['cursor_before']=before
        report['rng_resume']=session.rng_resume
        if args.resume_optimizer:
            saved=torch.load(Path(a.checkpoint)/'data.pt',map_location='cpu',weights_only=False)
            rng=saved['rng_by_rank'][rank]
            checks={
                'cursor':saved['cursor']==before,
                'torch_rng':torch.equal(torch.get_rng_state(),rng['torch']),
                'cuda_rng':torch.equal(torch.cuda.get_rng_state(device),rng['cuda']),
                'python_rng':random.getstate()==rng['python'],
            }
            steps_before=[int(state['step'].item()) for state in session.optimizer.state.values() if 'step' in state]
            checks['adam_step']=bool(steps_before) and min(steps_before)==max(steps_before)==initial_version
            report['resume_checks']=checks
            report['adam_step_before']=min(steps_before) if steps_before else None
            if not all(checks.values()):raise ValueError(f'Resume state mismatch: {checks}')
            del saved,rng
        cfg=session.model.module.config
        observation=ObservationConfig(top_crop_degrees=cfg.erp_top_crop_degrees,
            bottom_crop_degrees=cfg.erp_bottom_crop_degrees,panovggt_enabled=cfg.panovggt_enabled)
        image=Image.new('RGB',(1280,640),(80,100,120));stream=io.BytesIO();image.save(stream,format='JPEG')
        frame=base64.b64encode(stream.getvalue()).decode();rows=[]
        instruction='Walk forward and stop near the doorway.'
        for count,text,advantage in ((1,'FORWARD',1.),(3,'STOP',-1.)):
            prepared=build_inputs(session.processor,instruction,[image]*count,observation)
            response=session.processor.tokenizer.encode(text,add_special_tokens=False)
            gen=Generation(action_text=text,prompt_ids=prepared['inputs']['input_ids'][0].tolist(),
                response_ids=response,response_mask=[1]*len(response),image_indices=prepared['image_indices'],
                replay_metadata=dict(temperature=1.,raw_prompt=prepared['prompt'],observation=vars(observation)))
            rows.append(dict(instruction=instruction,image_buffer=[frame]*count,frame_count=count,
                             generation=gen,advantage=advantage))
        sampler=None
        for index in range(a.updates):
            version=initial_version+index
            if index:
                payload['cursor_before']=cursor.state_dict();cursor.take(1)
                payload['cursor']=cursor.state_dict()
            mark(f'update_{version+1}')
            payload['policy_version']=version
            session.model.eval()
            with torch.no_grad():
                values=[replay_microbatch(session.model,session.processor,[row],device)[0] for row in rows]
            for row,value in zip(rows,values):row['generation'].response_logprobs=value.tolist()
            del values
            metrics=session.update(payload,rows,args);report['updates'].append(metrics)
            interval_steps+=metrics['optimizer_steps']
            if a.checkpoint_test_root is not None and ((index+1)%a.save_freq==0 or index+1==a.updates):
                mark('checkpoint_export')
                root=a.checkpoint_test_root;root.mkdir(parents=True,exist_ok=True)
                args.batch=root/f'batch_v{version}.json'
                if rank==0:
                    args.batch.write_text(json.dumps({**payload,'synthetic_probe':True,'rows':[{**row,'generation':asdict(row['generation'])} for row in rows]})+'\n')
                dist.barrier()
                args.output=root/f'checkpoint_v{version+1}'
                export_metrics=dict(metrics,optimizer_steps=interval_steps,previous_durable_policy_version=durable_version,
                                    cumulative_optimizer_steps=version+1,synthetic_probe=True)
                session.export(args,payload,export_metrics)
                mark('checkpoint_audit')
                result=[{}]
                if rank==0:
                    try:
                        result[0]=audit(previous,args.output,expected_updates=version+1-durable_version)
                        (root/f'audit_v{version+1}.json').write_text(json.dumps(result[0],indent=2)+'\n')
                    except BaseException as error:result[0]={'error':repr(error)}
                dist.broadcast_object_list(result,src=0,group=control_group)
                if result[0].get('error'):raise RuntimeError(result[0]['error'])
                report['exports'].append(str(args.output))
                previous=args.output;durable_version=version+1;interval_steps=0
            if index==0:
                mark('gather_state')
                with FSDP.state_dict_type(session.model,StateDictType.FULL_STATE_DICT,
                                         FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
                    state=session.model.state_dict()
                mark('offload_actor')
                move_actor_between_phases(session.model,'cpu')
                outcome=[{}]
                if rank==0:
                    try:
                        mark('sampler_weight_sync')
                        with torch.random.fork_rng(devices=[rank]):
                            sampler=NativePolicy.load(a.checkpoint,device='cpu',dtype=torch.bfloat16,attention='sdpa',max_response_tokens=4)
                        sampler.model.load_state_dict(state,strict=True);del state
                        sampler.model.to(device).eval();mark('sampler_generate')
                        generations=sampler.generate_batch([(instruction,[frame]),(instruction,[frame]*3)])
                        if len(generations)!=2 or any(not g.response_ids or not all(math.isfinite(p) for p in g.response_logprobs) for g in generations):
                            raise ValueError('Invalid batched sampler responses')
                        report['generated_response_lengths']=[len(g.response_ids) for g in generations]
                    except BaseException as error:outcome[0]={'error':repr(error)}
                    finally:
                        if sampler is not None:sampler.model.to('cpu')
                        torch.cuda.empty_cache()
                dist.broadcast_object_list(outcome,src=0,group=control_group)
                if outcome[0].get('error'):raise RuntimeError(outcome[0]['error'])
                if rank==0:
                    del sampler;sampler=None;gc.collect()
                mark('reload_actor');move_actor_between_phases(session.model,device)
        steps=[int(value['step'].item()) for value in session.optimizer.state.values() if 'step' in value]
        if not steps or min(steps)!=initial_version+a.updates or max(steps)!=initial_version+a.updates:raise ValueError('Adam step state did not persist across phases')
        report.update(passed=True,adam_step_min=min(steps),adam_step_max=max(steps))
        mark('passed')
    except BaseException as error:
        report.update(error=repr(error),failed_stage=stage,peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.with_name(a.output.stem+f'_rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
        # Let torchrun terminate peers blocked in all_reduce; a failed rank must
        # not wait alone in destroy_process_group.
        os._exit(1)
    else:
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.with_name(a.output.stem+f'_rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
        dist.destroy_process_group()


if __name__=='__main__':main()
