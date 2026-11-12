"""Real FSDP optimizer/RNG round trip; does not perform a policy update."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig,FullOptimStateDictConfig
from .distributed import load_actor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    try:
        if args.output.exists():raise FileExistsError(args.output)
        progress=json.loads((args.checkpoint/'vln_progress.json').read_text())
        data_bytes=(args.checkpoint/'data.pt').read_bytes()
        if hashlib.sha256(data_bytes).hexdigest()!=progress['data_sha256']:raise ValueError('Data/RNG checksum mismatch')
        saved=torch.load(args.checkpoint/'data.pt',map_location='cpu',weights_only=False)
        if len(saved['rng_by_rank'])!=dist.get_world_size():raise ValueError('RNG world size changed')
        if saved['policy_version']!=progress['policy_version']:raise ValueError('Policy version mismatch')
        print(f'rank={rank} loading model and optimizer',flush=True)
        model=load_actor(str(args.checkpoint),torch.device('cuda',rank))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=progress['lr'],foreach=False)
        full=torch.load(args.checkpoint/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True) if rank==0 else None
        sharded=FSDP.scatter_full_optim_state_dict(full,model,optim=optimizer)
        optimizer.load_state_dict(sharded);del sharded
        print(f'rank={rank} optimizer restored; gathering for exact comparison',flush=True)
        with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True,rank0_only=True),
                                 FullOptimStateDictConfig(offload_to_cpu=True,rank0_only=True)):
            restored=FSDP.optim_state_dict(model,optimizer)
        if rank==0:
            if full['param_groups']!=restored['param_groups']:raise ValueError('Optimizer parameter groups changed')
            if full['state'].keys()!=restored['state'].keys():raise ValueError('Optimizer parameter state keys changed')
            for name,states in full['state'].items():
                if states.keys()!=restored['state'][name].keys():raise ValueError(f'Optimizer fields changed: {name}')
                for key,value in states.items():
                    actual=restored['state'][name][key]
                    if torch.is_tensor(value):
                        if value.dtype!=actual.dtype or not torch.equal(value,actual):raise ValueError(f'Optimizer tensor changed: {name}/{key}')
                    elif value!=actual:raise ValueError(f'Optimizer value changed: {name}/{key}')
            state_count=len(full['state'])
            steps=sorted({float(v['step']) for v in full['state'].values()})
            del full,restored
        rng=saved['rng_by_rank'][rank]
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python'])
        if not torch.equal(torch.get_rng_state(),rng['torch']) or not torch.equal(torch.cuda.get_rng_state(),rng['cuda']) or random.getstate()!=rng['python']:
            raise ValueError('RNG restoration failed')
        dist.barrier()
        if rank==0:
            report={'scope':'Distributed optimizer and RNG restoration only; no policy update',
                    'checkpoint':str(args.checkpoint),'policy_version':progress['policy_version'],
                    'state_parameters':state_count,'optimizer_steps':steps,
                    'all_optimizer_states_bitwise_equal':True,'all_rank_rng_restored':True,
                    'cursor':saved['cursor'],'world_size':dist.get_world_size()}
            with args.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2),flush=True)
        dist.barrier()
    except BaseException:
        detail=traceback.format_exc();print(detail,file=sys.stderr,flush=True)
        args.output.with_name(args.output.stem+f'.failure_rank{rank}.json').write_text(json.dumps({'rank':rank,'traceback':detail},indent=2)+'\n')
        os._exit(1)
    else:dist.destroy_process_group()


if __name__=='__main__':main()
