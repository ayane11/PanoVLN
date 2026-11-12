"""FSDP1 local-shard migration between phases, matching verl's lifecycle.

This deliberately does not enable CPUOffload during backward: gradients must
accumulate normally within each PPO minibatch.
"""
import torch
from torch.distributed.fsdp._runtime_utils import _lazy_init


@torch.no_grad()
def move_actor_between_phases(model,device):
    _lazy_init(model,model)
    if not model._is_root:raise ValueError('Phase migration requires root FSDP')
    if any(param.grad is not None for param in model.parameters()):
        raise ValueError('Clear gradients before phase migration')
    for handle in model._all_handles:
        if handle._offload_params:raise ValueError('Per-forward CPU offload cannot be mixed with phase offload')
        flat=handle.flat_param
        if flat.data.data_ptr()!=flat._local_shard.data_ptr() or flat.data.size()!=flat._local_shard.size():
            raise ValueError('Expected sharded actor at phase boundary')
        handle.flat_param_to(torch.device(device),non_blocking=False)
        flat._local_shard=flat.data
    # Frozen geometry is excluded from FSDP, so migrate it explicitly.
    model.module.panovggt.to(device)
    if torch.device(device).type=='cpu':torch.cuda.empty_cache()
