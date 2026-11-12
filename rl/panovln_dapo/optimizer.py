"""Explicit AdamW moment offload between steps, preserving scalar step state."""
import torch


def move_adamw_moments(optimizer, device):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError('Moment offload currently supports AdamW only')
    for state in optimizer.state.values():
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if key in state:
                state[key] = state[key].to(device=device)


def step_adamw_with_parameter_offload(optimizer):
    """Update one parameter at a time; keep all other Adam moments on CPU.

    Each parameter receives exactly one ordinary AdamW step. Global clipping
    must already have happened. No scheduler/closure or optimizer step hooks
    are supported here because they would observe per-parameter invocations.
    """
    if not isinstance(optimizer,torch.optim.AdamW):raise TypeError('Requires AdamW')
    if optimizer._optimizer_step_pre_hooks or optimizer._optimizer_step_post_hooks:
        raise ValueError('Per-parameter offload does not support optimizer step hooks')
    groups=optimizer.param_groups
    if any(group.get('foreach') or group.get('fused') or group.get('differentiable') or group.get('capturable') for group in groups):
        raise ValueError('Per-parameter offload requires ordinary non-capturable AdamW')
    try:
        for group in groups:
            for parameter in group['params']:
                if parameter.grad is None:continue
                state=optimizer.state[parameter]
                for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                    if key in state:state[key]=state[key].to(parameter.device)
                optimizer.param_groups=[dict(group,params=[parameter])]
                optimizer.step()
                for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                    if key in state:state[key]=state[key].to('cpu')
    finally:
        optimizer.param_groups=groups


def load_adamw_state_on_cpu(optimizer,state_dict):
    """Use PyTorch's own state loader without its automatic CUDA moment copies."""
    if not isinstance(optimizer,torch.optim.AdamW):raise TypeError('Requires AdamW')
    groups=optimizer.param_groups
    placeholders=[];cpu_groups=[]
    for group in groups:
        params=[torch.nn.Parameter(torch.empty(0,dtype=p.dtype,device='cpu')) for p in group['params']]
        placeholders.extend(zip(group['params'],params))
        cpu_groups.append(dict(group,params=params))
    temporary=torch.optim.AdamW(cpu_groups)
    temporary.load_state_dict(state_dict)
    for real,placeholder in placeholders:
        state=temporary.state.get(placeholder,{})
        for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if key in state and state[key].shape!=real.shape:
                raise ValueError(f'Adam checkpoint shape mismatch: {key}')
    optimizer.state.clear()
    for real,placeholder in placeholders:
        if placeholder in temporary.state:optimizer.state[real]=temporary.state[placeholder]
    for real_group,loaded_group in zip(groups,temporary.param_groups):
        real_group.update({key:value for key,value in loaded_group.items() if key!='params'})
