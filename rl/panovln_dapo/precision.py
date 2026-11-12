"""Select optional Qwen kernels at construction without changing model structure."""
from contextlib import contextmanager
import torch


@contextmanager
def kernel_construction(dtype, *, require_fla=True):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    if dtype == torch.bfloat16:
        if require_fla and not qwen.is_fast_path_available:
            raise RuntimeError('BF16 RL requires FLA and causal-conv1d fast path')
        yield
        return
    keys=('causal_conv1d_fn','causal_conv1d_update','chunk_gated_delta_rule',
          'fused_recurrent_gated_delta_rule','FusedRMSNormGated','is_fast_path_available')
    saved={key:getattr(qwen,key) for key in keys}
    try:
        for key in keys:setattr(qwen,key,False if key=='is_fast_path_available' else None)
        yield
    finally:
        for key,value in saved.items():setattr(qwen,key,value)
