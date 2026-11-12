"""FSDP policy construction validated by the real-model optimizer probe."""
from functools import partial
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from .precision import kernel_construction
from .attention import resolve_attention
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from src.qwen_vl import Qwen3_5ForConditionalGenerationForPanoVLN


def load_actor(checkpoint, device, *, compute_dtype=torch.float32, fsdp_size=-1, gradient_checkpointing=True,
               forward_prefetch=False, use_torch_compile=True, attention_backend='auto'):
    world=dist.get_world_size()
    shard_size=world if fsdp_size==-1 else fsdp_size
    if shard_size<1 or world%shard_size:
        raise ValueError('FSDP size must be -1 or a positive divisor of world size')
    mesh=None
    strategy=ShardingStrategy.FULL_SHARD
    if shard_size<world:
        mesh=init_device_mesh('cuda',(world//shard_size,shard_size),mesh_dim_names=('replicate','shard'))
        strategy=ShardingStrategy.HYBRID_SHARD
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Supported actor compute dtypes: float32, bfloat16')
    if compute_dtype==torch.bfloat16:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        if not qwen.is_fast_path_available: raise RuntimeError('BF16 RL requires FLA and causal-conv1d fast path')
    mixed_precision = None
    if compute_dtype == torch.bfloat16:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            # ERP geometry is FP32 coordinate metadata, not a model activation.
            # Casting it here changes sampling coordinates before the native
            # fusion can use them. Let the model cast image/compute tensors.
            cast_root_forward_inputs=False,
            cast_forward_inputs=False,
        )
    attention_backend=resolve_attention(attention_backend,'bf16' if compute_dtype==torch.bfloat16 else 'fp32')
    with kernel_construction(compute_dtype):
        model=Qwen3_5ForConditionalGenerationForPanoVLN.from_pretrained(
            checkpoint,local_files_only=True,torch_dtype=torch.float32,attn_implementation=attention_backend)
    if compute_dtype==torch.bfloat16:
        kernels={getattr(m.chunk_gated_delta_rule,'__module__','unknown')
                 for m in model.modules() if hasattr(m,'chunk_gated_delta_rule')}
        if not kernels or not all(name.startswith('fla.') for name in kernels):
            raise RuntimeError(f'Expected actual FLA bindings, found {kernels}')
        print(f'RL acceleration: compute=BF16 master=FP32 reduce=FP32 geometry_input=FP32 FLA={sorted(kernels)}',flush=True)
    model._ensure_panovggt_model(device,compute_dtype)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    else:
        model.gradient_checkpointing_disable()
    model.train()
    blocks={type(m) for m in model.modules() if type(m).__name__ in ('Qwen3_5DecoderLayer','Qwen3_5VisionBlock')}
    if not blocks:raise ValueError('Expected Qwen3.5 transformer blocks')
    # Keep tied embedding/head together at root; do not wrap individual Linear
    # modules. Frozen geometry stays replicated because its native hook moves it.
    wrapped = FSDP(model,device_id=device,use_orig_params=True,ignored_modules=[model.panovggt],
                mixed_precision=mixed_precision,sharding_strategy=strategy,device_mesh=mesh,
                forward_prefetch=forward_prefetch,
                auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls=blocks))

    wrapped._dapo_compile_entropy=use_torch_compile
    wrapped._dapo_compute_dtype=compute_dtype
    return wrapped
