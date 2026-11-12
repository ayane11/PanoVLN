"""Explicit native attention selection shared by collection and actor loading."""

def resolve_attention(backend, precision):
    if precision not in ('bf16','fp32'):raise ValueError('Unsupported attention precision')
    if backend=='auto':backend='flash_attention_2' if precision=='bf16' else 'sdpa'
    if backend not in ('sdpa','flash_attention_2'):raise ValueError('Unsupported attention backend')
    if backend=='flash_attention_2' and precision!='bf16':
        raise ValueError('FlashAttention2 requires BF16 compute in this RL implementation')
    return backend
