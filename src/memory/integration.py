"""Narrow hooks around Qwen3.5's unchanged position/cache implementation."""
import torch
from .bank import EpisodeMemory


def install_memory(model, settings):
    if hasattr(model, 'episode_memory'):
        return
    if settings.get('feature_dim', 2560) != model.config.text_config.hidden_size:
        raise ValueError('Memory feature_dim must match the language hidden size')
    model.config.memory_config = dict(settings)
    model.episode_memory = EpisodeMemory(settings)
    model._memory_context = None
    original = model.model.get_image_features

    def get_image_features(*args, **kwargs):
        output = original(*args, **kwargs)
        ctx = model._memory_context
        if ctx is not None and ctx['write']:
            features = output.pooler_output
            counts = ctx['image_num_images']
            current = ctx['image_current_index']
            if counts is None or current is None:
                raise ValueError('Memory writing requires explicit image grouping')
            offset = 0
            for i, (count, index) in enumerate(zip(counts.tolist(), current.tolist())):
                if count <= 0 or index < 0 or index >= count:
                    raise ValueError('Invalid current image index')
                model.episode_memory.write(ctx['episodes'][i], ctx['steps'][i], ctx['odom'][i], features[offset + index])
                offset += count
            if offset != len(features):
                raise ValueError('Image grouping does not match visual output')
        return output

    # Installed before the PanoVGGT wrapper: memory receives semantic features.
    model.model.get_image_features = get_image_features

    def inject(module, args, kwargs):
        ctx = model._memory_context
        if ctx is None:
            return args, kwargs
        embeds = kwargs.get('inputs_embeds')
        ids = ctx['input_ids']
        mask = ids.eq(model.config.memory_token_id)
        expected = model.episode_memory.latents.shape[1]
        if not torch.all(mask.sum(-1) == expected):
            raise ValueError(f'Each prefill must contain exactly {expected} memory tokens')
        retrieved = torch.stack([model.episode_memory.read(e, t, o) for e, t, o in zip(ctx['episodes'], ctx['steps'], ctx['odom'])])
        kwargs['inputs_embeds'] = embeds.masked_scatter(mask.unsqueeze(-1).expand_as(embeds), retrieved.to(embeds).reshape(-1))
        return args, kwargs

    model.model.language_model.register_forward_pre_hook(inject, with_kwargs=True)
