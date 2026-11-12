MEMORY_TOKENS = ('<|memory_start|>', '<|memory_pad|>', '<|memory_end|>')


def memory_prompt(count=300):
    return 'Global environment memory: ' + MEMORY_TOKENS[0] + MEMORY_TOKENS[1] * count + MEMORY_TOKENS[2] + '\n'


def configure_memory_tokens(processor, model=None):
    tokenizer = processor.tokenizer
    import inspect
    if 'replace_extra_special_tokens' in inspect.signature(tokenizer.add_special_tokens).parameters:
        tokenizer.add_special_tokens({'extra_special_tokens': list(MEMORY_TOKENS)}, replace_extra_special_tokens=False)
    else:
        tokenizer.add_special_tokens({'additional_special_tokens': list(MEMORY_TOKENS)}, replace_additional_special_tokens=False)
    ids = [tokenizer.convert_tokens_to_ids(t) for t in MEMORY_TOKENS]
    if any(len(tokenizer.encode(t, add_special_tokens=False)) != 1 for t in MEMORY_TOKENS):
        raise ValueError('Memory markers must each encode to one token')
    if model is not None:
        if model.get_input_embeddings().num_embeddings < len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        model.config.memory_token_id = ids[1]
        model.config.memory_special_token_ids = ids
        if getattr(model, "generation_config", None) is not None:
            suppressed = set(model.generation_config.suppress_tokens or [])
            model.generation_config.suppress_tokens = sorted(suppressed.union(ids))
    return ids
