"""Portable inference bundles: frozen base + LoRA/token deltas + memory weights."""
import json
from pathlib import Path
import torch


def export_bundle(model, processor, destination, base_path):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True, save_embedding_layers=False)
    root = model.get_base_model()
    root.config.save_pretrained(destination)
    processor.save_pretrained(destination)
    torch.save({k: v.detach().cpu() for k, v in root.episode_memory.state_dict().items()}, destination / 'episode_memory.pt')
    (destination / 'memory_bundle.json').write_text(json.dumps(dict(version=1,
        base_model=str(Path(base_path).resolve()), memory_config=root.config.memory_config,
        memory_special_token_ids=getattr(root.config, 'memory_special_token_ids', None))))


def load_bundle(path, attn_implementation='flash_attention_2', dtype=torch.bfloat16):
    from transformers import AutoProcessor
    from peft import PeftModel
    from src.qwen_vl import Qwen3_5Config, Qwen3_5ForConditionalGenerationForPanoVLN
    from .tokens import configure_memory_tokens
    path = Path(path)
    manifest = json.loads((path / 'memory_bundle.json').read_text())
    config = Qwen3_5Config.from_pretrained(path, local_files_only=True)
    # Load the frozen base first; initialize trained memory explicitly afterwards.
    config.memory_config = None
    config.use_cache = True
    config.text_config.use_cache = True
    model = Qwen3_5ForConditionalGenerationForPanoVLN.from_pretrained(manifest['base_model'],
        config=config, torch_dtype=dtype, attn_implementation=attn_implementation, local_files_only=True)
    from .integration import install_memory
    if hasattr(model.model, '_pano_origin_get_image_features'):
        model.model.get_image_features = model.model._pano_origin_get_image_features
        del model.model._pano_origin_get_image_features
    install_memory(model, manifest['memory_config'])
    model.episode_memory.to(dtype=dtype)
    model._install_image_feature_hook()
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    configure_memory_tokens(processor, model)
    model.episode_memory.load_state_dict(torch.load(path / 'episode_memory.pt', map_location='cpu', weights_only=True))
    # Preserve the trained computation: merging FP32 LoRA deltas into BF16
    # base weights rounds the sum and can change logits appreciably.
    model = PeftModel.from_pretrained(model, path)
    return model, processor


def prune_checkpoints(output_dir, keep=2, protected=()):
    """Remove only completed checkpoints created by this trainer in this run."""
    import shutil
    import re
    output = Path(output_dir).resolve()
    if keep < 1:
        raise ValueError('At least one checkpoint must be retained')
    protected = [Path(p).resolve() for p in protected if p]
    candidates = []
    for path in output.iterdir():
        match = re.fullmatch(r'checkpoint-(\d+)', path.name)
        if not match or path.is_symlink() or not (path / 'complete.json').is_file():
            continue
        record = json.loads((path / 'complete.json').read_text())
        if record.get('updates') != int(match.group(1)) or 'config' not in record:
            continue
        candidates.append((int(match.group(1)), path))
    removed = []
    for _, path in sorted(candidates, reverse=True)[keep:]:
        resolved = path.resolve()
        if any(p == resolved or resolved in p.parents for p in protected):
            continue
        shutil.rmtree(path)
        removed.append(path.name)
    return removed
