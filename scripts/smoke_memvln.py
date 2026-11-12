"""Opt-in GPU probe. Invoke only after user approval; no training job is spawned.

The capacity probe repeats one real visual feature and measures implementation
memory cost. It is not a navigation-quality experiment. Temporary bundles are
removed automatically; the compact JSON result can be recorded in .codex.
"""
import argparse
import json
import tempfile
from pathlib import Path
import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/train/config/memvln.yaml')
    parser.add_argument('--run', action='store_true', help='Explicitly enable GPU work after user approval')
    args = parser.parse_args()
    if not args.run:
        parser.error('GPU probe requires explicit --run and prior user approval')
    if torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one approved GPU with CUDA_VISIBLE_DEVICES')
    from src.train.train_memvln import build_model
    from src.memory.dataset import StreamDataset
    from src.train.data.collator import MultiModalDataCollator
    from src.memory.checkpoint import export_bundle, load_bundle
    cfg = yaml.safe_load(Path(args.config).read_text())
    print('Probe: loading base model and memory adapter', flush=True)
    model, processor = build_model(cfg)
    model.to('cuda').train()
    print('Probe: model placed on the single approved GPU', flush=True)
    root = model.get_base_model()
    dataset = StreamDataset(cfg['data']['jsonl'], processor, cfg['data']['image_root'], cfg['memory']['tokens'])
    with open(cfg['data']['jsonl'], 'rb') as handle:
        first = json.loads(handle.readline())
    if first['step_index'] != 0:
        raise ValueError('Probe requires an episode-start sample')
    entry = dict(offset=0, step=0, task='navigation', episode_uid='gpu-probe')
    item = dataset.sample(entry)
    meta = {key: item.pop(key) for key in ('episode_uid','memory_steps','current_odom','memory_write')}
    batch = {key: value.to('cuda') for key, value in MultiModalDataCollator(processor.tokenizer)([item]).items()}
    torch.cuda.reset_peak_memory_stats()
    print('Probe: real sample forward/backward', flush=True)
    loss = model(**batch, **meta).loss
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite navigation loss')
    loss.backward()
    gradients = {name: any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())
        for name, module in [('tome', root.episode_memory.tome), ('retrieval', root.episode_memory.retrieval), ('pose', root.episode_memory.pos_embed)]}
    if not all(gradients.values()):
        raise ValueError(f'Missing memory gradients: {gradients}')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    raw = next(iter(root.episode_memory.banks['gpu-probe'].values()))[1].clone()
    root.episode_memory.reset()
    for step in range(cfg['memory']['bank_frames'] + 1):
        root.episode_memory.write('gpu-probe', step, (0., step * .25, 0.), raw)
    meta.update(memory_steps=[cfg['memory']['bank_frames']], memory_write=False)
    print('Probe: full 100-entry bank forward/backward', flush=True)
    capacity_loss = model(**batch, **meta).loss
    capacity_loss.backward()
    if not torch.isfinite(capacity_loss):
        raise ValueError('Nonfinite full-bank loss')
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    report = dict(loss=float(loss.detach()), capacity_loss=float(capacity_loss.detach()), gradients=gradients,
        bank_frames=len(root.episode_memory.banks['gpu-probe']),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    print('Probe: export/reload comparison', flush=True)
    model.eval()
    source_inputs = {}
    def capture_source(module, args, kwargs):
        source_inputs['embeds'] = kwargs['inputs_embeds'].detach().cpu()
        source_inputs['positions'] = kwargs['position_ids'].detach().cpu()
    source_hook = root.model.language_model.register_forward_pre_hook(capture_source, with_kwargs=True)
    with torch.no_grad():
        expected = model(**batch, **meta, use_cache=True).logits.cpu()
        expected_no_cache = model(**batch, **meta, use_cache=False).logits.cpu()
        report['source_cache_mode_max_abs_error'] = float((expected - expected_no_cache).abs().max())
    source_hook.remove()
    print('Probe source cache mode difference:', report['source_cache_mode_max_abs_error'], flush=True)
    state = root.episode_memory.export_state()
    root.episode_memory.reset()
    model.to('cpu')
    del optimizer
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix='panovln-memory-probe-') as directory:
        export_bundle(model, processor, directory, cfg['model']['name_or_path'])
        loaded, _ = load_bundle(directory, cfg['model']['attn_implementation'])
        loaded.to('cuda').eval()
        loaded.episode_memory.import_state(state)
        loaded_root = loaded.get_base_model()
        loaded_root._ensure_panovggt_model(torch.device('cuda'), torch.bfloat16)
        # Compare both cache modes with the same mode before/after reload.
        # The training configuration disables cache, the inference bundle enables it.
        original_params = dict(model.named_parameters())
        mismatched = []
        for name, param in loaded.named_parameters():
            previous = original_params[name].detach()
            current = param.detach().cpu()
            if previous.dtype != current.dtype or not torch.equal(previous, current):
                mismatched.append(dict(name=name, before_dtype=str(previous.dtype),
                    after_dtype=str(current.dtype), max_abs_error=float((previous.float() - current.float()).abs().max())))
        report['parameter_mismatch_count'] = len(mismatched)
        report['parameter_mismatch_examples'] = mismatched[:5]
        original_buffers = dict(model.named_buffers())
        report['buffer_mismatches'] = [name for name, value in loaded.named_buffers()
            if name not in original_buffers or original_buffers[name].dtype != value.dtype
            or not torch.equal(original_buffers[name].cpu(), value.cpu())]
        print('Probe parameter/buffer comparison:', json.dumps(dict(parameter_mismatch_count=len(mismatched), examples=mismatched[:5], buffer_mismatches=report['buffer_mismatches'])), flush=True)
        restored_inputs = {}
        def capture_restored(module, args, kwargs):
            restored_inputs['embeds'] = kwargs['inputs_embeds'].detach().cpu()
            restored_inputs['positions'] = kwargs['position_ids'].detach().cpu()
        restored_hook = loaded.get_base_model().model.language_model.register_forward_pre_hook(capture_restored, with_kwargs=True)
        with torch.no_grad():
            actual = loaded(**batch, **meta, use_cache=True).logits.cpu()
            actual_no_cache = loaded(**batch, **meta, use_cache=False).logits.cpu()
        report['no_cache_reload_max_abs_error'] = float((actual_no_cache - expected_no_cache).abs().max())
        restored_hook.remove()
        delta = (source_inputs['embeds'].float() - restored_inputs['embeds'].float()).abs()
        memory_mask = batch['input_ids'].cpu().eq(root.config.memory_token_id)
        visual_mask = batch['mm_token_type_ids'].cpu().eq(1)
        report['embedding_max_abs_error'] = {name: float(delta[mask].max()) for name, mask in
            [('memory', memory_mask), ('visual', visual_mask), ('text', ~(memory_mask | visual_mask))]}
        report['position_ids_equal'] = torch.equal(source_inputs['positions'], restored_inputs['positions'])
        report['bundle_reload_max_abs_error'] = float((actual - expected).abs().max())
        print('Probe metrics before final assertion:', json.dumps(report), flush=True)
        if mismatched or report['buffer_mismatches']:
            raise ValueError('Reloaded effective parameters/buffers differ from the saved model')
        torch.testing.assert_close(actual_no_cache, expected_no_cache, atol=.03, rtol=.03)
        torch.testing.assert_close(actual, expected, atol=.03, rtol=.03)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
