"""Episode-stream training. No loader sharding; W&B is always offline."""
import argparse
import json
import os
from pathlib import Path

import torch
import yaml


def validate(cfg, world_size):
    if world_size < 1 or cfg['training']['global_batch'] % world_size:
        raise ValueError('global_batch must be divisible by world_size')
    if cfg['training'].get('stage') not in ('warmup', 'joint', 'dagger'):
        raise ValueError('stage must be warmup, joint or dagger')
    if cfg.get('wandb', {}).get('mode', 'offline') != 'offline':
        raise ValueError('This recipe requires W&B offline')
    if not cfg['memory'].get('enabled', True):
        raise ValueError('Use the baseline trainer for memory-disabled training')
    for name in ('name_or_path', 'panovggt_checkpoint_path'):
        if not Path(cfg['model'][name]).exists():
            raise FileNotFoundError(cfg['model'][name])
    for name in ('jsonl', 'image_root'):
        if not Path(cfg['data'][name]).exists():
            raise FileNotFoundError(cfg['data'][name])
    return cfg['training']['global_batch'] // world_size


def build_model(cfg):
    from transformers import AutoProcessor
    from peft import LoraConfig, get_peft_model
    from src.qwen_vl import Qwen3_5Config, Qwen3_5ForConditionalGenerationForPanoVLN
    from src.memory.tokens import configure_memory_tokens
    base = cfg['model']['name_or_path']
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    config = Qwen3_5Config.from_pretrained(base, local_files_only=True)
    # Instantiate memory only after loading the base so its random init is explicit.
    config.memory_config = None
    config.panovggt_checkpoint_path = cfg['model']['panovggt_checkpoint_path']
    model = Qwen3_5ForConditionalGenerationForPanoVLN.from_pretrained(base,
        config=config, torch_dtype=torch.bfloat16,
        attn_implementation=cfg['model']['attn_implementation'], local_files_only=True)
    from src.memory.integration import install_memory
    # Existing geometric wrapper must surround (not precede) semantic memory capture.
    if hasattr(model.model, '_pano_origin_get_image_features'):
        model.model.get_image_features = model.model._pano_origin_get_image_features
        del model.model._pano_origin_get_image_features
    install_memory(model, cfg['memory'])
    model.episode_memory.to(dtype=torch.bfloat16)
    model._install_image_feature_hook()
    token_ids = configure_memory_tokens(processor, model)
    model = apply_adapters(model, token_ids, cfg['training'], cfg['model'].get('initialize_from'))
    return model, processor


def apply_adapters(model, token_ids, training, initialize_from=None):
    from peft import LoraConfig, get_peft_model
    allowed = {'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z',
               'in_proj_b','in_proj_a','out_proj','gate_proj','up_proj','down_proj'}
    targets = [name for name, module in model.named_modules()
        if 'language_model.' in name and isinstance(module, torch.nn.Linear)
        and name.rsplit('.', 1)[-1] in allowed]
    if not targets:
        raise ValueError('No language LoRA targets found')
    embedding = next(name for name, module in model.named_modules() if module is model.get_input_embeddings())
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=training['lora_rank'],
        lora_alpha=training['lora_alpha'], lora_dropout=0., target_modules=targets,
        trainable_token_indices={embedding: token_ids}))
    root = model.get_base_model()
    if initialize_from:
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        initial = Path(initialize_from)
        manifest = json.loads((initial / 'memory_bundle.json').read_text())
        adapter = json.loads((initial / 'adapter_config.json').read_text())
        if manifest['memory_config'] != root.config.memory_config or adapter['r'] != training['lora_rank'] or adapter['lora_alpha'] != training['lora_alpha']:
            raise ValueError('Phase initialization architecture/LoRA config mismatch')
        if manifest.get('memory_special_token_ids') != getattr(root.config, 'memory_special_token_ids', None):
            raise ValueError('Phase initialization memory token IDs mismatch')
        result = set_peft_model_state_dict(model, load_peft_weights(str(initial), device='cpu'))
        # Tied lm_head/token adapters expose duplicate state-dict aliases that
        # PEFT intentionally saves only once. Check unique learned parameters.
        required = {name for name, _ in model.named_parameters()
            if 'lora_' in name or 'trainable_tokens_delta' in name}
        missing = required.intersection(result.missing_keys)
        if result.unexpected_keys or missing:
            raise ValueError(f'Incomplete adapter initialization: missing={sorted(missing)}, unexpected={result.unexpected_keys}')
        root.episode_memory.load_state_dict(torch.load(initial / 'episode_memory.pt', map_location='cpu', weights_only=True))
    for p in root.episode_memory.parameters():
        p.requires_grad = True
    if training['stage'] == 'warmup':
        for name, p in model.named_parameters():
            if 'lora_' in name:
                p.requires_grad = False
    root.config.use_cache = False
    root.config.text_config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    print('LoRA targets:', targets, flush=True)
    model.print_trainable_parameters()
    return model


def step_stream_sample(accelerator, model, optimizer, scheduler, root,
                       dataset, collator, schedule, entry):
    """Consume exactly one rank-local sample; progress advances after backward."""
    if entry['episode_start']:
        root.episode_memory.reset(entry['episode_uid'])
    item = dataset.sample(entry)
    metadata = {key: item.pop(key) for key in
        ('episode_uid', 'memory_steps', 'current_odom', 'memory_write')}
    batch = {key: value.to(accelerator.device) for key, value in collator([item]).items()}
    with accelerator.accumulate(model):
        loss = model(**batch, **metadata).loss
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(f"Nonfinite loss in {entry['episode_uid']} at step {entry['step']}")
        accelerator.backward(loss)
        optimizer.step()
        if accelerator.sync_gradients:
            scheduler.step()
        optimizer.zero_grad()
    result = torch.tensor([float(loss.detach()), 1, entry['task'] == 'navigation',
        entry['task'] == 'summary', int(batch['labels'].ne(-100).sum())], device=accelerator.device)
    schedule.consume(entry)
    if entry['episode_end']:
        root.episode_memory.reset(entry['episode_uid'])
    return result


def verify_restored_trainables(model, bundle):
    """Optional strict audit before a resumed forward; no GPU math is needed."""
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    expected = load_file(str(bundle / 'adapter_model.safetensors'))
    actual = get_peft_model_state_dict(model, save_embedding_layers=False)
    if expected.keys() != actual.keys():
        raise ValueError('Restored adapter keys differ from saved bundle')
    for key in expected:
        torch.testing.assert_close(actual[key].detach().cpu(), expected[key], atol=0, rtol=0, msg=key)
    expected_memory = torch.load(bundle / 'episode_memory.pt', map_location='cpu', weights_only=True)
    for key, value in model.get_base_model().episode_memory.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), expected_memory[key], atol=0, rtol=0, msg=key)
    print('Restored trainable tensors match checkpoint bundle exactly', flush=True)


def begin_checkpoint_write(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # Resuming an older checkpoint can overwrite a later checkpoint number.
    # Invalidate the old commit marker before any tensor files are replaced.
    (destination / 'complete.json').unlink(missing_ok=True)


def finish_checkpoint_write(destination, config, world_size, updates):
    destination = Path(destination)
    temporary = destination / 'complete.json.tmp'
    temporary.write_text(json.dumps(dict(config=config, world_size=world_size, updates=updates)))
    temporary.replace(destination / 'complete.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--init-from')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--world-size', type=int, default=1)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.init_from:
        cfg['model']['initialize_from'] = args.init_from
    if args.resume and args.init_from:
        raise ValueError('Use --resume for same-phase state, --init-from for a new phase')
    world = args.world_size if args.validate_only else int(os.environ.get('WORLD_SIZE', 1))
    accumulation = validate(cfg, world)
    if args.validate_only:
        print(json.dumps(dict(valid=True, world_size=world, accumulation=accumulation, wandb_mode='offline')))
        return
    deterministic = cfg['training'].get('deterministic', True)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    from accelerate import Accelerator, DeepSpeedPlugin
    from accelerate.utils import set_seed
    from transformers import get_cosine_schedule_with_warmup
    from src.memory.stream import build_episode_index, EpisodeSchedule, select_episode_subset, training_budget
    from src.memory.dataset import StreamDataset
    from src.train.data.collator import MultiModalDataCollator
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Output is nonempty: use a new directory or explicit --resume')
    output.mkdir(parents=True, exist_ok=True)
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_DIR'] = str(output.resolve())
    train = cfg['training']
    ds = dict(train_micro_batch_size_per_gpu=1, gradient_accumulation_steps=accumulation,
        train_batch_size=train['global_batch'], bf16=dict(enabled=True),
        gradient_clipping=train['max_grad_norm'], zero_optimization=dict(stage=2,
            overlap_comm=True, contiguous_gradients=True, reduce_bucket_size=50000000))
    if train.get('cpu_offload'):
        ds['zero_optimization']['offload_optimizer'] = dict(device='cpu', pin_memory=True)
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=accumulation,
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds))
    set_seed(train['seed'], deterministic=deterministic)
    index_path = output / 'episode_index.json'
    if accelerator.is_main_process and not index_path.exists():
        index_path.write_text(json.dumps(build_episode_index(cfg['data']['jsonl'], cfg['data']['summaries'], cfg['data'].get('sub_summaries_jsonl'))))
    accelerator.wait_for_everyone()
    index = json.loads(index_path.read_text())
    stat = Path(cfg['data']['jsonl']).stat()
    if (index['source'], index['size'], index['mtime_ns'], index['summaries']) != (
        str(Path(cfg['data']['jsonl']).resolve()), stat.st_size, stat.st_mtime_ns, cfg['data']['summaries']):
        raise ValueError('Stale episode index')
    sub = cfg['data'].get('sub_summaries_jsonl')
    if index.get('sub_summaries_source') != (str(Path(sub).resolve()) if sub else None) or index.get('sub_summaries_mtime_ns') != (Path(sub).stat().st_mtime_ns if sub else None):
        raise ValueError('Sub-summary annotations changed; use a new output directory')
    index = select_episode_subset(index, train.get('episode_subset'), train['seed'])
    schedule = EpisodeSchedule(index, accelerator.process_index, world, train['seed'])
    total = training_budget(index, world, train['seed'], train['epochs'], accumulation, train.get('max_updates'))
    model, processor = build_model(cfg)
    groups = []
    for memory in (True, False):
        for decay in (True, False):
            params = [p for name, p in model.named_parameters() if p.requires_grad
                and ('episode_memory.' in name) == memory and (p.ndim > 1) == decay]
            if params:
                groups.append(dict(params=params, lr=train['memory_learning_rate'] if memory else train['learning_rate'],
                    weight_decay=train['weight_decay'] if decay else 0.))
    optimizer = torch.optim.AdamW(groups)
    # No dataloader is prepared: EpisodeSchedule already owns rank partitioning.
    model, optimizer = accelerator.prepare(model, optimizer)
    # Manual optimizer-update clock: AcceleratedScheduler would multiply steps by
    # world size for a loader it assumes was sharded. Also bind after CPU offload
    # may have replaced the optimizer during prepare.
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total * train['warmup_ratio']), total)
    accelerator.register_for_checkpointing(scheduler)
    root = accelerator.unwrap_model(model).get_base_model()
    dataset = StreamDataset(cfg['data']['jsonl'], processor, cfg['data']['image_root'],
        cfg['memory']['tokens'], bool(root.config.panovggt_enabled))
    collator = MultiModalDataCollator(processor.tokenizer)
    updates = start_epoch = 0
    counters = dict(navigation_samples=0, summary_samples=0, target_tokens=0)
    if args.resume:
        complete = Path(args.resume) / 'complete.json'
        if not complete.is_file():
            raise ValueError('Checkpoint is incomplete: missing complete.json')
        contract = json.loads(complete.read_text())
        if contract['config'] != cfg or contract['world_size'] != world:
            raise ValueError('Resume config/world size changed; start a new phase with --init-from')
        progress = torch.load(Path(args.resume) / f'stream_rank{accelerator.process_index}.pt', map_location='cpu', weights_only=False)
        start_epoch, updates = progress['epoch'], progress['updates']
        schedule = EpisodeSchedule(index, accelerator.process_index, world, train['seed'], start_epoch)
        schedule.load_state_dict(progress['schedule'])
        counters = progress['counters']
    if accelerator.is_main_process:
        (output / 'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
        processor.save_pretrained(output / 'processor')
        import wandb
        run = wandb.init(project=cfg['wandb']['project'], mode='offline', dir=str(output), config=cfg,
            name=output.name, reinit=True)
    else:
        run = None

    # Restore RNG only after setup/logging, which can consume randomness.
    if args.resume:
        # Frozen weights come from the configured base checkpoint.
        accelerator.load_state(args.resume, load_module_strict=False)
        root.episode_memory.import_state(progress['memory'])
        if train.get('verify_resume_state', True):
            verify_restored_trainables(accelerator.unwrap_model(model), Path(args.resume) / 'inference')
    last_saved_updates = updates if args.resume else -1

    def save(epoch, stream):
        nonlocal last_saved_updates
        if updates == last_saved_updates:
            return
        destination = output / f'checkpoint-{updates}'
        if accelerator.is_main_process:
            begin_checkpoint_write(destination)
        accelerator.wait_for_everyone()
        accelerator.save_state(str(destination), exclude_frozen_parameters=True)
        torch.save(dict(epoch=epoch, updates=updates, counters=counters, schedule=stream.state_dict(),
            memory=root.episode_memory.export_state()), destination / f'stream_rank{accelerator.process_index}.pt')
        if accelerator.is_main_process:
            root.config.save_pretrained(destination)
            processor.save_pretrained(destination)
            from src.memory.checkpoint import export_bundle
            export_bundle(accelerator.unwrap_model(model), processor, destination / 'inference', cfg['model']['name_or_path'])
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            finish_checkpoint_write(destination, cfg, world, updates)
            from src.memory.checkpoint import prune_checkpoints
            removed = prune_checkpoints(output, train.get('save_total_limit', 2),
                protected=[args.resume, cfg['model'].get('initialize_from')])
            if removed:
                print('Removed superseded checkpoints:', removed, flush=True)
        accelerator.wait_for_everyone()
        last_saved_updates = updates

    for epoch in range(start_epoch, train['epochs']):
        if epoch != start_epoch or not args.resume:
            schedule = EpisodeSchedule(index, accelerator.process_index, world, train['seed'], epoch)
            root.episode_memory.reset()
        usable = len(schedule.items) // accumulation * accumulation
        if accelerator.is_main_process:
            print(f'Epoch {epoch}: tail truncation per rank={schedule.truncated_counts}; accumulation tail={len(schedule.items) - usable}', flush=True)
        model.train()
        statistics = torch.zeros(5, device=accelerator.device)
        for entry in schedule:
            if schedule.consumed >= usable or updates >= total:
                break
            statistics += step_stream_sample(accelerator, model, optimizer, scheduler,
                root, dataset, collator, schedule, entry)
            if accelerator.sync_gradients:
                updates += 1
                global_stats = accelerator.reduce(statistics, reduction='sum').tolist()
                for name, value in zip(counters, global_stats[2:]):
                    counters[name] += int(value)
                if accelerator.is_main_process:
                    print(json.dumps(dict(update=updates, loss=global_stats[0] / global_stats[1],
                        **counters, lr=scheduler.get_last_lr()[0])), flush=True)
                if run:
                    run.log(dict(loss=global_stats[0] / global_stats[1], updates=updates, epoch=epoch,
                        **counters, lr=scheduler.get_last_lr()[0]), step=updates)
                statistics.zero_()
                if updates % train['save_updates'] == 0:
                    save(epoch, schedule)
        save(epoch, schedule)
        if updates >= total:
            break
    if updates != total:
        raise RuntimeError(f'Training ended at {updates} updates, expected {total}')
    if run:
        run.finish()
    accelerator.end_training()


if __name__ == '__main__':
    main()
