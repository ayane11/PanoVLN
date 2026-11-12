"""CPU-only multi-rank check of the production stream step. Not a GPU test."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from transformers import get_cosine_schedule_with_warmup
from test_memory_integration import IntegrationTests
from src.memory.stream import EpisodeSchedule
from src.train.train_memvln import apply_adapters, step_stream_sample
from src.train.data.collator import MultiModalDataCollator


class SyntheticDataset:
    def sample(self, entry):
        ids = torch.tensor([1, 3, 3, 3, 62, 60, 60, 60, 60, 5, 6])
        item = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
            labels=torch.tensor([-100] * 10 + [6]), episode_uid=[entry['episode_uid']],
            memory_steps=[entry['step']], current_odom=[[0., entry['step'] * .25, 0.]],
            memory_write=entry['task'] == 'navigation')
        if entry['task'] == 'navigation':
            item.update(pixel_values=torch.full((16, 96), .1 * (entry['step'] + 1)),
                image_grid_thw=torch.tensor([[1, 4, 4]]), image_num_images=torch.tensor([1]),
                image_current_index=torch.tensor([0]), mm_token_type_ids=ids.eq(60).long())
        else:
            item['input_ids'] = torch.tensor([1, 3, 3, 3, 5, 6])
            item['attention_mask'] = torch.ones(6, dtype=torch.long)
            item['labels'] = torch.tensor([-100] * 5 + [6])
        return item


def main():
    if torch.cuda.is_available():
        raise RuntimeError('This worker must be launched with all GPUs hidden')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir")
    checkpoint_dir = parser.parse_args().checkpoint_dir
    torch.set_num_threads(1)
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    set_seed(42)
    model = apply_adapters(IntegrationTests().make(tied=True), [1, 3, 5],
        dict(lora_rank=2, lora_alpha=4, stage='joint'))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    model, optimizer = accelerator.prepare(model, optimizer)
    root = accelerator.unwrap_model(model).get_base_model()
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 2)
    accelerator.register_for_checkpointing(scheduler)
    class Tokenizer:
        pad_token_id = 0
    collator = MultiModalDataCollator(Tokenizer())
    # Vary task type across ranks at the same microstep. All summaries stay causal.
    episodes = {}
    for i in range(accelerator.num_processes * 2):
        rows = [dict(offset=0, step=0, task='navigation')]
        if i % 2:
            rows += [dict(offset=0, step=0, task='summary'), dict(offset=0, step=1, task='navigation')]
        else:
            rows += [dict(offset=0, step=1, task='navigation'), dict(offset=0, step=1, task='summary')]
        episodes[str(i)] = rows
    index = dict(episodes=episodes, source='synthetic', size=0, mtime_ns=0)
    schedule = EpisodeSchedule(index, accelerator.process_index, accelerator.num_processes)
    stats = torch.zeros(5)
    updates = 0
    def run_remaining():
        nonlocal updates, stats
        for entry in schedule:
            if updates == 2:
                break
            stats += step_stream_sample(accelerator, model, optimizer, scheduler,
                root, SyntheticDataset(), collator, schedule, entry)
            if accelerator.sync_gradients:
                updates += 1
                if updates == 1 and checkpoint_dir:
                    accelerator.save_state(checkpoint_dir)
                    torch.save(dict(schedule=schedule.state_dict(), memory=root.episode_memory.export_state(),
                        stats=stats.clone()), Path(checkpoint_dir) / f'rank-{accelerator.process_index}.pt')
                    accelerator.wait_for_everyone()
    run_remaining()
    assert updates == 2 and schedule.consumed == 4 and scheduler.last_epoch == 2
    if checkpoint_dir:
        expected = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        expected_stats = stats.clone()
        accelerator.load_state(checkpoint_dir)
        saved = torch.load(Path(checkpoint_dir) / f'rank-{accelerator.process_index}.pt', weights_only=False)
        schedule.load_state_dict(saved['schedule'])
        root.episode_memory.import_state(saved['memory'])
        stats, updates = saved['stats'], 1
        run_remaining()
        torch.testing.assert_close(stats, expected_stats)
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.testing.assert_close(param, expected[name])
        assert scheduler.last_epoch == 2
    # Synchronization must produce equal trainable parameters on every rank.
    trainable = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
    gathered = accelerator.gather(trainable).reshape(accelerator.num_processes, -1)
    torch.testing.assert_close(gathered, gathered[0].expand_as(gathered))
    globally = accelerator.reduce(stats, reduction='sum')
    assert int(globally[1]) == 4 * accelerator.num_processes
    assert int(globally[2] + globally[3]) == int(globally[1])
    if accelerator.is_main_process:
        print(json.dumps(dict(world_size=accelerator.num_processes, backend='cpu-gloo',
            updates=updates, consumed_per_rank=4, scheduler_steps=scheduler.last_epoch,
            navigation_samples=int(globally[2]), summary_samples=int(globally[3]),
            trainable_parameters_agree=True, mid_episode_resume=bool(checkpoint_dir))), flush=True)
    accelerator.end_training()


if __name__ == '__main__':
    main()
