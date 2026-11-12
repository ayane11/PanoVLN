"""CPU coverage for update clocks and same-phase optimizer/RNG recovery."""
import tempfile
import unittest
from pathlib import Path
import torch
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from src.memory import EpisodeMemory


class TrainingStateTests(unittest.TestCase):
    def test_overwriting_checkpoint_invalidates_old_completion_marker(self):
        import json
        from src.train.train_memvln import begin_checkpoint_write, finish_checkpoint_write
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'checkpoint-2'
            destination.mkdir()
            marker = destination / 'complete.json'
            marker.write_text(json.dumps(dict(updates=2, config={})))
            tensor_file = destination / 'model.pt'
            tensor_file.write_bytes(b'old tensor state')
            begin_checkpoint_write(destination)
            self.assertFalse(marker.exists())
            self.assertTrue(tensor_file.exists())
            # An interrupted overwrite cannot be considered resumable.
            tensor_file.write_bytes(b'new tensor state')
            self.assertFalse(marker.exists())
            finish_checkpoint_write(destination, {'seed': 42}, 1, 2)
            self.assertEqual(json.loads(marker.read_text()),
                dict(config={'seed': 42}, world_size=1, updates=2))
            self.assertFalse((destination / 'complete.json.tmp').exists())

    def test_pruning_preserves_referenced_and_unknown_checkpoints(self):
        import json
        from src.memory.checkpoint import prune_checkpoints
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'; output.mkdir()
            original = Path(directory) / 'base'; original.mkdir()
            for step in range(1, 6):
                path = output / f'checkpoint-{step}'; path.mkdir()
                (path / 'complete.json').write_text(json.dumps(dict(updates=step, config={})))
            incomplete = output / 'checkpoint-6'; incomplete.mkdir()
            (output / 'checkpoint-999').symlink_to(original, target_is_directory=True)
            removed = prune_checkpoints(output, 2, [output / 'checkpoint-1' / 'inference'])
            self.assertEqual(set(removed), {'checkpoint-2', 'checkpoint-3'})
            for name in ('checkpoint-1','checkpoint-4','checkpoint-5','checkpoint-6','checkpoint-999'):
                self.assertTrue((output / name).exists())
            self.assertTrue(original.is_dir())

    def test_accumulation_scheduler_optimizer_memory_and_rng_resume(self):
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        torch.manual_seed(44)
        model = EpisodeMemory(dict(feature_dim=16, heads=4, tokens=3, bank_frames=3,
            frame_tokens=4, pose_hidden=16, checkpoint_blocks=True))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        model, optimizer = accelerator.prepare(model, optimizer)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 5)
        accelerator.register_for_checkpointing(scheduler)
        model.write('a', 0, (0., 0., 0.), torch.randn(8, 16))

        def update():
            for _ in range(2):
                with accelerator.accumulate(model):
                    loss = model.read('a', 0, (0., 0., 0.)).square().sum()
                    accelerator.backward(loss)
                    optimizer.step()
                    if accelerator.sync_gradients:
                        scheduler.step()
                    optimizer.zero_grad()
            return float(loss.detach())

        update()
        self.assertEqual(scheduler.last_epoch, 1)
        with tempfile.TemporaryDirectory() as directory:
            accelerator.save_state(directory)
            memory = model.export_state()
            expected_loss = update()
            expected_rng = torch.rand(5)
            expected_params = {n: p.detach().clone() for n, p in model.named_parameters()}
            expected_lr = scheduler.get_last_lr()
            accelerator.load_state(directory)
            model.import_state(memory)
            self.assertAlmostEqual(expected_loss, update(), places=5)
            torch.testing.assert_close(torch.rand(5), expected_rng)
            self.assertEqual(scheduler.get_last_lr(), expected_lr)
            self.assertEqual(scheduler.last_epoch, 2)
            for name, param in model.named_parameters():
                torch.testing.assert_close(param, expected_params[name])
        accelerator.end_training()


if __name__ == '__main__':
    unittest.main()
