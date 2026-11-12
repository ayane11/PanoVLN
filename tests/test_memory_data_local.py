"""Opt-in contract test using the existing full-size processor and real data."""
import json
import os
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get('PANOVLN_LOCAL_DATA_TEST') == '1', 'requires local PanoVLN data')
class LocalDataTests(unittest.TestCase):
    def test_portable_bundle_full_reload(self):
        import torch
        from transformers import AutoProcessor
        from test_memory_integration import IntegrationTests
        from src.memory.tokens import configure_memory_tokens
        from src.memory.integration import install_memory
        from src.memory.checkpoint import export_bundle, load_bundle
        from src.train.train_memvln import apply_adapters
        processor = AutoProcessor.from_pretrained('checkpoints/PanoVLN', local_files_only=True)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'base'
            bundle = Path(directory) / 'bundle'
            root = IntegrationTests().make(False, tied=True)
            tokens = configure_memory_tokens(processor, root)
            root.save_pretrained(base)
            settings = dict(feature_dim=32, heads=4, tokens=3,
                bank_frames=3, frame_tokens=4, pose_hidden=16)
            install_memory(root, settings)
            model = apply_adapters(root, tokens, dict(lora_rank=2, lora_alpha=4, stage='joint'))
            root.episode_memory.write('a', 0, (0., 0., 0.), torch.randn(8, 32))
            ids = torch.tensor([[tokens[0], tokens[1], tokens[1], tokens[1], tokens[2], 6]])
            args = dict(input_ids=ids, episode_uid=['a'], memory_steps=[0],
                current_odom=[[0., 0., 0.]], memory_write=False)
            model(**args, labels=torch.tensor([[-100] * 5 + [6]])).loss.backward()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                expected = model(**args).logits
            export_bundle(model, processor, bundle, base)
            from src.qwen_vl import Qwen3_5ForConditionalGenerationForPanoVLN
            phase_root = Qwen3_5ForConditionalGenerationForPanoVLN.from_pretrained(base, attn_implementation='eager')
            install_memory(phase_root, settings)
            phase = apply_adapters(phase_root, tokens,
                dict(lora_rank=2, lora_alpha=4, stage='joint'), str(bundle)).eval()
            phase_root.episode_memory.import_state(root.episode_memory.export_state())
            with torch.no_grad():
                torch.testing.assert_close(expected, phase(**args).logits, atol=1e-5, rtol=1e-4)
            loaded, restored_processor = load_bundle(bundle, 'eager', torch.float32)
            self.assertTrue(loaded.config.use_cache)
            self.assertEqual(restored_processor.tokenizer.convert_tokens_to_ids('<|memory_pad|>'), tokens[1])
            self.assertTrue(set(tokens).issubset(loaded.generation_config.suppress_tokens))
            loaded.episode_memory.import_state(root.episode_memory.export_state())
            loaded.eval()
            with torch.no_grad():
                torch.testing.assert_close(expected, loaded(**args).logits, atol=1e-5, rtol=1e-4)
                generated = loaded.generate(input_ids=ids[:, :-1], episode_uid=['a'],
                    memory_steps=[0], current_odom=[[0., 0., 0.]], memory_write=False,
                    max_new_tokens=2, do_sample=False, pad_token_id=0, eos_token_id=7)
            self.assertGreater(generated.shape[1], ids.shape[1] - 1)
            self.assertEqual(len(loaded.episode_memory.banks['a']), 1)

    def test_real_episode_navigation_and_summary(self):
        from transformers import AutoProcessor
        from src.memory.tokens import configure_memory_tokens
        from src.memory.stream import build_episode_index, EpisodeSchedule
        from src.memory.dataset import StreamDataset
        processor = AutoProcessor.from_pretrained('checkpoints/PanoVLN', local_files_only=True)
        token_ids = configure_memory_tokens(processor)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.jsonl'
            rows = []
            with open('/data/vln/train_r2r_rxr.jsonl') as handle:
                first = json.loads(next(handle)); rows.append(first)
                for line in handle:
                    row = json.loads(line)
                    if (row['dataset'], row['episode_id']) != (first['dataset'], first['episode_id']):
                        break
                    rows.append(row)
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            schedule = EpisodeSchedule(build_episode_index(path))
            dataset = StreamDataset(str(path), processor, '/data/vln', panovggt_enabled=False)
            navigation = dataset.sample(schedule.items[0])
            self.assertEqual(int(navigation['input_ids'].eq(token_ids[1]).sum()), 300)
            self.assertTrue(navigation['memory_write'])
            self.assertEqual(navigation['image_num_images'].tolist(), [1])
            summary = dataset.sample(schedule.items[-1])
            self.assertFalse(summary['memory_write'])
            self.assertNotIn('pixel_values', summary)
            self.assertEqual(int(summary['input_ids'].eq(token_ids[1]).sum()), 300)
            input_text = processor.tokenizer.decode(summary['input_ids'][summary['labels'].eq(-100)])
            self.assertNotIn(first['instruction'].strip(), input_text)
            target = processor.tokenizer.decode(summary['labels'][summary['labels'].ne(-100)], skip_special_tokens=True)
            self.assertEqual(target.strip(), first['instruction'].strip())


if __name__ == '__main__':
    unittest.main()
