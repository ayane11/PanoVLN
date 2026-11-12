import unittest
import torch
from src.qwen_vl import Qwen3_5Config, Qwen3_5ForConditionalGenerationForPanoVLN


class IntegrationTests(unittest.TestCase):
    def make(self, memory=True, tied=False):
        config = Qwen3_5Config(text_config=dict(vocab_size=64, hidden_size=32,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, head_dim=16, layer_types=['linear_attention', 'full_attention'],
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            partial_rotary_factor=.5, rope_parameters=dict(rope_type='default',
            rope_theta=10000., partial_rotary_factor=.5, mrope_section=[1, 1, 2])),
            vision_config=dict(hidden_size=32, intermediate_size=64, depth=1,
                num_heads=4, out_hidden_size=32, patch_size=4, spatial_merge_size=2,
                temporal_patch_size=2, num_position_embeddings=64),
            image_token_id=60, video_token_id=61, vision_start_token_id=62,
            panovggt_enabled=False)
        config.tie_word_embeddings = tied
        config.text_config.tie_word_embeddings = tied
        config._attn_implementation = 'eager'
        if memory:
            config.memory_config = dict(feature_dim=32, heads=4, tokens=3,
                bank_frames=3, frame_tokens=4, pose_hidden=16)
            config.memory_token_id = 3
        return Qwen3_5ForConditionalGenerationForPanoVLN(config)

    def test_summary_forward_backward_and_serialization(self):
        torch.manual_seed(1)
        model = self.make()
        model.episode_memory.write('episode', 0, (0., 0., 0.), torch.randn(8, 32))
        ids = torch.tensor([[1, 3, 3, 3, 5, 6]])
        kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
            episode_uid=['episode'], memory_steps=[0], current_odom=[[0., 0., 0.]], memory_write=False)
        model.train()
        out = model(**kwargs, labels=torch.tensor([[-100] * 5 + [6]]))
        out.loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.episode_memory.retrieval.parameters()))
        self.assertIsNone(model._memory_context)
        model.eval()
        with torch.no_grad():
            a = model(**kwargs).logits
            clone = self.make().eval()
            clone.load_state_dict(model.state_dict())
            clone.episode_memory.import_state(model.episode_memory.export_state())
            torch.testing.assert_close(a, clone(**kwargs).logits)

    def test_images_grouping_and_cached_generation(self):
        model = self.make().eval()
        pixels = torch.randn(32, 96)
        grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
        ids = torch.tensor([[1, 3, 3, 3, 62, 60, 60, 60, 60, 5, 62, 60, 60, 60, 60, 6]])
        with torch.no_grad():
            semantic = model.model.get_image_features(pixels, grid).pooler_output[1]
            model(input_ids=ids, pixel_values=pixels, image_grid_thw=grid,
                mm_token_type_ids=ids.eq(60).long(),
                image_num_images=torch.tensor([2]), image_current_index=torch.tensor([1]),
                episode_uid=['images'], memory_steps=[0], current_odom=[[0., 0., 0.]])
            torch.testing.assert_close(model.episode_memory.banks['images'][0][1], semantic)
            output = model.generate(input_ids=torch.tensor([[1, 3, 3, 3, 5]]),
                episode_uid=['images'], memory_steps=[0], current_odom=[[0., 0., 0.]],
                memory_write=False, max_new_tokens=2, do_sample=False,
                pad_token_id=0, eos_token_id=63)
        self.assertGreater(output.shape[1], 5)
        self.assertEqual(len(model.episode_memory.banks['images']), 1)
        self.assertIsNone(model._memory_context)

    def test_padded_batch_with_different_image_counts(self):
        model = self.make().eval()
        pixels = torch.randn(48, 96)
        grid = torch.tensor([[1, 4, 4]] * 3)
        ids = torch.tensor([
            [1, 3, 3, 3, 62, 60, 60, 60, 60, 5, 62, 60, 60, 60, 60, 6],
            [1, 3, 3, 3, 62, 60, 60, 60, 60, 6, 0, 0, 0, 0, 0, 0]])
        with torch.no_grad():
            semantic = model.model.get_image_features(pixels, grid).pooler_output
            model(input_ids=ids, attention_mask=ids.ne(0), pixel_values=pixels,
                image_grid_thw=grid, mm_token_type_ids=ids.eq(60).long(),
                image_num_images=torch.tensor([2, 1]), image_current_index=torch.tensor([1, 0]),
                episode_uid=['a', 'b'], memory_steps=[0, 0], current_odom=[[0., 0., 0.]] * 2)
        torch.testing.assert_close(model.episode_memory.banks['a'][0][1], semantic[1])
        torch.testing.assert_close(model.episode_memory.banks['b'][0][1], semantic[2])
        with self.assertRaisesRegex(ValueError, 'one entry per batch'):
            model(input_ids=ids, episode_uid=['a'], memory_steps=[0], current_odom=[[0., 0., 0.]])
        with self.assertRaisesRegex(ValueError, 'same episode'):
            model(input_ids=ids, episode_uid=['a', 'a'], memory_steps=[0, 1], current_odom=[[0., 0., 0.]] * 2)

    def test_lora_token_rows_and_adapter_reload(self):
        import tempfile
        from pathlib import Path
        from src.train.train_memvln import apply_adapters
        from peft import PeftModel
        torch.manual_seed(9)
        root = self.make(tied=True)
        base_state = {k: v.detach().clone() for k, v in root.state_dict().items()}
        model = apply_adapters(root, [1, 3, 5], dict(lora_rank=2, lora_alpha=4, stage='joint'))
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(any('in_proj_qkv' in n for n in trainable))
        self.assertFalse(any('visual.' in n for n in trainable))
        root.episode_memory.write('test', 0, (0., 0., 0.), torch.randn(8, 32))
        ids = torch.tensor([[1, 3, 3, 3, 5, 6]])
        kwargs = dict(input_ids=ids, episode_uid=['test'], memory_steps=[0],
            current_odom=[[0., 0., 0.]], memory_write=False, use_cache=False)
        model(**kwargs, labels=torch.tensor([[-100] * 5 + [6]])).loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
            for n, p in model.named_parameters() if 'trainable_tokens' in n))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(**kwargs).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, save_embedding_layers=False)
            clone = self.make(tied=True)
            clone.load_state_dict(base_state)
            clone.episode_memory.load_state_dict(root.episode_memory.state_dict())
            clone.episode_memory.import_state(root.episode_memory.export_state())
            loaded = PeftModel.from_pretrained(clone, directory).eval()
            with torch.no_grad():
                torch.testing.assert_close(expected, loaded(**kwargs).logits)
            merged = loaded.merge_and_unload().eval()
            with torch.no_grad():
                torch.testing.assert_close(expected, merged(**kwargs).logits, atol=1e-5, rtol=1e-4)

    def test_disabled_matches_baseline(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        a = self.make(False).eval()
        b = Qwen3_5ForConditionalGeneration(a.config).eval()
        b.load_state_dict(a.state_dict())
        ids = torch.tensor([[1, 2, 4]])
        with torch.no_grad():
            torch.testing.assert_close(a(ids).logits, b(ids).logits)


if __name__ == '__main__':
    unittest.main()
