"""Run in the Pano environment with CUDA_VISIBLE_DEVICES empty (no model load)."""
import unittest
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor
from panovln_dapo.inputs import build_inputs, ObservationConfig
from src.eval.eval import (build_eval_messages, build_eval_generation_prompt,
                           select_vln_eval_image_indices, preprocess_vln_eval_images)
from src.train.data.data import build_vln_image_geometry_batch, preprocess_panovggt_current_image


class NativeInputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = Path(__file__).resolve().parents[3] / 'checkpoints' / 'PanoVLN'
        cls.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)

    def test_native_equivalence(self):
        for n in (1, 12, 105):
            with self.subTest(frames=n):
                frames = [Image.new('RGB', (1280, 640), (i % 256, 40, 80)) for i in range(n)]
                result = build_inputs(self.processor, 'Walk forward and stop.', frames)
                indices = select_vln_eval_image_indices(n, 10, 100)
                images = preprocess_vln_eval_images(frames, indices)
                prompt = build_eval_generation_prompt(self.processor, build_eval_messages('Walk forward and stop.', images))
                native = self.processor(text=[prompt], images=images, return_tensors='pt', padding=True)
                self.assertEqual(indices, result['image_indices'])
                self.assertEqual(prompt, result['prompt'])
                for key, value in native.items():
                    self.assertTrue(torch.equal(value, result['inputs'][key]), key)
                self.assertTrue(torch.equal(result['inputs']['image_erp_geometry'], build_vln_image_geometry_batch(len(images))))
                self.assertTrue(torch.equal(result['inputs']['panovggt_pixel_values'], preprocess_panovggt_current_image(frames[-1]).unsqueeze(0)))
                self.assertEqual(result['inputs']['image_num_images'].tolist(), [len(images)])
                self.assertEqual(result['inputs']['image_current_index'].tolist(), [len(images)-1])

    def test_no_silent_truncation(self):
        with self.assertRaisesRegex(ValueError, 'do not truncate'):
            build_inputs(self.processor, 'Go.', [Image.new('RGB', (1280, 640))], ObservationConfig(max_prompt_tokens=1))


if __name__ == '__main__':
    unittest.main()
