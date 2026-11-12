import copy
import tempfile
import unittest
from pathlib import Path
from src.train.train_memvln import validate


class LauncherTests(unittest.TestCase):
    def test_batch_math_and_offline_requirement(self):
        with tempfile.TemporaryDirectory() as path:
            cfg = dict(model=dict(name_or_path=path, panovggt_checkpoint_path=path),
                memory=dict(enabled=True), data=dict(jsonl=path, image_root=path),
                training=dict(global_batch=24, stage='joint'), wandb=dict(mode='offline'))
            self.assertEqual(validate(cfg, 3), 8)
            self.assertEqual(validate(cfg, 4), 6)
            with self.assertRaises(ValueError):
                validate(cfg, 5)
            cfg['wandb']['mode'] = 'online'
            with self.assertRaises(ValueError):
                validate(cfg, 3)


if __name__ == '__main__':
    unittest.main()
