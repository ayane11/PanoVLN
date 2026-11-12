"""Rank-local stream samples using the existing PanoVLN visual preprocessing."""
import json
import os
import torch
from src.train.data.data import SupervisedDataset
from .bank import integrate_actions
from .tokens import memory_prompt


class StreamDataset(SupervisedDataset):
    def __init__(self, path, processor, image_root, memory_tokens=300, panovggt_enabled=True):
        # Offsets are supplied by the audited schedule: do not rescan multi-GB data.
        self.jsonl_path = path
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_root = image_root
        self.image_token = processor.image_token
        self.model_max_length = None
        self.erp_top_crop_degrees = self.erp_bottom_crop_degrees = 20.
        self.panovggt_enabled = panovggt_enabled
        self.prompt_format = 'chat_template'
        self.memory_token_count = memory_tokens
        self._fp = None
        self.offsets = []

    def sample(self, entry):
        self.offsets = [entry['offset']]
        row = self._load_example(0)
        if entry['task'] == 'navigation':
            # Audit every selected file; missing observations must fail visibly.
            from src.train.data.data import build_vln_image_selection
            for i in build_vln_image_selection(row['step_index'], len(row['images']) - 1):
                path = row['images'][i]
                if not os.path.isfile(path if os.path.isabs(path) else os.path.join(self.image_root, path)):
                    raise FileNotFoundError(path)
            item = super().__getitem__(0)
        else:
            prompt = self.processor.apply_chat_template([dict(role='user', content=[dict(type='text',
                text=memory_prompt(self.memory_token_count) + 'Given the environment memory, infer and summarize the instruction.')])],
                tokenize=False, add_generation_prompt=True)
            target = entry.get('summary_text', row['instruction']) + (self.tokenizer.eos_token or '')
            encoded = self.tokenizer(prompt + target, return_tensors='pt', add_special_tokens=False)
            item = {k: v.squeeze(0) for k, v in encoded.items()}
            count = len(self.tokenizer.encode(target, add_special_tokens=False))
            item['labels'] = item['input_ids'].clone()
            item['labels'][:-count] = -100
        item['episode_uid'] = [entry['episode_uid']]
        item['memory_steps'] = [entry['step']]
        item['current_odom'] = [integrate_actions(row['history_actions'])]
        item['memory_write'] = entry['task'] == 'navigation'
        return item
