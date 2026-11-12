"""Single source of Pano observation inputs for rollout and actor replay."""
from dataclasses import dataclass
import torch
from src.train.data.data import (
    VLN_SYSTEM_PROMPT, build_vln_user_content, build_vln_image_selection,
    preprocess_vln_current_image, preprocess_vln_memory_image,
    preprocess_panovggt_current_image, build_vln_image_geometry_batch,
    resolve_current_image_index,
)
from src.train.utils import build_prompt_and_target


@dataclass(frozen=True)
class ObservationConfig:
    max_memory_images: int = 10
    memory_pool_window_frames: int = 100
    top_crop_degrees: float = 20.0
    bottom_crop_degrees: float = 20.0
    panovggt_enabled: bool = True
    max_prompt_tokens: int = 4096


def build_inputs(processor, instruction, raw_frames, config=ObservationConfig()):
    if not raw_frames:
        raise ValueError("An observation history is required")
    if config.max_memory_images < 0 or config.memory_pool_window_frames < 1:
        raise ValueError("Invalid history configuration")
    indices = build_vln_image_selection(len(raw_frames)-1, len(raw_frames)-1,
                                       config.max_memory_images, config.memory_pool_window_frames)
    crop = dict(top_crop_degrees=config.top_crop_degrees, bottom_crop_degrees=config.bottom_crop_degrees)
    images = [(preprocess_vln_current_image if i == len(indices)-1 else preprocess_vln_memory_image)(raw_frames[j], **crop)
              for i,j in enumerate(indices)]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": VLN_SYSTEM_PROMPT}]},
        {"role": "user", "content": build_vln_user_content(instruction=instruction, num_images=len(images))},
    ]
    prompt = build_prompt_and_target(messages, "chat_template", processor, require_target=False)["prompt"]
    tensors = processor(text=[prompt], images=images, return_tensors="pt", padding=True)
    if tensors["input_ids"].shape[-1] > config.max_prompt_tokens:
        raise ValueError(f"Prompt has {tensors['input_ids'].shape[-1]} tokens, budget {config.max_prompt_tokens}; do not truncate images")
    tensors["image_erp_geometry"] = build_vln_image_geometry_batch(len(images), **crop)
    tensors["image_num_images"] = torch.tensor([len(images)], dtype=torch.long)
    tensors["image_current_index"] = torch.tensor([resolve_current_image_index(len(images))], dtype=torch.long)
    if config.panovggt_enabled:
        tensors["panovggt_pixel_values"] = preprocess_panovggt_current_image(raw_frames[-1]).unsqueeze(0)
    return {"inputs": tensors, "prompt": prompt, "image_indices": indices}
