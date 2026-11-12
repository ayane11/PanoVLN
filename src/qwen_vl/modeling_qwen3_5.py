import json
import math
import sys
from pathlib import Path
from types import MethodType

from src.vln_config import configure_vln_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

PANOVGGT_AGGREGATOR_LAYER = -1
PANOVGGT_AGGREGATOR_CONTEXT_DIM = 2048
PANOVGGT_MLP_HIDDEN_SIZE = 4096
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
VENDORED_PANOVGGT_DIR = SRC_ROOT / "panovggt"
VENDORED_PANOVGGT_CONFIG_PATH = VENDORED_PANOVGGT_DIR / "training" / "config" / "default.yaml"

def ensure_panovggt_config(config) -> None:
    configure_vln_model(config)
    vision_config = config.vision_config
    text_config = getattr(config, "text_config", None)
    text_hidden_size = getattr(text_config, "hidden_size", getattr(vision_config, "out_hidden_size", 3584))
    defaults = {
        "panovggt_enabled": False,
        "panovggt_checkpoint_path": "./checkpoints/PanoVGGT/model.pt",
        "panovggt_alpha_value": 0.1,
        "panovggt_force_fp32": False,
        "panovggt_output_dim": int(getattr(vision_config, "out_hidden_size", text_hidden_size)),
    }
    for field_name, default_value in defaults.items():
        if not hasattr(config, field_name):
            setattr(config, field_name, default_value)


def _ensure_vendored_panovggt_available() -> None:
    if not VENDORED_PANOVGGT_DIR.exists():
        raise FileNotFoundError(
            "PanoVGGT is enabled but the vendored source directory is missing: "
            f"{VENDORED_PANOVGGT_DIR}"
        )
    if not VENDORED_PANOVGGT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "PanoVGGT is enabled but the vendored config is missing: "
            f"{VENDORED_PANOVGGT_CONFIG_PATH}"
        )

    src_root = str(SRC_ROOT)
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


def build_panovggt_model_from_vendored_config():
    _ensure_vendored_panovggt_available()

    try:
        from omegaconf import OmegaConf
        from panovggt.models import aggregator as panovggt_aggregator
        from panovggt.models.panovggt_model import PanoVGGTModel
    except Exception as exc:
        raise ImportError(
            "PanoVGGT is enabled but could not be imported from "
            f"{VENDORED_PANOVGGT_DIR}"
        ) from exc

    cfg = OmegaConf.load(str(VENDORED_PANOVGGT_CONFIG_PATH))
    OmegaConf.resolve(cfg)
    mc = cfg.model
    original_dinov2_loader = getattr(panovggt_aggregator.Aggregator, "_try_load_dinov2", None)
    if original_dinov2_loader is not None:
        panovggt_aggregator.Aggregator._try_load_dinov2 = lambda *args, **kwargs: None
    try:
        model = PanoVGGTModel(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            enable_camera=False,
            enable_depth=False,
            enable_point=False,
            enable_global_points=False,
            aggregator=OmegaConf.to_container(mc.aggregator, resolve=True),
        )
    finally:
        if original_dinov2_loader is not None:
            panovggt_aggregator.Aggregator._try_load_dinov2 = original_dinov2_loader
    freeze_panovggt_model(model)
    return model


def freeze_panovggt_model(model: nn.Module | None) -> None:
    if model is None:
        return
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def load_panovggt_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected PanoVGGT checkpoint format: {type(ckpt)!r}")
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in ckpt.items()
    }
    load_result = model.load_state_dict(state_dict, strict=False)
    required_prefixes = ["aggregator."]
    missing_required_keys = [
        key
        for key in getattr(load_result, "missing_keys", [])
        if any(key.startswith(prefix) for prefix in required_prefixes)
    ]
    if missing_required_keys:
        raise RuntimeError(
            "PanoVGGT checkpoint is missing required weights, "
            f"first missing key: {missing_required_keys[0]}"
        )
    freeze_panovggt_model(model)

class PanoVGGTGeometryMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        ensure_panovggt_config(config)

        self.enabled = bool(getattr(config, "panovggt_enabled", False))
        self.layer = PANOVGGT_AGGREGATOR_LAYER
        self.context_dim = PANOVGGT_AGGREGATOR_CONTEXT_DIM
        self.output_dim = int(getattr(config, "panovggt_output_dim", config.text_config.hidden_size))
        self.hidden_dim = PANOVGGT_MLP_HIDDEN_SIZE
        self.alpha_init = float(getattr(config, "panovggt_alpha_value", 0.1))
        self.register_buffer(
            "alpha_value",
            torch.tensor(self.alpha_init, dtype=torch.float32),
        )
        self.spatial_merge_size = int(getattr(config.vision_config, "spatial_merge_size", 2))
        if self.spatial_merge_size <= 0:
            raise ValueError(f"Qwen spatial_merge_size must be positive, got {self.spatial_merge_size}")
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.mlp_input_dim = self.context_dim * self.spatial_merge_unit

        self.input_norm = nn.RMSNorm(self.context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.reset_parameters()

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_value

    def reset_parameters(self) -> None:
        self.input_norm.reset_parameters()
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.alpha_value.fill_(self.alpha_init)

    def _infer_patch_grid(
        self,
        *,
        patch_count: int,
        image_height: int,
        image_width: int,
        patch_size: int,
    ) -> tuple[int, int]:
        if patch_size > 0:
            patch_h = image_height // patch_size
            patch_w = image_width // patch_size
            if patch_h > 0 and patch_w > 0 and patch_h * patch_w == patch_count:
                return patch_h, patch_w

        aspect = max(float(image_width) / max(float(image_height), 1.0), 1e-6)
        best_pair = None
        best_error = float("inf")
        for patch_h in range(1, int(math.sqrt(patch_count)) + 2):
            if patch_count % patch_h != 0:
                continue
            patch_w = patch_count // patch_h
            error = abs((float(patch_w) / float(patch_h)) - aspect)
            if error < best_error:
                best_error = error
                best_pair = (patch_h, patch_w)
        if best_pair is None:
            raise AssertionError(f"Cannot infer PanoVGGT patch grid for {patch_count} tokens")
        return best_pair

    def _sample_geometry_grid(
        self,
        source_grid: torch.Tensor,
        target_h: int,
        target_w: int,
        geometry: torch.Tensor | None,
    ) -> torch.Tensor:
        if target_h <= 0 or target_w <= 0:
            return source_grid.new_zeros((0, source_grid.shape[1]))

        if geometry is None:
            vertical_fov = math.pi
            center_latitude = 0.0
        else:
            vertical_fov = float(geometry[0].detach().cpu())
            center_latitude = float(geometry[1].detach().cpu())

        ys = torch.arange(target_h, device=source_grid.device, dtype=torch.float32) + 0.5
        xs = torch.arange(target_w, device=source_grid.device, dtype=torch.float32) + 0.5
        lat = (ys[:, None] / float(target_h) - 0.5) * vertical_fov + center_latitude
        lon = (xs[None, :] / float(target_w) - 0.5) * (2.0 * math.pi)
        lat = lat.expand(target_h, target_w)
        lon = lon.expand(target_h, target_w)
        grid = torch.stack(
            [
                (lon / math.pi).clamp(-1.0 + 1e-6, 1.0 - 1e-6),
                (lat / (0.5 * math.pi)).clamp(-1.0 + 1e-6, 1.0 - 1e-6),
            ],
            dim=-1,
        ).unsqueeze(0)
        sampled = F.grid_sample(
            source_grid.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled[0].permute(1, 2, 0).reshape(target_h * target_w, source_grid.shape[1])


    def _sample_token_geometry(
        self,
        source_grid: torch.Tensor,
        target_h: int,
        target_w: int,
        geometry: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        group_h = target_h * self.spatial_merge_size
        group_w = target_w * self.spatial_merge_size
        sampled = self._sample_geometry_grid(
            source_grid,
            target_h=group_h,
            target_w=group_w,
            geometry=geometry,
        )
        sampled = self.input_norm(sampled.to(dtype=dtype))
        sampled = sampled.reshape(
            target_h,
            self.spatial_merge_size,
            target_w,
            self.spatial_merge_size,
            self.context_dim,
        )
        sampled = sampled.permute(0, 2, 1, 3, 4).contiguous()
        return sampled.reshape(target_h * target_w, self.mlp_input_dim)


    def forward(
        self,
        panovggt_pixel_values: torch.Tensor,
        panovggt_model: nn.Module,
        target_grid_thw: torch.Tensor,
        target_lengths: list[int],
        image_erp_geometry: torch.Tensor | None,
        output_device: torch.device,
        output_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        if (
            not self.enabled
            or panovggt_model is None
            or panovggt_pixel_values is None
            or panovggt_pixel_values.numel() == 0
        ):
            return []

        if panovggt_pixel_values.ndim == 4:
            panovggt_pixel_values = panovggt_pixel_values.unsqueeze(1)
        if panovggt_pixel_values.ndim != 5:
            raise AssertionError(
                "Expected panovggt_pixel_values shape [B, S, 3, H, W] "
                f"or [B, 3, H, W], got {tuple(panovggt_pixel_values.shape)}"
            )

        encoder = panovggt_model
        encoder_param = next(encoder.parameters())
        param = next(self.mlp.parameters())
        images = panovggt_pixel_values.to(device=encoder_param.device, dtype=encoder_param.dtype)

        with torch.no_grad():
            aggregated = encoder.aggregator(images)
            if isinstance(aggregated, (list, tuple)):
                token_list = aggregated[0]
                patch_start_idx = int(aggregated[1])
                tokens = token_list[self.layer] if isinstance(token_list, list) else token_list
            else:
                tokens = aggregated
                patch_start_idx = 0

            if tokens.ndim != 4:
                raise AssertionError(f"Expected PanoVGGT tokens [B, S, P, C], got {tuple(tokens.shape)}")
            _, _, _, image_h, image_w = images.shape
        tokens = tokens[:, -1, patch_start_idx:, :]
        if tokens.shape[-1] != self.context_dim:
            raise AssertionError(
                f"PanoVGGT context dim mismatch: expected {self.context_dim}, got {tokens.shape[-1]}"
            )

        _, _, _, image_h, image_w = images.shape
        patch_size = int(getattr(encoder, "patch_size", 0))
        patch_h, patch_w = self._infer_patch_grid(
            patch_count=int(tokens.shape[1]),
            image_height=int(image_h),
            image_width=int(image_w),
            patch_size=patch_size,
        )
        source_grid = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)

        deltas = []
        geometry = image_erp_geometry
        if geometry is not None:
            geometry = geometry.to(device=param.device, dtype=torch.float32)
        if int(source_grid.shape[0]) != len(target_lengths) or len(target_lengths) != int(target_grid_thw.shape[0]):
            raise AssertionError(
                "PanoVGGT batch size mismatch: "
                f"source={int(source_grid.shape[0])}, target_lengths={len(target_lengths)}, "
                f"target_grid_thw={int(target_grid_thw.shape[0])}"
            )
        if geometry is not None and int(geometry.shape[0]) != len(target_lengths):
            raise AssertionError(
                "PanoVGGT geometry batch size mismatch: "
                f"geometry={int(geometry.shape[0])}, target_lengths={len(target_lengths)}"
            )
        for sample_index, (grid_thw, target_len) in enumerate(zip(target_grid_thw.tolist(), target_lengths)):
            num_frames, grid_h, grid_w = [int(value) for value in grid_thw]
            if num_frames != 1:
                raise AssertionError(
                    "PanoVGGT geometry fusion expects a single current panorama per Qwen image, "
                    f"got image_grid_thw={grid_thw}"
                )
            if grid_h % self.spatial_merge_size != 0 or grid_w % self.spatial_merge_size != 0:
                raise AssertionError(
                    "PanoVGGT target grid is not divisible by Qwen spatial_merge_size: "
                    f"image_grid_thw={grid_thw}, spatial_merge_size={self.spatial_merge_size}"
                )
            target_h = max(1, grid_h // self.spatial_merge_size)
            target_w = max(1, grid_w // self.spatial_merge_size)
            expected_len = target_h * target_w
            if expected_len != int(target_len):
                raise AssertionError(
                    "Cannot map PanoVGGT geometry to the Qwen 2D visual-token grid: "
                    f"image_grid_thw={grid_thw}, spatial_merge_size={self.spatial_merge_size}, "
                    f"expected_len={expected_len}, qwen_len={int(target_len)}"
                )

            sample_geometry = None if geometry is None else geometry[sample_index]
            geo = self._sample_token_geometry(
                source_grid[sample_index:sample_index + 1],
                target_h=target_h,
                target_w=target_w,
                geometry=sample_geometry,
                dtype=param.dtype,
            )
            if geo.shape[0] != int(target_len):
                raise AssertionError(
                    "PanoVGGT sampled geometry length mismatch: "
                    f"sampled_len={geo.shape[0]}, qwen_len={int(target_len)}"
                )
            projected = self.mlp(geo)
            projected = self.alpha.to(dtype=projected.dtype) * projected
            deltas.append(projected.to(device=output_device, dtype=output_dtype))

        return deltas


class Qwen3_5ForConditionalGenerationForPanoVLN(Qwen3_5ForConditionalGeneration):
    _keys_to_ignore_on_load_unexpected = list(
        getattr(Qwen3_5ForConditionalGeneration, "_keys_to_ignore_on_load_unexpected", []) or []
    ) + [
        r"panovggt\..*",
    ]

    PANOVGGT_STATE_KEYS = (
        "panovggt_mlp.alpha_value",
        "panovggt_mlp.input_norm.weight",
        "panovggt_mlp.mlp.0.weight",
        "panovggt_mlp.mlp.0.bias",
        "panovggt_mlp.mlp.2.weight",
        "panovggt_mlp.mlp.2.bias",
    )
    def __init__(self, config):
        configure_vln_model(config)
        panovggt_enabled = bool(getattr(config, "panovggt_enabled", False))
        if panovggt_enabled:
            ensure_panovggt_config(config)
        super().__init__(config)

        if panovggt_enabled:
            ensure_panovggt_config(config)

        self.panovggt_mlp = PanoVGGTGeometryMLP(config) if panovggt_enabled else None

        self.panovggt = None
        self._panovggt_weights_ready = False
        self._panovggt_dtype = None
        self._panovggt_device = None
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_geometry = None
        self._pano_runtime_panovggt_pixel_values = None
        memory_config = getattr(config, "memory_config", None)
        if memory_config and memory_config.get("enabled", True):
            from src.memory.integration import install_memory
            install_memory(self, memory_config)
        self._install_image_feature_hook()
        self.register_load_state_dict_post_hook(self._load_missing_pano_parameters_post_hook)

    def _panovggt_enabled(self) -> bool:
        return self.panovggt_mlp is not None and bool(getattr(self.panovggt_mlp, "enabled", False))


    def _runtime_grid_matches_image_grid(self, grid_thw: torch.Tensor | None) -> bool:
        runtime_grid = self._pano_runtime_grid_thw
        if grid_thw is None or runtime_grid is None:
            return False
        if grid_thw is runtime_grid:
            return True
        if grid_thw.device == runtime_grid.device and tuple(grid_thw.shape) == tuple(runtime_grid.shape):
            return grid_thw.data_ptr() == runtime_grid.data_ptr()
        return False

    def _select_current_items(
        self,
        image_grid_thw: torch.Tensor,
        num_image_outputs: int | None = None,
    ) -> list[tuple[int, int]]:
        current_indices = self._current_image_flat_indices(image_grid_thw)
        if not current_indices:
            return []
        return [
            (batch_index, image_index)
            for batch_index, image_index in current_indices
            if 0 <= image_index < int(image_grid_thw.shape[0])
            and (num_image_outputs is None or image_index < num_image_outputs)
        ]

    def _panovggt_current_batch_inputs(
        self,
        *,
        panovggt_pixel_values: torch.Tensor,
        current_items: list[tuple[int, int]],
    ) -> torch.Tensor:
        if panovggt_pixel_values is None or panovggt_pixel_values.numel() == 0:
            raise AssertionError("PanoVGGT is enabled but panovggt_pixel_values is missing")
        if int(panovggt_pixel_values.shape[0]) != len(current_items):
            raise AssertionError(
                "PanoVGGT current-image batch size mismatch: "
                f"panovggt_batch={int(panovggt_pixel_values.shape[0])}, "
                f"current_items={len(current_items)}"
            )
        batch_indices = torch.tensor(
            [item[0] for item in current_items],
            device=panovggt_pixel_values.device,
            dtype=torch.long,
        )
        return panovggt_pixel_values.index_select(0, batch_indices)

    def _target_geometry_for_indices(
        self,
        target_indices: list[int],
    ) -> torch.Tensor | None:
        geometry = self._pano_runtime_image_geometry
        if geometry is None:
            return None
        return geometry[target_indices].detach().to(device="cpu", dtype=torch.float32)

    def _apply_post_merger_panovggt_residual(self, vision_output, image_grid_thw: torch.Tensor):
        if image_grid_thw is None or vision_output.pooler_output is None:
            return vision_output

        image_embeds = list(vision_output.pooler_output)
        current_items = self._select_current_items(
            image_grid_thw,
            num_image_outputs=len(image_embeds),
        )
        if not current_items:
            return vision_output

        panovggt_pixel_values = self._pano_runtime_panovggt_pixel_values
        panovggt_inputs = self._panovggt_current_batch_inputs(
            panovggt_pixel_values=panovggt_pixel_values,
            current_items=current_items,
        )
        target_indices = [item[1] for item in current_items]
        target_grid_thw = image_grid_thw[target_indices].detach().to(device="cpu", dtype=torch.long)
        target_lengths = [int(image_embeds[index].shape[0]) for index in target_indices]
        geometry = self._target_geometry_for_indices(target_indices)

        panovggt_model = self._ensure_panovggt_model(
            device=image_embeds[target_indices[0]].device,
            dtype=image_embeds[target_indices[0]].dtype,
        )
        if panovggt_model is None or self.panovggt_mlp is None:
            return vision_output
        deltas = self.panovggt_mlp(
            panovggt_inputs,
            panovggt_model=panovggt_model,
            target_grid_thw=target_grid_thw,
            target_lengths=target_lengths,
            image_erp_geometry=geometry,
            output_device=image_embeds[target_indices[0]].device,
            output_dtype=image_embeds[target_indices[0]].dtype,
        )
        for image_index, delta in zip(target_indices, deltas):
            if delta.shape != image_embeds[image_index].shape:
                raise AssertionError(
                    "PanoVGGT delta shape mismatch: "
                    f"delta={tuple(delta.shape)}, qwen={tuple(image_embeds[image_index].shape)}"
                )
            image_embeds[image_index] = image_embeds[image_index] + delta

        vision_output.pooler_output = tuple(image_embeds)
        return vision_output


    def _install_image_feature_hook(self) -> None:
        if not self._panovggt_enabled():
            return
        if hasattr(self.model, "_pano_origin_get_image_features"):
            return

        self.model._pano_origin_get_image_features = self.model.get_image_features
        owner = self

        def get_image_features_with_pano_residuals(this, pixel_values, image_grid_thw=None, **kwargs):
            is_image_visual_call = owner._runtime_grid_matches_image_grid(image_grid_thw)
            vision_output = this._pano_origin_get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs,
            )
            if not is_image_visual_call:
                return vision_output
            if owner._panovggt_enabled():
                vision_output = owner._apply_post_merger_panovggt_residual(
                    vision_output,
                    image_grid_thw,
                )
            return vision_output

        self.model.get_image_features = MethodType(get_image_features_with_pano_residuals, self.model)

    def _set_runtime_pano_context(
        self,
        *,
        image_grid_thw: torch.Tensor | None,
        image_num_images: torch.Tensor | None,
        image_current_index: torch.Tensor | None,
        image_erp_geometry: torch.Tensor | None,
        panovggt_pixel_values: torch.Tensor | None,
    ) -> None:
        self._pano_runtime_grid_thw = image_grid_thw
        self._pano_runtime_image_num_images = image_num_images
        self._pano_runtime_image_current_index = image_current_index
        self._pano_runtime_image_geometry = image_erp_geometry
        self._pano_runtime_panovggt_pixel_values = panovggt_pixel_values

    def _clear_runtime_pano_context(self) -> None:
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_geometry = None
        self._pano_runtime_panovggt_pixel_values = None


    def _load_external_panovggt_weights(self) -> None:
        if not self._panovggt_enabled():
            return
        if self.panovggt is None:
            self.panovggt = build_panovggt_model_from_vendored_config()
        load_panovggt_checkpoint(
            self.panovggt,
            str(getattr(self.config, "panovggt_checkpoint_path")),
        )
        self._panovggt_weights_ready = True
        self._panovggt_device = None
        self._panovggt_dtype = None

    @staticmethod
    def _load_prefixed_checkpoint_tensors(checkpoint_dir: Path, prefix: str) -> dict[str, torch.Tensor]:
        if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
            return {}

        state_dict: dict[str, torch.Tensor] = {}

        def add_tensor(key: str, tensor: torch.Tensor) -> None:
            if key.startswith(prefix):
                state_dict[key[len(prefix) :]] = tensor

        safetensors_index = checkpoint_dir / "model.safetensors.index.json"
        if safetensors_index.exists():
            from safetensors import safe_open

            weight_map = json.loads(safetensors_index.read_text()).get("weight_map", {})
            files_to_keys: dict[str, list[str]] = {}
            for key, filename in weight_map.items():
                if key.startswith(prefix):
                    files_to_keys.setdefault(filename, []).append(key)
            for filename, keys in files_to_keys.items():
                with safe_open(str(checkpoint_dir / filename), framework="pt", device="cpu") as handle:
                    for key in keys:
                        add_tensor(key, handle.get_tensor(key))
            return state_dict

        safetensors_path = checkpoint_dir / "model.safetensors"
        if safetensors_path.exists():
            from safetensors import safe_open

            with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith(prefix):
                        add_tensor(key, handle.get_tensor(key))
            return state_dict

        def add_from_torch_file(path: Path, keys: set[str] | None = None) -> None:
            loaded = torch.load(str(path), map_location="cpu")
            if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                loaded = loaded["state_dict"]
            if not isinstance(loaded, dict):
                return
            for key, tensor in loaded.items():
                if keys is not None and key not in keys:
                    continue
                add_tensor(key, tensor)

        torch_index = checkpoint_dir / "pytorch_model.bin.index.json"
        if torch_index.exists():
            weight_map = json.loads(torch_index.read_text()).get("weight_map", {})
            files_to_keys: dict[str, set[str]] = {}
            for key, filename in weight_map.items():
                if key.startswith(prefix):
                    files_to_keys.setdefault(filename, set()).add(key)
            for filename, keys in files_to_keys.items():
                add_from_torch_file(checkpoint_dir / filename, keys)
            return state_dict

        torch_path = checkpoint_dir / "pytorch_model.bin"
        if torch_path.exists():
            add_from_torch_file(torch_path)

        return state_dict

    def _load_saved_panovggt_weights(self, pretrained_model_name_or_path) -> bool:
        if not self._panovggt_enabled():
            return False

        checkpoint_dir = Path(str(pretrained_model_name_or_path))
        state_dict = self._load_prefixed_checkpoint_tensors(checkpoint_dir, "panovggt.")
        if not state_dict:
            return False

        if self.panovggt is None:
            self.panovggt = build_panovggt_model_from_vendored_config()

        incompatible = self.panovggt.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            self.panovggt = None
            self._panovggt_weights_ready = False
            raise RuntimeError(
                f"Saved PanoVGGT weights in {checkpoint_dir} do not match the encoder: "
                f"missing={incompatible.missing_keys[:5]}, "
                f"unexpected={incompatible.unexpected_keys[:5]}. "
                "Refusing to replace checkpoint weights with external initialization weights."
            )
        freeze_panovggt_model(self.panovggt)
        self._panovggt_weights_ready = True
        self._panovggt_device = None
        self._panovggt_dtype = None
        return True

    def _mark_panovggt_weights_ready(self) -> None:
        if self.panovggt is None:
            return
        freeze_panovggt_model(self.panovggt)
        self._panovggt_weights_ready = True
        self._panovggt_device = None
        self._panovggt_dtype = None

    def _ensure_panovggt_model(self, device: torch.device, dtype: torch.dtype):
        if not self._panovggt_enabled():
            return None
        if self.panovggt is None:
            self.panovggt = build_panovggt_model_from_vendored_config()
        if not self._panovggt_weights_ready:
            self._load_external_panovggt_weights()
        if bool(getattr(self.config, "panovggt_force_fp32", False)):
            target_dtype = torch.float32
        else:
            # Keep the frozen encoder in Qwen's low-precision vision dtype when
            # possible. PanoVGGT's attention has a bf16 flash path; forcing fp32
            # makes its 36-layer panorama encoder much slower.
            target_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.float32
        if self._panovggt_device != device or self._panovggt_dtype != target_dtype:
            self.panovggt.to(device=device, dtype=target_dtype)
            self._panovggt_device = device
            self._panovggt_dtype = target_dtype
        freeze_panovggt_model(self.panovggt)
        return self.panovggt

    def _current_image_flat_indices(self, image_grid_thw: torch.Tensor | None) -> list[tuple[int, int]]:
        if image_grid_thw is None or image_grid_thw.numel() == 0:
            return []

        num_images = int(image_grid_thw.shape[0])
        device = image_grid_thw.device
        image_num_images = self._pano_runtime_image_num_images
        if image_num_images is None:
            image_num_images = torch.tensor([num_images], device=device, dtype=torch.long)
        else:
            image_num_images = image_num_images.to(device=device, dtype=torch.long)

        if int(image_num_images.sum().item()) != num_images:
            raise AssertionError(
                "image_num_images does not sum to image_grid_thw rows: "
                f"counts={image_num_images.tolist()}, grids={num_images}"
            )

        image_current_index = self._pano_runtime_image_current_index
        if image_current_index is None:
            image_current_index = image_num_images - 1
        else:
            image_current_index = image_current_index.to(device=device, dtype=torch.long)

        offsets = image_num_images.cumsum(0) - image_num_images
        valid = image_num_images > 0
        if torch.any(valid):
            invalid = (image_current_index < 0) | (image_current_index >= image_num_images)
            if torch.any(invalid & valid):
                raise AssertionError(
                    "image_current_index is out of range for at least one sample: "
                    f"indices={image_current_index.tolist()}, counts={image_num_images.tolist()}"
                )

        flat_indices = offsets + image_current_index
        return [
            (batch_index, int(flat_indices[batch_index].item()))
            for batch_index in range(int(image_num_images.shape[0]))
            if bool(valid[batch_index].item())
        ]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_erp_geometry: torch.FloatTensor | None = None,
        image_num_images: torch.LongTensor | None = None,
        image_current_index: torch.LongTensor | None = None,
        panovggt_pixel_values: torch.Tensor | None = None,
        episode_uid=None,
        memory_steps=None,
        current_odom=None,
        memory_write=True,
        **kwargs,
    ):
        if episode_uid is not None:
            if not hasattr(self, "episode_memory"):
                raise ValueError("Memory metadata supplied to a model without memory")
            if input_ids is None or memory_steps is None or current_odom is None:
                raise ValueError("Memory prefill requires IDs, physical steps and odometry")
            batch_size = input_ids.shape[0]
            if isinstance(episode_uid, str) or any(len(field) != batch_size for field in (episode_uid, memory_steps, current_odom)):
                raise ValueError("Memory metadata must contain one entry per batch sample")
            if len(set(episode_uid)) != batch_size:
                raise ValueError("One batch cannot contain two states from the same episode")
            if labels is not None and torch.any(labels[input_ids.eq(self.config.memory_token_id)] != -100):
                raise ValueError("Memory placeholder labels must be masked with -100")
            self._memory_context = dict(episodes=episode_uid, steps=memory_steps,
                odom=current_odom, write=bool(memory_write), input_ids=input_ids,
                image_num_images=image_num_images, image_current_index=image_current_index)
        if self._panovggt_enabled():
            self._set_runtime_pano_context(
                image_grid_thw=image_grid_thw,
                image_num_images=image_num_images,
                image_current_index=image_current_index,
                image_erp_geometry=image_erp_geometry,
                panovggt_pixel_values=panovggt_pixel_values,
            )
        try:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        finally:
            self._memory_context = None
            self._clear_runtime_pano_context()

    def _load_missing_pano_parameters_post_hook(self, module, incompatible_keys) -> None:
        del module
        missing_keys = set(incompatible_keys.missing_keys)

        panovggt_module = self.panovggt_mlp
        if (
            panovggt_module is not None
            and panovggt_module.enabled
            and any(key.startswith("panovggt_mlp.") for key in missing_keys)
        ):
            panovggt_module.reset_parameters()
        if panovggt_module is not None and panovggt_module.enabled:
            if any(key.startswith("panovggt.") for key in missing_keys):
                self._load_external_panovggt_weights()
            else:
                self._mark_panovggt_weights_ready()

    def _reset_panovggt_parameters_after_pretrained_load(self) -> None:
        if self._panovggt_enabled() and self.panovggt_mlp is not None:
            self.panovggt_mlp.reset_parameters()

    @classmethod
    def _checkpoint_has_panovggt_weights(cls, pretrained_model_name_or_path) -> bool | None:
        return cls._checkpoint_has_any_weights(pretrained_model_name_or_path, cls.PANOVGGT_STATE_KEYS)

    @classmethod
    def _checkpoint_has_any_weights(cls, pretrained_model_name_or_path, state_keys) -> bool | None:
        checkpoint_dir = Path(str(pretrained_model_name_or_path))
        if not checkpoint_dir.exists():
            return None

        index_candidates = (
            checkpoint_dir / "model.safetensors.index.json",
            checkpoint_dir / "pytorch_model.bin.index.json",
        )
        for index_path in index_candidates:
            if not index_path.exists():
                continue
            try:
                weight_map = json.loads(index_path.read_text()).get("weight_map", {})
            except Exception:
                return None
            return all(key in weight_map for key in state_keys)

        safetensors_path = checkpoint_dir / "model.safetensors"
        if safetensors_path.exists():
            try:
                from safetensors import safe_open

                with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
                    keys = set(handle.keys())
                return all(key in keys for key in state_keys)
            except Exception:
                return None

        torch_path = checkpoint_dir / "pytorch_model.bin"
        if torch_path.exists():
            try:
                loaded = torch.load(str(torch_path), map_location="cpu")
                if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                    loaded = loaded["state_dict"]
                if not isinstance(loaded, dict):
                    return None
                return all(key in loaded for key in state_keys)
            except Exception:
                return None

        return None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        has_panovggt_mlp_weights = cls._checkpoint_has_panovggt_weights(pretrained_model_name_or_path)
        if has_panovggt_mlp_weights is False:
            model._reset_panovggt_parameters_after_pretrained_load()
        if model._panovggt_enabled():
            # The encoder and fusion MLP are independent checkpoint components.
            # Always prefer saved encoder weights, even when the MLP needs init.
            loaded_saved_panovggt = model._load_saved_panovggt_weights(pretrained_model_name_or_path)
            if not loaded_saved_panovggt and not model._panovggt_weights_ready:
                model._load_external_panovggt_weights()
        return model

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        image_erp_geometry=None,
        image_num_images=None,
        image_current_index=None,
        panovggt_pixel_values=None,
        episode_uid=None,
        memory_steps=None,
        current_odom=None,
        memory_write=True,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            is_first_iteration=is_first_iteration,
            image_erp_geometry=image_erp_geometry,
            image_num_images=image_num_images,
            image_current_index=image_current_index,
            panovggt_pixel_values=panovggt_pixel_values,
            **kwargs,
        )
        if episode_uid is not None:
            model_inputs.update(episode_uid=episode_uid, memory_steps=memory_steps,
                current_odom=current_odom, memory_write=memory_write)
        if not is_first_iteration and use_cache:
            for key in ("episode_uid", "memory_steps", "current_odom", "memory_write"):
                model_inputs.pop(key, None)
            model_inputs["image_erp_geometry"] = None
            model_inputs["image_num_images"] = None
            model_inputs["image_current_index"] = None
            model_inputs["panovggt_pixel_values"] = None
        return model_inputs
