"""DINOv3 ViT-L as a plain nn.Module (vendored from ComfyUI-TRELLIS2, MIT).

Why this exists:
    `transformers.DINOv3ViTModel` exposes `device` as a read-only @property.
    ComfyUI's `load_models_gpu` performs `model.device = X` for bookkeeping,
    which raises `AttributeError: property 'device' has no setter` when the
    backbone is wrapped in a `ModelPatcher`. Re-implementing DINOv3 as a
    plain `nn.Module` (with the same state-dict keys as the HF checkpoint)
    sidesteps the property entirely.

Source:
    trellis2/ComfyUI/custom_nodes/ComfyUI-TRELLIS2/nodes/trellis2/dinov3.py
    Adapted: removed TRELLIS2's `DinoV3FeatureExtractor` wrapper (Pixal3D
    has its own equivalent at trainers/flow_matching/mixins/image_conditioned_proj.py);
    inlined SDPA shim in place of `from .attention_sparse` so we don't depend
    on ComfyUI's optional `comfy.attention_sparse` package; added `config`
    SimpleNamespace shim so the existing Pixal3D code that reads
    `self.model.config.patch_size` / `self.model.config.hidden_size`
    keeps working; added `DinoV3ViT_from_hf_cache` for HF cache loading.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
import comfy.utils

ops = comfy.ops.manual_cast


def _sdpa(q, k, v):
    """SDPA shim matching comfy.attention_sparse's (B, L, H, D) calling
    convention. Internally transposes to torch's (B, H, N, D), runs
    F.scaled_dot_product_attention (which auto-routes to flash-attn / xformers
    when available), then transposes back."""
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# Config (hardcoded for ViT-L, matching the safetensors checkpoint)
# ---------------------------------------------------------------------------

VITL_CONFIG = dict(
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=24,
    num_attention_heads=16,
    attention_dropout=0.0,
    layer_norm_eps=1e-6,
    patch_size=16,
    num_channels=3,
    query_bias=True,
    key_bias=False,
    value_bias=True,
    proj_bias=True,
    mlp_bias=True,
    layerscale_value=1e-5,
    drop_path_rate=0.4,
    num_register_tokens=4,
    rope_theta=100.0,
)


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _get_patch_coords(num_h: int, num_w: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Patch center coordinates in [-1, +1]."""
    ch = torch.arange(0.5, num_h, dtype=dtype, device=device) / num_h
    cw = torch.arange(0.5, num_w, dtype=dtype, device=device) / num_w
    coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), dim=-1).flatten(0, 1)
    return 2.0 * coords - 1.0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    """Apply RoPE to q/k, skipping prefix tokens (CLS + register)."""
    n_prefix = q.shape[-2] - cos.shape[-2]
    q_pre, q_patch = q.split((n_prefix, cos.shape[-2]), dim=-2)
    k_pre, k_patch = k.split((n_prefix, cos.shape[-2]), dim=-2)
    q_patch = q_patch * cos + _rotate_half(q_patch) * sin
    k_patch = k_patch * cos + _rotate_half(k_patch) * sin
    return torch.cat((q_pre, q_patch), dim=-2), torch.cat((k_pre, k_patch), dim=-2)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class RoPEEmbedding(nn.Module):
    """Compute cos/sin RoPE embeddings from pixel_values shape."""

    def __init__(self, head_dim: int, patch_size: int, rope_theta: float = 100.0):
        super().__init__()
        self.patch_size = patch_size
        inv_freq = 1.0 / (rope_theta ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor):
        _, _, h, w = pixel_values.shape
        nh, nw = h // self.patch_size, w // self.patch_size
        device = pixel_values.device
        coords = _get_patch_coords(nh, nw, torch.float32, device)
        angles = 2 * math.pi * coords[:, :, None] * self.inv_freq.to(device=device)[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        cos, sin = torch.cos(angles), torch.sin(angles)
        dtype = pixel_values.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class Embeddings(nn.Module):
    def __init__(self, hidden_size, patch_size, num_channels, num_register_tokens, dtype=None, device=None, operations=ops):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size, dtype=dtype, device=device))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size, dtype=dtype, device=device))
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, hidden_size, dtype=dtype, device=device))
        self.patch_embeddings = operations.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, dtype=dtype, device=device)

    def forward(self, pixel_values, bool_masked_pos=None):
        B = pixel_values.shape[0]
        x = self.patch_embeddings(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        if bool_masked_pos is not None:
            x = torch.where(bool_masked_pos.unsqueeze(-1), self.mask_token.to(device=x.device, dtype=x.dtype), x)
        cls = self.cls_token.to(device=x.device, dtype=x.dtype).expand(B, -1, -1)
        reg = self.register_tokens.to(device=x.device, dtype=x.dtype).expand(B, -1, -1)
        return torch.cat([cls, reg, x], dim=1)


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, query_bias, key_bias, value_bias, proj_bias, dtype=None, device=None, operations=ops):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = operations.Linear(hidden_size, hidden_size, bias=query_bias, dtype=dtype, device=device)
        self.k_proj = operations.Linear(hidden_size, hidden_size, bias=key_bias, dtype=dtype, device=device)
        self.v_proj = operations.Linear(hidden_size, hidden_size, bias=value_bias, dtype=dtype, device=device)
        self.o_proj = operations.Linear(hidden_size, hidden_size, bias=proj_bias, dtype=dtype, device=device)

    def forward(self, x, position_embeddings=None):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rope(q, k, cos, sin)
        # _sdpa expects (B, L, H, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = _sdpa(q, k, v)
        return self.o_proj(out.contiguous().reshape(B, N, -1))


class LayerScale(nn.Module):
    def __init__(self, hidden_size, init_value, dtype=None, device=None):
        super().__init__()
        self.lambda1 = nn.Parameter(init_value * torch.ones(hidden_size, dtype=dtype, device=device))

    def forward(self, x):
        return x * self.lambda1.to(device=x.device, dtype=x.dtype)


def _drop_path(x, drop_prob, training):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep) * mask


class MLP(nn.Module):
    """Matches DINOv3ViTMLP key names: mlp.up_proj, mlp.down_proj."""
    def __init__(self, hidden_size, intermediate_size, bias, dtype=None, device=None, operations=ops):
        super().__init__()
        self.up_proj = operations.Linear(hidden_size, intermediate_size, bias=bias, dtype=dtype, device=device)
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, layer_norm_eps,
                 layerscale_value, drop_path_rate, query_bias, key_bias, value_bias,
                 proj_bias, mlp_bias, dtype=None, device=None, operations=ops):
        super().__init__()
        self.norm1 = operations.LayerNorm(hidden_size, eps=layer_norm_eps, dtype=dtype, device=device)
        self.attention = Attention(hidden_size, num_heads, query_bias, key_bias, value_bias, proj_bias, dtype=dtype, device=device, operations=operations)
        self.layer_scale1 = LayerScale(hidden_size, layerscale_value, dtype=dtype, device=device)
        self.drop_path_rate = drop_path_rate

        self.norm2 = operations.LayerNorm(hidden_size, eps=layer_norm_eps, dtype=dtype, device=device)
        self.mlp = MLP(hidden_size, intermediate_size, mlp_bias, dtype=dtype, device=device, operations=operations)
        self.layer_scale2 = LayerScale(hidden_size, layerscale_value, dtype=dtype, device=device)

    def forward(self, x, position_embeddings=None):
        r = x
        x = self.attention(self.norm1(x), position_embeddings=position_embeddings)
        x = self.layer_scale1(x)
        x = _drop_path(x, self.drop_path_rate, self.training) + r

        r = x
        x = self.mlp(self.norm2(x))
        x = self.layer_scale2(x)
        x = _drop_path(x, self.drop_path_rate, self.training) + r
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DINOv3ViT(nn.Module):
    """DINOv3 ViT-L as a plain nn.Module.

    State dict keys match the HuggingFace `DINOv3ViTModel` checkpoint exactly,
    so existing safetensors load with `strict=True`.

    Exposes `.config` as a SimpleNamespace mirroring `VITL_CONFIG` so callers
    that reach for `self.config.patch_size` / `self.config.hidden_size` (as
    Pixal3D's `DinoV3ProjFeatureExtractor.__init__` does) keep working.
    """

    def __init__(self, cfg=None, dtype=None, device=None, operations=ops):
        super().__init__()
        c = {**VITL_CONFIG, **(cfg or {})}
        head_dim = c["hidden_size"] // c["num_attention_heads"]

        self.embeddings = Embeddings(
            c["hidden_size"], c["patch_size"], c["num_channels"], c["num_register_tokens"],
            dtype=dtype, device=device, operations=operations,
        )
        self.rope_embeddings = RoPEEmbedding(head_dim, c["patch_size"], c.get("rope_theta", 100.0))
        self.layer = nn.ModuleList([
            Block(
                c["hidden_size"], c["num_attention_heads"], c["intermediate_size"],
                c["layer_norm_eps"], c["layerscale_value"], c["drop_path_rate"],
                c["query_bias"], c["key_bias"], c["value_bias"], c["proj_bias"], c["mlp_bias"],
                dtype=dtype, device=device, operations=operations,
            )
            for _ in range(c["num_hidden_layers"])
        ])
        self.norm = operations.LayerNorm(c["hidden_size"], eps=c["layer_norm_eps"], dtype=dtype, device=device)

        # SimpleNamespace shim so existing code that reads `self.model.config.X`
        # (e.g. patch_size, hidden_size) doesn't need to change.
        self.config = SimpleNamespace(**c)

    def forward(self, pixel_values, bool_masked_pos=None):
        x = self.embeddings(pixel_values, bool_masked_pos)
        pos = self.rope_embeddings(pixel_values)
        for block in self.layer:
            x = block(x, position_embeddings=pos)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Loader: HuggingFace cache -> vendored DINOv3ViT
# ---------------------------------------------------------------------------

# Gated upstream repos -> public reuploads.
DINOV3_MODEL_REMAP = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m",
}

# Clean local safetensors filenames to check (in order of preference).
LOCAL_SAFETENSORS_NAMES = [
    "dinov3-vitl-pretrain.safetensors",
    "dinov3-vitl.safetensors",
    "model.safetensors",
]


def _find_local_safetensors(cache_dir: str) -> Optional[str]:
    for name in LOCAL_SAFETENSORS_NAMES:
        path = os.path.join(cache_dir, name)
        if os.path.isfile(path):
            return path
    return None


def DinoV3ViT_from_hf_cache(model_name: str) -> DINOv3ViT:
    """Load DINOv3 ViT-L from ComfyUI's `models/dinov3/` cache or HF.

    Resolution order:
      1. Local file at `models/dinov3/<known names>.safetensors`.
      2. `hf_hub_download(remapped(model_name), "model.safetensors", local_dir=cache_dir)`.

    State-dict is loaded via `comfy.utils.load_torch_file` then
    `model.load_state_dict(strict=False)` -- strict=False because some HF
    checkpoints carry pooler/classification heads we don't have. Missing
    keys in the model itself raise (caught by us before returning).
    """
    import folder_paths

    actual_model_name = DINOV3_MODEL_REMAP.get(model_name, model_name)
    cache_dir = os.path.join(folder_paths.models_dir, "dinov3")
    os.makedirs(cache_dir, exist_ok=True)

    local = _find_local_safetensors(cache_dir)
    if local is None:
        from huggingface_hub import hf_hub_download
        hf_hub_download(actual_model_name, "model.safetensors", local_dir=cache_dir)
        local = os.path.join(cache_dir, "model.safetensors")

    state_dict = comfy.utils.load_torch_file(local, safe_load=True)
    model = DINOv3ViT()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(
            f"DINOv3 vendored load: missing keys (model expected, checkpoint didn't supply): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    # `unexpected` is fine -- HF checkpoints often carry pooler/cls heads we don't use.
    model.eval()
    return model
