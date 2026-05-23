# Vendored modules

## `dinov3_vendored.py`

DINOv3 ViT-L as a plain `nn.Module`, vendored from ComfyUI-TRELLIS2.

**Source**: `trellis2/ComfyUI/custom_nodes/ComfyUI-TRELLIS2/nodes/trellis2/dinov3.py`
**Upstream license**: MIT (TRELLIS2's `LICENSE`)
**Why vendored**: `transformers.DINOv3ViTModel` exposes `device` as a read-only
`@property`. ComfyUI's `load_models_gpu` performs `model.device = X` for
bookkeeping, which raises `AttributeError` and breaks our `ModelPatcher` wrap.
A plain `nn.Module` with state-dict keys matching the HF checkpoint sidesteps
the property entirely while loading the same safetensors files.

**Modifications from upstream**:
1. Removed TRELLIS2's `DinoV3FeatureExtractor` wrapper class -- Pixal3D has its
   own equivalent at `nodes/pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`.
2. Replaced `from .attention_sparse import scaled_dot_product_attention` with
   an inline `_sdpa` shim that calls `torch.nn.functional.scaled_dot_product_attention`
   under TRELLIS2's `(B, L, H, D)` calling convention. We don't depend on the
   optional `comfy.attention_sparse` package this way.
3. Added `self.config = SimpleNamespace(**c)` on `DINOv3ViT.__init__` so callers
   that read `self.model.config.patch_size` / `.hidden_size` (Pixal3D's
   `DinoV3ProjFeatureExtractor.__init__` at lines 411-413 of
   `image_conditioned_proj.py`) keep working without modification.
4. Renamed `_load_dinov3_from_safetensors` -> `DinoV3ViT_from_hf_cache` and
   adapted to load via the existing `models/dinov3/` ComfyUI cache or HF fallback.
