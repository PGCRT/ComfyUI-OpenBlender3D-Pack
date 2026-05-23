"""OpenBlender 3D Pack: curated bundled ComfyUI nodes for OpenBlender."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACK_DIR / "vendor"

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

log = logging.getLogger("openblender3d")

try:
    from .download_progress import install_hf_progress_patch
except Exception:
    from download_progress import install_hf_progress_patch

install_hf_progress_patch()

try:
    from .openblender_lazy import register_manifest_components
except Exception:
    from openblender_lazy import register_manifest_components

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_manifest_components(__name__)
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
