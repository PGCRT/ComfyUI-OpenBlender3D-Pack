"""OpenBlender 3D Pack: curated bundled ComfyUI nodes for OpenBlender."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACK_DIR / "vendor"

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

log = logging.getLogger("openblender3d")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
warnings.filterwarnings("ignore", message=r".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", message=r".*cache-system uses symlinks by default.*")


class _SubprocessModelSpamFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Requested to load SubprocessModel" in msg:
            return False
        return True


_spam_filter = _SubprocessModelSpamFilter()
logging.getLogger().addFilter(_spam_filter)
logging.getLogger("comfy.model_management").addFilter(_spam_filter)

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
