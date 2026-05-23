"""Pixal3D package bootstrap.

This pack may run against a shared env that has base wheel names (e.g. cumesh,
o_voxel) instead of variant names (cumesh_vb, o_voxel_vb_ap). We expose strict,
explicit aliases here so submodules importing variant names remain compatible.
"""

from __future__ import annotations

import importlib
import logging
import sys

log = logging.getLogger("pixal3d")


def _alias_module(primary: str, fallback: str) -> None:
    if primary in sys.modules:
        return
    try:
        mod = importlib.import_module(primary)
        sys.modules[primary] = mod
        return
    except Exception:
        pass
    try:
        mod = importlib.import_module(fallback)
        sys.modules[primary] = mod
        log.warning("[pixal3d] Using fallback module alias: %s -> %s", primary, fallback)
    except Exception:
        # Let the real import fail later with full traceback.
        pass


_alias_module("cumesh_vb", "cumesh")
_alias_module("o_voxel_vb_ap", "o_voxel")
_alias_module("flex_gemm_ap", "flex_gemm")

from . import models
from . import modules
from . import pipelines
from . import representations
from . import utils
