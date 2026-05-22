from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from openblender_lazy import component_specs, load_manifest
from comfy_env.packages.cuda_wheels import get_torch_version_for_cuda

manifest = load_manifest()
specs = component_specs()
assert manifest.get("policy", {}).get("lazy_component_registration") is True
assert {s.id for s in specs} >= {"pixal3d", "geometrypack", "camerapack", "multiband"}
for spec in specs:
    assert spec.label
    assert spec.module
    assert spec.path.name == spec.id
assert get_torch_version_for_cuda("12.8") == "2.8"
print(json.dumps({"ok": True, "components": len(specs)}, indent=2))
