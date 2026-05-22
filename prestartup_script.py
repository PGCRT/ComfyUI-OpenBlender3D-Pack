"""Fast prestartup for the bundled OpenBlender 3D pack."""

from __future__ import annotations

import json
import os
import shutil
import subprocess as _subprocess
import sys
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACK_DIR / "vendor"

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

_orig_run = _subprocess.run
_orig_popen_init = _subprocess.Popen.__init__

def _patched_run(*args, **kwargs):
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        kwargs.setdefault("errors", "replace")
    return _orig_run(*args, **kwargs)

def _patched_popen_init(self, *args, **kwargs):
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        kwargs.setdefault("errors", "replace")
    return _orig_popen_init(self, *args, **kwargs)

_subprocess.run = _patched_run
_subprocess.Popen.__init__ = _patched_popen_init

try:
    from openblender_lazy import component_specs, prepare_runtime_environment
except Exception:
    component_specs = lambda: []
    prepare_runtime_environment = lambda: None

prepare_runtime_environment()
os.environ.setdefault("COMFY_ENV_METADATA_TRUST_CACHE", "1")

try:
    from comfy_env import get_comfyui_dir, setup_env
    from comfy_3d_viewers import copy_viewer
except Exception as exc:
    print(f"[OpenBlender3D-Pack] prestartup import warning: {exc}", file=sys.stderr)
    get_comfyui_dir = lambda _p: None
    setup_env = lambda _p: None
    copy_viewer = None


def _state_path() -> Path:
    comfyui_dir = get_comfyui_dir(PACK_DIR) if callable(get_comfyui_dir) else None
    portable_root = Path(comfyui_dir).resolve().parent if comfyui_dir else PACK_DIR.parent.parent.resolve().parent
    return portable_root / "OpenBlender-envs" / "env_state.json"


def _cheap_state_summary() -> dict:
    state_file = _state_path()
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            return {"status": "unreadable", "path": str(state_file)}
    return {"status": "missing", "path": str(state_file)}


def _shared_env_status() -> dict:
    try:
        from comfy_env.environment.cache import get_cache_dir, sanitize_name
    except Exception as exc:
        return {"ready": False, "reason": f"cache helpers unavailable: {exc}"}
    shared_name = os.environ.get("COMFY_ENV_SHARED_NAME", "openblender")
    safe_name = sanitize_name(shared_name)
    build_dir = get_cache_dir() / f"_env_{safe_name}"
    python_path = build_dir / ".pixi" / "envs" / "default" / ("python.exe" if sys.platform == "win32" else "bin/python")
    done_marker = build_dir / ".done"
    if not build_dir.exists():
        return {"ready": False, "reason": f"missing shared env build dir: {build_dir}"}
    if not done_marker.exists():
        return {"ready": False, "reason": f"missing done marker: {done_marker}"}
    if not python_path.exists():
        return {"ready": False, "reason": f"missing env python: {python_path}"}
    missing_links = []
    for spec in component_specs():
        for rel in getattr(spec, "env_configs", []):
            cfg = spec.path / rel
            if not cfg.exists():
                continue
            local_env = cfg.parent / f"_env_{safe_name}"
            if not local_env.exists():
                missing_links.append(str(local_env))
    if missing_links:
        return {"ready": False, "reason": "missing component env links", "missing_links": missing_links[:8]}
    return {"ready": True, "reason": "shared env present", "python": str(python_path)}


def _repair_shared_env(reason: str) -> None:
    print(f"[OpenBlender3D-Pack] Isolation env missing/broken; rebuilding ({reason})", file=sys.stderr)
    cmd = [sys.executable, str(PACK_DIR / "install.py"), "--repair", "--all-components"]
    result = _subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[OpenBlender3D-Pack] repair command failed", file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    print("[OpenBlender3D-Pack] Isolation env rebuild completed", file=sys.stderr)

def _sync_web_assets() -> None:
    web_dir = PACK_DIR / "web"
    web_dir.mkdir(exist_ok=True)
    for spec in component_specs():
        component_web = spec.path / "web"
        if component_web.exists():
            shutil.copytree(component_web, web_dir, dirs_exist_ok=True)
        if copy_viewer is not None:
            for viewer in spec.viewers:
                component_web.mkdir(exist_ok=True)
                copy_viewer(viewer, component_web)
                shutil.copytree(component_web, web_dir, dirs_exist_ok=True)


try:
    setup_env(str(PACK_DIR))
except Exception as exc:
    print(f"[OpenBlender3D-Pack] setup_env warning: {exc}", file=sys.stderr)

try:
    force_repair = os.environ.get("OPENBLENDER_REPAIR_ON_STARTUP", "").lower() in ("1", "true", "yes")
    disable_auto_repair = os.environ.get("OPENBLENDER_DISABLE_AUTO_REPAIR", "").lower() in ("1", "true", "yes")
    env_status = _shared_env_status()
    if force_repair:
        _repair_shared_env("forced by OPENBLENDER_REPAIR_ON_STARTUP")
    elif not env_status.get("ready") and not disable_auto_repair:
        _repair_shared_env(str(env_status.get("reason", "unknown")))
    else:
        state = _cheap_state_summary()
        print(
            f"[OpenBlender3D-Pack] Fast prestartup; env_state={state.get('status', 'unknown')}; isolation={env_status.get('reason')}",
            file=sys.stderr,
        )
except Exception as exc:
    print(f"[OpenBlender3D-Pack] startup repair warning: {exc}", file=sys.stderr)

try:
    _sync_web_assets()
except Exception as exc:
    print(f"[OpenBlender3D-Pack] web asset sync warning: {exc}", file=sys.stderr)
print("[OpenBlender3D-Pack] prestartup complete", file=sys.stderr, flush=True)

