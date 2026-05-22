"""Environment cache and path utilities for comfy-env."""

import glob
import hashlib
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import tomli


def _get_default_cache_dir() -> Path:
    """Get platform-specific default cache directory.

    Defaults to an 'OpenBlender-envs' folder next to the portable ComfyUI
    folder to keep OpenBlender isolated environments in one predictable place.
    """
    if sys.platform == "win32":
        python_dir = Path(sys.executable).parent
        portable_root = python_dir.parent if python_dir.name == "python_embeded" else python_dir
        return portable_root / "OpenBlender-envs"
    else:
        return Path.home() / ".comfy-envs"


CACHE_DIR = _get_default_cache_dir()


def get_cache_dir() -> Path:
    """Get cache dir, checking COMFY_ENV_CACHE_DIR env var each time."""
    cache_dir = Path(os.environ.get("COMFY_ENV_CACHE_DIR", _get_default_cache_dir()))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_version() -> str:
    """Get comfy-env version string for cache busting."""
    try:
        from importlib.metadata import version

        return version("comfy-env")
    except Exception:
        return "0.0.0-dev"


def compute_config_hash(config_path: Path) -> str:
    h = hashlib.sha256(config_path.read_bytes())
    h.update(_get_version().encode())
    return h.hexdigest()[:8]


def sanitize_name(name: str) -> str:
    name = name.lower()
    for prefix in ("comfyui-", "comfyui_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.replace("-", "_").replace(" ", "_")


def get_env_name(node_dir: Path, config_path: Path) -> str:
    """Generate env name: <nodename>_<subfolder>_<hash>"""
    node_name = sanitize_name(node_dir.name)
    config_parent = config_path.parent
    if config_parent == node_dir:
        subfolder = ""
    else:
        try:
            subfolder = config_parent.relative_to(node_dir).as_posix().replace("/", "_")
        except ValueError:
            subfolder = sanitize_name(config_parent.name)
    return f"{node_name}_{subfolder}_{compute_config_hash(config_path)}"


def get_local_env_path(main_node_dir: Path, config_path: Path) -> Path:
    """Return path for _env_* symlink in config_path.parent. Hash-only name so identical configs share envs."""
    shared_name = os.environ.get("COMFY_ENV_SHARED_NAME")
    if not shared_name and config_path.exists():
        try:
            with open(config_path, "rb") as f:
                shared_name = tomli.load(f).get("options", {}).get("shared_env_name")
        except Exception:
            shared_name = None

    if shared_name:
        safe_name = sanitize_name(str(shared_name))
        return config_path.parent / f"_env_{safe_name}"

    h = compute_config_hash(config_path)[:6]
    return config_path.parent / f"_env_{h}"


def resolve_env_path(node_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Find _env_* dir in node_dir.

    OB Fork: Also checks cache directory if no junction exists in node_dir.
    """
    # First try to find _env_* in node_dir (for backward compatibility with junctions)
    try:
        for item in node_dir.iterdir():
            if item.name.startswith("_env_") and item.is_dir():
                return _get_env_paths(item)
    except OSError:
        pass

    # OB Fork: Check for .comfy-env-cache-info file (no junction mode)
    cache_info_file = node_dir / ".comfy-env-cache-info"
    if cache_info_file.exists():
        try:
            target_path = Path(cache_info_file.read_text().strip())
            if target_path.exists():
                return _get_env_paths(target_path)
        except Exception:
            pass

    # OB Fork: Try to find environment in cache directory
    cache_dir = get_cache_dir()
    env_name = None
    try:
        for item in node_dir.iterdir():
            if item.name.startswith("_env_"):
                env_name = item.name
                break
    except OSError:
        pass

    if env_name:
        cache_env_path = cache_dir / env_name
        if cache_env_path.exists():
            # Return paths pointing to .pixi/envs/default inside the cache env
            pixi_env = cache_env_path / ".pixi" / "envs" / "default"
            if pixi_env.exists():
                return _get_env_paths(pixi_env)

    return None, None, None


def _get_env_paths(env_path: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    if sys.platform == "win32":
        return env_path, env_path / "Lib" / "site-packages", env_path / "Library" / "bin"
    matches = glob.glob(str(env_path / "lib" / "python*" / "site-packages"))
    return env_path, Path(matches[0]) if matches else None, env_path / "lib"
