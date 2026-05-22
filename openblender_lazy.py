"""Manifest-driven lazy registration for ComfyUI-OpenBlender3D-Pack."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACK_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACK_DIR / "vendor"
COMPONENTS_DIR = PACK_DIR / "components"
MANIFEST_PATH = PACK_DIR / "openblender_components.json"

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

log = logging.getLogger("openblender3d")

_COMPONENT_DEFAULTS: dict[str, dict[str, Any]] = {
    "motioncapture": {"label": "MotionCapture", "module": "components.motioncapture", "nodes_package": "nodes", "env_configs": ["nodes/comfy-env.toml"], "viewers": ["fbx", "fbx_compare", "bvh", "fbx_animation", "compare_smpl_bvh", "smpl", "smpl_camera"], "copy_fbx_to_3d": True},
    "sam3dbody": {"label": "SAM3DBody", "module": "components.sam3dbody", "nodes_package": "nodes", "env_configs": ["nodes/comfy-env.toml"], "viewers": ["fbx"]},
    "pixal3d": {"label": "Pixal3D", "module": "components.pixal3d", "nodes_package": "nodes", "env_configs": ["nodes/comfy-env.toml"], "viewers": []},
    "gaussianpack": {"label": "GaussianPack", "module": "components.gaussianpack", "nodes_package": "nodes", "viewers": []},
    "hymotion": {"label": "HY-Motion", "module": "components.hymotion", "nodes_package": "nodes", "viewers": []},
    "camerapack": {"label": "CameraPack", "module": "components.camerapack", "nodes_file": "nodes.py", "viewers": []},
    "multiband": {"label": "Multiband", "module": "components.multiband", "nodes_package": "nodes", "viewers": []},
    "geometrypack": {"label": "GeometryPack", "module": "components.geometrypack", "nodes_package": "nodes", "env_configs": ["nodes/blender/comfy-env.toml", "nodes/gpu/comfy-env.toml", "nodes/main/comfy-env.toml"], "viewers": []},
    "skintokens": {"label": "SkinTokens", "module": "components.skintokens", "nodes_package": "nodes", "env_configs": ["nodes/comfy-env.toml"], "viewers": []},
    "lito": {"label": "LiTo", "module": "components.lito", "nodes_package": "nodes", "env_configs": ["nodes/comfy-env.toml"], "viewers": []},
    "depthanythingv3": {"label": "DepthAnythingV3", "module": "components.depthanythingv3", "nodes_package": "nodes", "env_configs": ["nodes/comfy-env.toml"], "viewers": []},
    "moge2": {"label": "MoGe2", "module": "components.moge2", "nodes_package": "nodes", "env_configs": ["nodes/comfy-env.toml"], "viewers": []},
    "checker": {"label": "Checker", "module": "components.checker", "nodes_file": "nodes.py", "viewers": []},
}

_COMPONENT_ORDER = [
    "motioncapture", "sam3dbody", "pixal3d", "gaussianpack", "hymotion",
    "camerapack", "multiband", "geometrypack", "skintokens", "lito", "depthanythingv3", "moge2", "checker",
]


@dataclass(frozen=True)
class ComponentSpec:
    id: str
    label: str
    module: str
    path: Path
    source: str | None = None
    nodes_package: str = "nodes"
    nodes_file: str | None = None
    viewers: list[str] = field(default_factory=list)
    copy_fbx_to_3d: bool = False
    startup: str = "lazy"
    env_configs: list[str] = field(default_factory=list)

    @property
    def exists(self) -> bool:
        return self.path.exists()


def load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Failed to parse %s", MANIFEST_PATH)
    return {"name": "ComfyUI-OpenBlender3D-Pack", "shared_env_name": "openblender", "components": []}


def component_specs() -> list[ComponentSpec]:
    manifest = load_manifest()
    by_id: dict[str, dict[str, Any]] = {}
    for item in manifest.get("components", []):
        if isinstance(item, dict) and item.get("id"):
            by_id[str(item["id"])] = dict(item)

    specs: list[ComponentSpec] = []
    seen: set[str] = set()
    for cid in [*list(by_id), *[c for c in _COMPONENT_ORDER if c not in by_id]]:
        if cid in seen:
            continue
        seen.add(cid)
        default = _COMPONENT_DEFAULTS.get(cid, {})
        data = {**default, **by_id.get(cid, {})}
        specs.append(ComponentSpec(
            id=cid,
            label=str(data.get("label") or cid),
            module=str(data.get("module") or f"components.{cid}"),
            path=COMPONENTS_DIR / cid,
            source=data.get("source"),
            nodes_package=str(data.get("nodes_package") or "nodes"),
            nodes_file=data.get("nodes_file"),
            viewers=list(data.get("viewers") or []),
            copy_fbx_to_3d=bool(data.get("copy_fbx_to_3d", False)),
            startup=str(data.get("startup") or "lazy"),
            env_configs=list(data.get("env_configs") or []),
        ))
    return specs


def selected_component_specs(selected: set[str] | None = None) -> list[ComponentSpec]:
    specs = component_specs()
    if not selected:
        return specs
    wanted = {s.lower() for s in selected}
    return [s for s in specs if s.id.lower() in wanted]


def _force_vendored_comfy_env() -> None:
    for name in list(sys.modules):
        if name == "comfy_env" or name.startswith("comfy_env."):
            mod = sys.modules.get(name)
            file = getattr(mod, "__file__", "") if mod else ""
            if str(VENDOR_DIR) not in str(file):
                sys.modules.pop(name, None)


def prepare_runtime_environment() -> None:
    _force_vendored_comfy_env()
    manifest = load_manifest()
    os.environ.setdefault("COMFY_ENV_SHARED_NAME", str(manifest.get("shared_env_name") or "openblender"))
    os.environ.setdefault("COMFY_ENV_METADATA_TRUST_CACHE", "1")


def _merge_mappings(target: dict[str, Any], display_target: dict[str, str], mappings: dict[str, Any], display: dict[str, str], label: str) -> None:
    overlap = set(target).intersection(mappings)
    if overlap:
        log.warning("%s overrides existing node ids: %s", label, sorted(overlap))
    target.update(mappings)
    display_target.update(display)


def _import_component(spec: ComponentSpec, root_package: str | None) -> tuple[dict[str, Any], dict[str, str]]:
    errors: list[BaseException] = []
    names = []
    if root_package:
        names.append((f".components.{spec.id}", root_package))
    names.append((f"components.{spec.id}", None))
    names.append((spec.module, None))
    for name, package in names:
        try:
            module = importlib.import_module(name, package=package)
            return getattr(module, "NODE_CLASS_MAPPINGS", {}), getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise errors[-1]
    return {}, {}


def _scan_isolated_source(spec: ComponentSpec, source_dir: Path, package_name: str, env: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    from comfy_env.isolation.metadata import build_proxy_class, fetch_metadata

    meta = fetch_metadata(env_dir=env["env_dir"], node_dir=source_dir, package_name=package_name, working_dir=spec.path, env_vars=env["env_vars"], host_torch_sp=env.get("host_torch_sp"))
    nodes_meta = meta.get("nodes", {})
    display = meta.get("display", {})
    mappings: dict[str, Any] = {}
    sys_path_list = [str(env["sp"]), str(spec.path)]
    if env.get("share_torch") and env.get("host_torch_sp"):
        sys_path_list.insert(0, str(env["host_torch_sp"]))
    lib_path = str(env["lib"]) if env.get("lib") else None
    for name, node_meta in nodes_meta.items():
        mappings[name] = build_proxy_class(
            node_name=name,
            meta=node_meta,
            env_dir=env["env_dir"],
            package_root=spec.path,
            sys_path=sys_path_list,
            lib_path=lib_path,
            env_vars=env["env_vars"],
            health_check_timeout=env["health_check_timeout"],
        )
    return mappings, display


def _isolation_envs_for(spec: ComponentSpec) -> dict[Path, dict[str, Any]]:
    from comfy_env.config.types import DEFAULT_HEALTH_CHECK_TIMEOUT

    envs: dict[Path, dict[str, Any]] = {}
    try:
        import folder_paths
        comfyui_base = folder_paths.base_path
    except Exception:
        comfyui_base = None
    try:
        import tomllib as toml
    except Exception:
        import tomli as toml

    shared_name = os.environ.get("COMFY_ENV_SHARED_NAME", "openblender").lower().replace("-", "_").replace(" ", "_")
    for rel in spec.env_configs:
        cfg = spec.path / rel
        if not cfg.exists():
            continue
        env_dir = cfg.parent / f"_env_{shared_name}"
        if not env_dir.is_dir():
            # Fall back to the first local _env_* folder, but do not resolve Windows
            # junctions here; resolving them costs seconds per component on some
            # Windows installs and the junction path itself works for python/sp/lib.
            try:
                env_dir = next(p for p in cfg.parent.iterdir() if p.name.startswith("_env_") and p.is_dir())
            except StopIteration:
                continue
        if sys.platform == "win32":
            sp = env_dir / "Lib" / "site-packages"
            lib = env_dir / "Library" / "bin"
        else:
            matches = sorted((env_dir / "lib").glob("python*/site-packages"))
            sp = matches[0] if matches else None
            lib = env_dir / "lib"
        if sp is None or not sp.exists():
            continue
        env_vars: dict[str, str] = {}
        health_check_timeout = DEFAULT_HEALTH_CHECK_TIMEOUT
        try:
            with open(cfg, "rb") as f:
                data = toml.load(f)
            env_vars = {str(k): str(v) for k, v in data.get("env_vars", {}).items()}
            health_check_timeout = float(data.get("options", {}).get("health_check_timeout", DEFAULT_HEALTH_CHECK_TIMEOUT))
        except Exception as exc:
            log.warning("Failed to parse %s: %s", cfg, exc)
        if comfyui_base:
            env_vars["COMFYUI_BASE"] = str(comfyui_base)
        envs[cfg.parent] = {
            "env_dir": env_dir,
            "sp": sp,
            "lib": lib,
            "env_vars": env_vars,
            "health_check_timeout": health_check_timeout,
            "share_torch": False,
            "host_torch_sp": None,
        }
    return envs

def _register_component(spec: ComponentSpec, root_package: str | None) -> tuple[dict[str, Any], dict[str, str]]:
    if not spec.exists:
        log.warning("Skipping missing OpenBlender component: %s", spec.id)
        return {}, {}

    nodes_dir = spec.path / spec.nodes_package
    envs = _isolation_envs_for(spec)
    mappings: dict[str, Any] = {}
    display: dict[str, str] = {}

    if nodes_dir.is_dir():
        if nodes_dir in envs:
            log.info("Lazy-registering %s from cached isolation metadata", spec.label)
            return _scan_isolated_source(spec, nodes_dir, spec.nodes_package, envs[nodes_dir])
        if envs:
            for subdir in sorted(nodes_dir.iterdir()):
                if not subdir.is_dir() or not (subdir / "__init__.py").exists():
                    continue
                if subdir.name.startswith("_") or subdir.name.startswith("."):
                    continue
                env = envs.get(subdir)
                if env is None:
                    continue
                sub_mappings, sub_display = _scan_isolated_source(spec, subdir, f"{spec.nodes_package}.{subdir.name}", env)
                _merge_mappings(mappings, display, sub_mappings, sub_display, f"{spec.label}/{subdir.name}")
            if mappings:
                return mappings, display

    if spec.env_configs:
        log.error("No usable isolation env for %s; startup repair should rebuild it", spec.label)
        return {}, {}
    return _import_component(spec, root_package)


def register_manifest_components(root_package: str | None) -> tuple[dict[str, Any], dict[str, str]]:
    prepare_runtime_environment()
    from comfy_env.isolation.wrap import _cleanup_stale_workers
    _cleanup_stale_workers()

    all_mappings: dict[str, Any] = {}
    all_display: dict[str, str] = {}
    for spec in component_specs():
        try:
            mappings, display = _register_component(spec, root_package)
            _merge_mappings(all_mappings, all_display, mappings, display, spec.label)
            log.info("Registered %s: %d nodes", spec.label, len(mappings))
        except Exception:
            log.exception("Failed to register OpenBlender component: %s", spec.label)
    return all_mappings, all_display



