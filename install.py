"""Installer, verifier, and repair entrypoint for OpenBlender 3D Pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PACK_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PACK_DIR / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

from comfy_env.config import CONFIG_FILE_NAME
from comfy_env.environment.cache import get_cache_dir
from comfy_env.install import install as comfy_install
from openblender_lazy import load_manifest, selected_component_specs

PIXAL3D_IMPORT_GROUPS = {
    "required": {
        "flex_gemm": ["flex_gemm_ap", "flex_gemm"],
        "cumesh": ["cumesh_vb", "cumesh"],
        "o_voxel": ["o_voxel_vb_ap", "o_voxel"],
    },
    "attention": {
        "flash_attn": ["flash_attn"],
        "flash_attn_interface": ["flash_attn_interface"],
    },
    "optional": {
        "drtk": ["drtk"],
        "natten": ["natten"],
        "nvdiffrast.torch": ["nvdiffrast.torch"],
        "nvdiffrec_render": ["nvdiffrec_render"],
        "triton": ["triton"],
    },
}

NATTEN_RELEASE_BASE = "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/natten-latest"


def _run_python(env_python: Path, code: str, timeout: int = 30) -> dict[str, Any]:
    if not env_python.exists():
        return {"ok": False, "detail": f"missing python: {env_python}"}
    try:
        from comfy_env.isolation.wrap import build_isolation_env
        proc_env = build_isolation_env(env_python)
    except Exception:
        proc_env = None
    result = subprocess.run([str(env_python), "-c", code], capture_output=True, text=True, timeout=timeout, env=proc_env)
    return {"ok": result.returncode == 0, "stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "returncode": result.returncode}


def _env_python(shared_name: str) -> Path:
    return get_cache_dir() / f"_env_{shared_name}" / ".pixi" / "envs" / "default" / ("python.exe" if sys.platform == "win32" else "bin/python")


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _component_config_state(spec) -> list[dict[str, Any]]:
    rows = []
    for cfg in sorted(spec.path.rglob(CONFIG_FILE_NAME)):
        rows.append({"path": str(cfg.relative_to(PACK_DIR)), "sha256": _hash_file(cfg)})
    return rows


def _import_group_status(env_python: Path, groups: dict[str, list[str]]) -> dict[str, Any]:
    code = r'''
import importlib, json
checks = json.loads(r"""__GROUPS__""")
out = {}
for label, names in checks.items():
    detail = []
    ok = False
    imported = None
    for name in names:
        try:
            mod = importlib.import_module(name)
            imported = name
            ok = True
            version = getattr(mod, "__version__", "")
            extra = ""
            if name == "natten":
                extra = f" HAS_LIBNATTEN={getattr(mod, 'HAS_LIBNATTEN', 'unknown')}"
            detail.append(f"{name} OK {version}{extra}".strip())
            break
        except Exception as exc:
            detail.append(f"{name}: {type(exc).__name__}: {exc}")
    out[label] = {"ok": ok, "imported": imported, "detail": "; ".join(detail)}
print(json.dumps(out))
'''.replace("__GROUPS__", json.dumps(groups))
    result = _run_python(env_python, code)
    if not result["ok"]:
        return {"_error": result}
    try:
        return json.loads(result["stdout"])
    except Exception:
        return {"_error": result}


def _runtime_tags(env_python: Path) -> dict[str, Any]:
    code = r'''
import json, re, sys
out = {"python": sys.version.split()[0], "cp_tag": f"cp{sys.version_info.major}{sys.version_info.minor}"}
try:
    import torch
    out["torch"] = getattr(torch, "__version__", "")
    out["cuda"] = getattr(torch.version, "cuda", None)
except Exception as exc:
    out["torch_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
'''
    result = _run_python(env_python, code)
    if not result.get("ok"):
        return {"ok": False, "reason": "runtime probe failed", "detail": result}
    try:
        data = json.loads(result.get("stdout") or "{}")
    except Exception:
        return {"ok": False, "reason": "runtime probe parse failed", "detail": result}
    torch_ver = str(data.get("torch") or "")
    cuda_ver = str(data.get("cuda") or "")
    cp_tag = str(data.get("cp_tag") or "")
    m_torch = re.match(r"^(\d+)\.(\d+)", torch_ver)
    m_cuda = re.match(r"^(\d+)\.(\d+)", cuda_ver)
    if not (m_torch and m_cuda and cp_tag):
        return {
            "ok": False,
            "reason": "missing torch/cuda/python tags",
            "detail": {"torch": torch_ver, "cuda": cuda_ver, "cp_tag": cp_tag},
        }
    tmaj, tmin = m_torch.group(1), m_torch.group(2)
    cmaj, cmin = m_cuda.group(1), m_cuda.group(2)
    return {
        "ok": True,
        "python": data.get("python"),
        "cp_tag": cp_tag,
        "torch": torch_ver,
        "torch_short": f"{tmaj}.{tmin}",
        "torch_token_options": [f"{tmaj}.{tmin}", f"{tmaj}{tmin}"],
        "cuda": cuda_ver,
        "cuda_token_options": [f"{cmaj}{cmin}", f"{cmaj}{cmin.zfill(1)}"],
    }


def _natten_wheel_candidates(tags: dict[str, Any]) -> list[str]:
    cp = tags["cp_tag"]
    urls: list[str] = []
    seen: set[str] = set()
    for cu in tags.get("cuda_token_options", []):
        for ttoken in tags.get("torch_token_options", []):
            name = f"natten-0.21.6+cu{cu}torch{ttoken}-{cp}-{cp}-win_amd64.whl"
            url = f"{NATTEN_RELEASE_BASE}/{name}"
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def _pip_install(env_python: Path, target: str, timeout: int = 120) -> dict[str, Any]:
    cmd = [str(env_python), "-m", "pip", "install", "--no-deps", target]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": (result.stdout or "").strip(),
        "stderr": (result.stderr or "").strip(),
        "cmd": cmd,
    }


def ensure_pixal3d_required_modules(env_python: Path, dry_run: bool) -> dict[str, Any]:
    required_groups = PIXAL3D_IMPORT_GROUPS["required"]
    status = _import_group_status(env_python, required_groups)
    if status and not status.get("_error") and all(v.get("ok") for v in status.values()):
        return {"status": "OK", "reason": "all required imports available"}

    attempts: list[dict[str, Any]] = []
    for label, names in required_groups.items():
        grp = status.get(label, {}) if isinstance(status, dict) else {}
        if grp.get("ok"):
            continue
        installed = False
        for pkg in names:
            if dry_run:
                attempts.append({"group": label, "package": pkg, "status": "DRY_RUN"})
                installed = True
                break
            res = _pip_install(env_python, pkg)
            attempts.append({
                "group": label,
                "package": pkg,
                "ok": res.get("ok"),
                "returncode": res.get("returncode"),
                "stderr_tail": (res.get("stderr") or "")[-300:],
            })
            if res.get("ok"):
                installed = True
                break
        if not installed and not dry_run:
            continue

    post = _import_group_status(env_python, required_groups)
    if post and not post.get("_error") and all(v.get("ok") for v in post.values()):
        return {"status": "OK", "reason": "required imports resolved", "attempts": attempts}
    return {"status": "BROKEN", "reason": "missing required pixal3d modules", "attempts": attempts, "post": post}


def ensure_natten_for_pixal3d(env_python: Path, dry_run: bool) -> dict[str, Any]:
    if sys.platform != "win32":
        return {"status": "SKIPPED", "reason": "natten auto-wheel resolver currently enabled for Windows only"}
    already = _import_group_status(env_python, {"natten": ["natten"]}).get("natten", {})
    if already.get("ok"):
        return {"status": "OK", "reason": "already installed", "detail": already.get("detail")}

    tags = _runtime_tags(env_python)
    if not tags.get("ok"):
        return {"status": "SKIPPED", "reason": "cannot resolve runtime tags", "detail": tags}

    candidates = _natten_wheel_candidates(tags)
    if dry_run:
        return {"status": "DRY_RUN", "reason": "would try candidate wheels", "candidates": candidates}

    attempts = []
    for url in candidates:
        res = _pip_install(env_python, url)
        attempts.append({"url": url, "ok": res.get("ok"), "returncode": res.get("returncode"), "stderr_tail": (res.get("stderr") or "")[-300:]})
        if res.get("ok"):
            check = _import_group_status(env_python, {"natten": ["natten"]}).get("natten", {})
            if check.get("ok"):
                return {"status": "OK", "reason": f"installed from {url}", "attempts": attempts, "detail": check.get("detail")}
    return {
        "status": "SKIPPED",
        "reason": "no compatible natten wheel installed",
        "tags": tags,
        "attempts": attempts,
    }


def verify(selected: set[str] | None, write_state: bool) -> dict[str, Any]:
    manifest = load_manifest()
    shared_name = str(manifest.get("shared_env_name") or "openblender")
    env_python = _env_python(shared_name)
    state: dict[str, Any] = {
        "status": "OK",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shared_env_name": shared_name,
        "env_python": str(env_python),
        "components": {},
    }
    runtime = _run_python(env_python, "import sys, torch, json; print(json.dumps({'python': sys.version.split()[0], 'torch': getattr(torch, '__version__', None), 'cuda': getattr(torch.version, 'cuda', None), 'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))")
    state["runtime"] = runtime
    if not runtime.get("ok"):
        state["status"] = "BROKEN"

    for spec in selected_component_specs(selected):
        cstate = {"label": spec.label, "exists": spec.exists, "configs": _component_config_state(spec), "status": "OK" if spec.exists else "MISSING"}
        if spec.id == "pixal3d" and not runtime.get("ok"):
            cstate["status"] = "BROKEN"
            cstate["runtime_error"] = runtime
            state["status"] = "BROKEN"
        elif spec.id == "pixal3d" and runtime.get("ok"):
            required = _import_group_status(env_python, PIXAL3D_IMPORT_GROUPS["required"])
            attention = _import_group_status(env_python, PIXAL3D_IMPORT_GROUPS["attention"])
            optional = _import_group_status(env_python, PIXAL3D_IMPORT_GROUPS["optional"])
            cstate.update({"required": required, "attention": attention, "optional": optional})
            required_ok = required and not required.get("_error") and all(v.get("ok") for v in required.values())
            attention_ok = attention and not attention.get("_error") and any(v.get("ok") for v in attention.values())
            natten = optional.get("natten", {}) if isinstance(optional, dict) else {}
            natten_degraded = natten.get("ok") and "HAS_LIBNATTEN=True" not in natten.get("detail", "")
            if not required_ok or not attention_ok:
                cstate["status"] = "BROKEN"
                state["status"] = "BROKEN"
            elif natten_degraded or not natten.get("ok"):
                cstate["status"] = "DEGRADED"
                cstate["natten_status"] = ensure_natten_for_pixal3d(env_python, dry_run=True)
                if state["status"] == "OK":
                    state["status"] = "DEGRADED"
        state["components"][spec.id] = cstate

    if write_state:
        out = get_cache_dir() / "env_state.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(state, indent=2), encoding="utf-8")
        state["state_path"] = str(out)
    return state


def ensure_main_python_trimesh(dry_run: bool) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "trimesh"]
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[OpenBlender3D-Pack] WARNING: failed to install trimesh in main embedded Python")
        if result.stderr:
            print(result.stderr)


def _component_missing_or_empty(spec) -> bool:
    p = Path(spec.path)
    if not p.exists():
        return True
    try:
        return not any(p.iterdir())
    except Exception:
        return True


def ensure_component_sources(selected: set[str] | None, dry_run: bool) -> None:
    missing = []
    for spec in selected_component_specs(selected):
        if _component_missing_or_empty(spec):
            missing.append(spec.id)

    if not missing:
        return
    if dry_run:
        print(f"[OpenBlender3D-Pack] DRY_RUN: missing bundled components: {', '.join(sorted(missing))}")
        return
    raise RuntimeError(
        "Missing bundled OpenBlender components: "
        + ", ".join(sorted(missing))
        + ". Reinstall ComfyUI-OpenBlender3D-Pack from a complete release bundle."
    )


def repair(selected: set[str] | None, dry_run: bool) -> None:
    ensure_main_python_trimesh(dry_run)
    ensure_component_sources(selected, dry_run)
    sample_config = next(PACK_DIR.rglob(CONFIG_FILE_NAME), None)
    if sample_config is None:
        raise FileNotFoundError(f"No {CONFIG_FILE_NAME} found under {PACK_DIR}")
    print(f"[OpenBlender3D-Pack] Repairing shared isolation env from {sample_config}")
    comfy_install(config=sample_config, node_dir=PACK_DIR, dry_run=dry_run)
    manifest = load_manifest()
    shared_name = str(manifest.get("shared_env_name") or "openblender")
    env_python = _env_python(shared_name)
    pixal3d_modules = ensure_pixal3d_required_modules(env_python, dry_run=dry_run)
    print(f"[OpenBlender3D-Pack] Pixal3D required modules: {json.dumps(pixal3d_modules, ensure_ascii=False)}")
    natten_result = ensure_natten_for_pixal3d(env_python, dry_run=dry_run)
    print(f"[OpenBlender3D-Pack] Pixal3D natten resolver: {json.dumps(natten_result, ensure_ascii=False)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenBlender3D-Pack installer/verifier")
    parser.add_argument("--verify", action="store_true", help="Verify env state without installing")
    parser.add_argument("--repair", action="store_true", help="Install or repair isolation envs")
    parser.add_argument("--component", action="append", help="Limit verify/reporting to a component id")
    parser.add_argument("--all-components", action="store_true", help="Operate on all components")
    parser.add_argument("--dry-run", action="store_true", help="Print intended install actions without changing packages")
    args = parser.parse_args(argv)

    selected = set(args.component or []) if not args.all_components else None
    if not args.verify and not args.repair:
        args.repair = True
        args.verify = True
        args.all_components = True
        selected = None

    if args.repair:
        repair(selected, args.dry_run)
    if args.verify:
        state = verify(selected, write_state=not args.dry_run)
        print(json.dumps(state, indent=2))
        return 0 if state.get("status") in ("OK", "DEGRADED") else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


