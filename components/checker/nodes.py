import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", (name or "").strip().lower())


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except Exception as e:
        return 1, "", str(e)


def _cpu_id() -> str:
    if os.name == "nt":
        code, out, _ = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ]
        )
        if code == 0 and out:
            line = out.splitlines()[0].strip()
            if line:
                return line
        code, out, _ = _run(["wmic", "cpu", "get", "Name"])
        if code == 0 and out:
            lines = [ln.strip() for ln in out.splitlines() if ln.strip() and "Name" not in ln]
            if lines:
                return lines[0]
        code, out, _ = _run(["wmic", "cpu", "get", "ProcessorId"])
        if code == 0 and out:
            lines = [ln.strip() for ln in out.splitlines() if ln.strip() and "ProcessorId" not in ln]
            if lines:
                return lines[0]
    return platform.processor() or "unknown"


def _ram_available_gb() -> str:
    try:
        import psutil
        gb = psutil.virtual_memory().available / (1024 ** 3)
        return f"{gb:.2f} GB"
    except Exception:
        return "unknown"


def _nvidia_smi_summary() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,memory.total,memory.free,driver_version",
        "--format=csv,noheader",
    ]
    code, out, err = _run(cmd)
    if code != 0:
        return f"unavailable ({err or 'nvidia-smi failed'})"
    rows = [r.strip() for r in out.splitlines() if r.strip()]
    if not rows:
        return "unavailable (no GPU rows)"
    return " | ".join(rows)


def _comfyui_version() -> str:
    try:
        import comfyui_version
        v = getattr(comfyui_version, "__version__", None)
        rev = getattr(comfyui_version, "__git_hash__", None)
        if v and rev:
            return f"{v} ({rev})"
        return str(v or "unknown")
    except Exception:
        return "unknown"


def _comfyui_location() -> str:
    try:
        import folder_paths
        return str(Path(folder_paths.base_path).resolve())
    except Exception:
        return str(Path(__file__).resolve().parents[3])


def _embedded_runtime() -> dict:
    result = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch": "missing",
        "cuda": "missing",
    }
    try:
        import torch
        result["pytorch"] = getattr(torch, "__version__", "unknown")
        result["cuda"] = getattr(getattr(torch, "version", object()), "cuda", None) or "none"
    except Exception:
        pass
    return result


def _find_isolated_python(pack_dir: Path) -> str | None:
    preferred = sorted(pack_dir.glob("components/**/_env_openblender/python.exe"))
    if preferred:
        p = preferred[0]
        real = Path(os.path.realpath(str(p)))
        return str(real if real.exists() else p)
    fallback = sorted(pack_dir.glob("components/**/_env_*/python.exe"))
    if fallback:
        p = fallback[0]
        real = Path(os.path.realpath(str(p)))
        return str(real if real.exists() else p)
    return None


def _python_json(python_exe: str, code: str) -> dict:
    rc, out, err = _run([python_exe, "-c", code])
    if rc != 0:
        return {"ok": False, "error": err or out or f"exit code {rc}"}
    try:
        return {"ok": True, "data": json.loads(out)}
    except Exception:
        return {"ok": False, "error": f"invalid json output: {out}"}


def _isolated_runtime(python_exe: str | None) -> dict:
    if not python_exe:
        return {"python": "missing", "pytorch": "missing", "cuda": "missing", "path": "missing"}
    code = """import json
import sys
o = {'python': sys.version.split()[0], 'pytorch': 'missing', 'cuda': 'missing'}
try:
    import torch
    o['pytorch'] = getattr(torch, '__version__', 'unknown')
    o['cuda'] = getattr(getattr(torch, 'version', object()), 'cuda', None) or 'none'
except Exception:
    pass
print(json.dumps(o))
"""
    info = _python_json(python_exe, code)
    if not info.get("ok"):
        return {"python": "error", "pytorch": "error", "cuda": "error", "path": python_exe, "error": info["error"]}
    data = info["data"]
    data["path"] = python_exe
    return data


def _extract_embedded_requirements(pack_dir: Path) -> list[str]:
    req_file = pack_dir / "requirements.txt"
    names: list[str] = ["tomli", "tomli-w"]
    if not req_file.exists():
        return names
    for raw in req_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        name = re.split(r"[<>=!~ ;@\\[]", line, maxsplit=1)[0].strip()
        if name:
            names.append(name)
    seen = set()
    out = []
    for n in names:
        c = _canonical(n)
        if c and c not in seen:
            seen.add(c)
            out.append(n)
    return out


def _installed_distributions(python_exe: str | None) -> set[str]:
    if python_exe is None:
        try:
            import importlib.metadata as md
            return {_canonical(d.metadata.get("Name", "")) for d in md.distributions()}
        except Exception:
            return set()
    code = (
        "import json,re,importlib.metadata as m;"
        "c=lambda s: re.sub(r'[-_.]+','-',(s or '').strip().lower());"
        "print(json.dumps(sorted({c(d.metadata.get('Name','')) for d in m.distributions() if d.metadata.get('Name')})))"
    )
    info = _python_json(python_exe, code)
    if not info.get("ok"):
        return set()
    return set(info["data"])


def _extract_isolated_expected_groups(pack_dir: Path) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {"cuda": set(), "pypi": set()}
    for cf in pack_dir.glob("components/**/comfy-env.toml"):
        try:
            import tomllib
            data = tomllib.loads(cf.read_text(encoding="utf-8"))
        except Exception:
            continue
        pypi = data.get("pypi-dependencies", {})
        if isinstance(pypi, dict):
            groups["pypi"].update(_canonical(k) for k in pypi.keys())
        cuda = data.get("cuda", {})
        if isinstance(cuda, dict):
            pkgs = cuda.get("packages", [])
            if isinstance(pkgs, list):
                groups["cuda"].update(_canonical(str(k)) for k in pkgs)
    groups["cuda"].discard("")
    groups["pypi"].discard("")
    return groups


class OpenBlenderChecker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "include_package_lists": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = "OpenBlender/Checker"

    def run(self, include_package_lists=True):
        pack_dir = Path(__file__).resolve().parents[2]
        embedded_python = sys.executable
        comfy_location = _comfyui_location()
        embedded_runtime = _embedded_runtime()

        isolated_python = _find_isolated_python(pack_dir)
        isolated_runtime = _isolated_runtime(isolated_python)

        embedded_expected = sorted({_canonical(x): x for x in _extract_embedded_requirements(pack_dir)}.values(), key=str.lower)
        embedded_installed = _installed_distributions(None)
        embedded_missing = [n for n in embedded_expected if _canonical(n) not in embedded_installed]
        # comfy_env can be provided by vendored source, not only by pip package.
        vendored_comfy_env = (pack_dir / "vendor" / "comfy_env" / "__init__.py").exists()
        if not vendored_comfy_env and "comfy-env" not in [m.lower() for m in embedded_missing]:
            embedded_missing.append("comfy-env")

        isolated_groups = _extract_isolated_expected_groups(pack_dir)
        isolated_expected = sorted(isolated_groups["cuda"] | isolated_groups["pypi"])
        isolated_installed = _installed_distributions(isolated_python) if isolated_python else set()
        if isolated_runtime.get("error"):
            isolated_missing = [f"unavailable ({isolated_runtime['error']})"]
            isolated_missing_cuda = list(isolated_missing)
            isolated_missing_pypi = list(isolated_missing)
        else:
            isolated_missing = [n for n in isolated_expected if n and n not in isolated_installed]
            isolated_missing_cuda = [n for n in sorted(isolated_groups["cuda"]) if n not in isolated_installed]
            isolated_missing_pypi = [n for n in sorted(isolated_groups["pypi"]) if n not in isolated_installed]

        lines = [
            "OpenBlender Checker",
            "",
            f"- OS: {platform.platform()}",
            f"- CPU ID: {_cpu_id()}",
            f"- System RAM available: {_ram_available_gb()}",
            f"- NVIDIA SMI: {_nvidia_smi_summary()}",
            f"- ComfyUI location: {comfy_location}",
            f"- ComfyUI version: {_comfyui_version()}",
            f"- ComfyUI embedded python version: {embedded_runtime['python']}",
            f"- ComfyUI embedded pytorch version: {embedded_runtime['pytorch']}",
            f"- ComfyUI embedded cuda version: {embedded_runtime['cuda']}",
            f"- Isolated environment python version: {isolated_runtime.get('python', 'missing')}",
            f"- Isolated environment pytorch version: {isolated_runtime.get('pytorch', 'missing')}",
            f"- Isolated environment cuda version: {isolated_runtime.get('cuda', 'missing')}",
            "",
            f"- Missing ComfyUI embedded dependencies: {', '.join(embedded_missing) if embedded_missing else 'none'}",
            f"- Missing Isolated environment dependencies: {', '.join(isolated_missing) if isolated_missing else 'none'}",
            f"- Embedded checked package count: {len(embedded_expected)}",
            f"- Isolated checked package count: {len(isolated_expected)}",
            f"- Isolated CUDA package count: {len(isolated_groups['cuda'])}",
            f"- Isolated PyPI package count: {len(isolated_groups['pypi'])}",
            f"- Missing Isolated CUDA dependencies: {', '.join(isolated_missing_cuda) if isolated_missing_cuda else 'none'}",
            f"- Missing Isolated PyPI dependencies: {', '.join(isolated_missing_pypi) if isolated_missing_pypi else 'none'}",
            "",
            f"- Embedded python path: {embedded_python}",
            f"- Isolated python path: {isolated_runtime.get('path', 'missing')}",
        ]
        if include_package_lists:
            lines.extend(
                [
                    "",
                    "Embedded checked packages:",
                    ", ".join(embedded_expected) if embedded_expected else "none",
                    "",
                    "Isolated checked packages:",
                    ", ".join(isolated_expected) if isolated_expected else "none",
                    "",
                    "Isolated CUDA checked packages:",
                    ", ".join(sorted(isolated_groups["cuda"])) if isolated_groups["cuda"] else "none",
                    "",
                    "Isolated PyPI checked packages:",
                    ", ".join(sorted(isolated_groups["pypi"])) if isolated_groups["pypi"] else "none",
                ]
            )

        report = "\n".join(lines)
        print("[OpenBlender Checker]\n" + report)
        return (report,)


NODE_CLASS_MAPPINGS = {
    "OpenBlenderChecker": OpenBlenderChecker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenBlenderChecker": "Checker",
}
