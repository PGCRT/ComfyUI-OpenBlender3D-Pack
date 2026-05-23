import os
import site
import sys
import contextlib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=r".*A module that was compiled using NumPy 1.x cannot be run.*",
)

# On Windows, bpy needs its own DLL directory in the search path.
for sp in site.getsitepackages():
    bpy_dir = Path(sp) / "bpy"
    if bpy_dir.exists():
        os.add_dll_directory(str(bpy_dir))
        break

# Import bpy early to lock in its DLLs before trimesh loads conflicting ones.
# Some bpy wheels emit noisy NumPy ABI diagnostics to stderr even when the
# helper can continue; keep that detail in the helper log instead of ComfyUI.
try:
    if os.environ.get("OPENBLENDER_DEBUG_SKINTOKENS_BPY") == "1":
        import bpy  # noqa: F401
    else:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stderr(devnull):
                import bpy  # noqa: F401
except Exception as exc:
    print(f"[skintokens] bpy preload skipped: {type(exc).__name__}: {exc}")

from src.server.bpy_server import run

def main():
    run()

if __name__ == "__main__":
    main()
