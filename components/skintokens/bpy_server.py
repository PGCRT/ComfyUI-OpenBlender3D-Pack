import os
import site
import sys
from pathlib import Path

# On Windows, bpy needs its own DLL directory in the search path.
for sp in site.getsitepackages():
    bpy_dir = Path(sp) / "bpy"
    if bpy_dir.exists():
        os.add_dll_directory(str(bpy_dir))
        break

# Import bpy early to lock in its DLLs before trimesh loads conflicting ones.
import bpy  # noqa: F401

from src.server.bpy_server import run

def main():
    run()

if __name__ == "__main__":
    main()