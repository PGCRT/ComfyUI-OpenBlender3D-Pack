from pathlib import Path
from comfy_env import setup_env, copy_files
from comfy_3d_viewers import copy_viewer

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

setup_env(str(SCRIPT_DIR / "nodes"))

# Copy FBX viewer
copy_viewer("fbx", SCRIPT_DIR / "web")

# Copy assets
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input")
