import atexit
import contextlib
import logging
import os
import signal
import site
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", message=r".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=r".*repetition_penalty.*inputs_embeds.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

COMPONENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
DEFAULT_VAE = "experiments/skin_vae_2_10_32768/last.ckpt"
REPO_ID = "VAST-AI/SkinTokens"
LLM_REPO = "Qwen/Qwen3-0.6B"

_MODEL = None
_TOKENIZER = None
_TRANSFORM = None
_CURRENT_MODEL_KEY = None
_BPY_PROC = None
_BPY_LOG = None
_BPY_DLL_HANDLES = []


@contextlib.contextmanager
def _quiet_hf_logging():
    noisy_loggers = ("httpx", "httpcore", "huggingface_hub")
    old_levels = {name: logging.getLogger(name).level for name in noisy_loggers}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*unauthenticated requests to the HF Hub.*"
        )
        warnings.filterwarnings("ignore", message=r".*deprecated.*")
        try:
            for name in noisy_loggers:
                logging.getLogger(name).setLevel(logging.ERROR)
            yield
        finally:
            for name, level in old_levels.items():
                logging.getLogger(name).setLevel(level)


def _ensure_repo_on_path() -> None:
    component = str(COMPONENT_DIR)
    if component not in sys.path:
        sys.path.insert(0, component)


def _ensure_bpy_dll_path() -> None:
    """Ensure Windows can resolve bpy-dependent DLLs in direct-load mode."""
    global _BPY_DLL_HANDLES
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    if _BPY_DLL_HANDLES:
        return

    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    candidates.extend(sys.path)

    seen = set()
    for base in candidates:
        if not base:
            continue
        bpy_dir = Path(base) / "bpy"
        if not bpy_dir.exists():
            continue
        key = str(bpy_dir).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            # Keep handles alive for process lifetime.
            _BPY_DLL_HANDLES.append(os.add_dll_directory(str(bpy_dir)))
        except Exception:
            continue


def _resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate

    for base in _candidate_bases():
        resolved = (base / path).resolve()
        if resolved.exists():
            return resolved

    return (COMPONENT_DIR / path).resolve()


def _candidate_bases():
    yield COMPONENT_DIR
    try:
        import folder_paths

        yield Path(folder_paths.get_input_directory())
        yield Path(folder_paths.get_input_directory()) / "3d"
        yield Path(folder_paths.get_output_directory())
    except Exception:
        pass


def _download_file(filename: str) -> Path:
    target = COMPONENT_DIR / filename
    if target.exists():
        return target

    from huggingface_hub import hf_hub_download

    with _quiet_hf_logging():
        return Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=str(COMPONENT_DIR),
            )
        )


def _ensure_model_files(model_ckpt: str) -> Path:
    ckpt = _resolve_path(model_ckpt)
    if not ckpt.exists() and model_ckpt == DEFAULT_CKPT:
        ckpt = _download_file(DEFAULT_CKPT)

    vae = _resolve_path(DEFAULT_VAE)
    if not vae.exists():
        _download_file(DEFAULT_VAE)

    llm_dir = COMPONENT_DIR / "models" / "Qwen3-0.6B"
    if not llm_dir.exists():
        from huggingface_hub import snapshot_download

        with _quiet_hf_logging():
            snapshot_download(
                repo_id=LLM_REPO,
                local_dir=str(llm_dir),
                ignore_patterns=["*.bin", "*.safetensors"],
            )

    if not ckpt.exists():
        raise FileNotFoundError(f"SkinTokens checkpoint not found: {model_ckpt}")
    return ckpt


def _load_model(model_ckpt: str, hf_path: Optional[str]):
    global _MODEL, _TOKENIZER, _TRANSFORM, _CURRENT_MODEL_KEY

    _ensure_repo_on_path()
    ckpt = _ensure_model_files(model_ckpt)
    key = (str(ckpt), hf_path or None)
    if _MODEL is not None and key == _CURRENT_MODEL_KEY:
        return _MODEL, _TOKENIZER, _TRANSFORM

    from src.data.transform import Transform
    from src.server.spec import get_model
    from src.tokenizer.parse import get_tokenizer
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    old_cwd = Path.cwd()
    os.chdir(COMPONENT_DIR)
    try:
        _MODEL = get_model(str(ckpt), hf_path=hf_path or None)
    finally:
        os.chdir(old_cwd)

    _TOKENIZER = get_tokenizer(**_MODEL.tokenizer_config)
    _TRANSFORM = Transform.parse(**_MODEL.transform_config["predict_transform"])
    _CURRENT_MODEL_KEY = key
    return _MODEL, _TOKENIZER, _TRANSFORM


def _post_bpy_payload(endpoint: str, payload):
    from src.server.spec import BPY_SERVER, bytes_to_object, object_to_bytes
    import requests

    payload_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f"skintokens_{endpoint}_", suffix=".pt", delete=False) as f:
            f.write(object_to_bytes(payload))
            payload_path = f.name
        response = requests.post(
            f"{BPY_SERVER}/{endpoint}",
            data=object_to_bytes({"payload_path": payload_path}),
        )
        response.raise_for_status()
        result = bytes_to_object(response.content)
        if isinstance(result, dict) and result.get("error") is not None:
            raise RuntimeError(result.get("traceback") or result["error"])
        return result
    finally:
        if payload_path is not None:
            try:
                os.remove(payload_path)
            except OSError:
                pass


def _ensure_bpy_server(timeout: int = 30) -> None:
    global _BPY_PROC, _BPY_LOG

    _ensure_repo_on_path()
    from src.server.spec import BPY_SERVER
    import requests

    try:
        requests.get(f"{BPY_SERVER}/ping", timeout=1)
        return
    except Exception:
        pass

    if _BPY_PROC is None or _BPY_PROC.poll() is not None:
        log_dir = COMPONENT_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if _BPY_LOG is not None and not _BPY_LOG.closed:
            _BPY_LOG.close()
        _BPY_LOG = open(log_dir / "skintokens_bpy_server.log", "a", encoding="utf-8")
        kwargs = {
            "args": [sys.executable, str(COMPONENT_DIR / "bpy_server.py")],
            "cwd": str(COMPONENT_DIR),
            "stdout": _BPY_LOG,
            "stderr": _BPY_LOG,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["preexec_fn"] = os.setsid
        _BPY_PROC = subprocess.Popen(**kwargs)
        atexit.register(_stop_bpy_server)

    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(f"{BPY_SERVER}/ping", timeout=1)
            return
        except Exception:
            time.sleep(0.5)
    detail = ""
    if _BPY_LOG is not None:
        try:
            _BPY_LOG.flush()
        except Exception:
            pass
        log_path = getattr(_BPY_LOG, "name", "")
        if log_path:
            detail = f"; see log: {log_path}"
    raise RuntimeError(f"SkinTokens bpy_server failed to start{detail}")


def _stop_bpy_server() -> None:
    if _BPY_PROC is None or _BPY_PROC.poll() is not None:
        return
    try:
        if os.name == "nt":
            _BPY_PROC.terminate()
        else:
            os.killpg(os.getpgid(_BPY_PROC.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    if _BPY_LOG is not None:
        try:
            _BPY_LOG.close()
        except Exception:
            pass


def _output_path(input_path: Path) -> Path:
    try:
        import folder_paths

        out_dir = Path(folder_paths.get_output_directory()) / "skintokens"
    except Exception:
        out_dir = COMPONENT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{input_path.stem}_rigged.glb"


class SkinTokensLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_ckpt": ("STRING", {"default": DEFAULT_CKPT}),
            },
            "optional": {
                "hf_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("SKINTOKENS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "OpenBlender/SkinTokens"

    def load(self, model_ckpt: str, hf_path: str = ""):
        ckpt = _ensure_model_files(model_ckpt)
        return ({"model_ckpt": str(ckpt), "hf_path": hf_path or None},)


class SkinTokensRigMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SKINTOKENS_MODEL",),
                "mesh_path": ("STRING", {"default": "examples/giraffe.glb"}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 200}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "repetition_penalty": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 10, "min": 1, "max": 20}),
                "use_skeleton": ("BOOLEAN", {"default": False}),
                "use_transfer": ("BOOLEAN", {"default": False}),
                "use_postprocess": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_file",)
    FUNCTION = "rig"
    CATEGORY = "OpenBlender/SkinTokens"
    OUTPUT_NODE = True

    def rig(
        self,
        model,
        mesh_path: str,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        num_beams: int,
        use_skeleton: bool,
        use_transfer: bool,
        use_postprocess: bool,
    ):
        _ensure_repo_on_path()
        _ensure_bpy_dll_path()
        tokenrig, tokenizer, transform = _load_model(model["model_ckpt"], model.get("hf_path"))

        from torch import Tensor
        from src.data.dataset import DatasetConfig, RigDatasetModule
        from src.rig_package.parser.bpy import BpyParser, transfer_rigging
        from src.data.vertex_group import voxel_skin

        input_path = _resolve_path(mesh_path)
        if not input_path.exists():
            raise FileNotFoundError(f"SkinTokens mesh not found: {mesh_path}")
        out_path = _output_path(input_path)

        datapath = {
            "data_name": None,
            # num_workers=0 here, so direct bpy loading is safe and avoids
            # flaky subprocess RPC crashes on Windows.
            "loader": "bpy",
            "filepaths": {"articulation": [str(input_path)]},
        }
        dataset_config = DatasetConfig.parse(
            shuffle=False,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            datapath=datapath,
        ).split_by_cls()
        module = RigDatasetModule(
            predict_dataset_config=dataset_config,
            predict_transform=transform,
            tokenizer=tokenizer,
            process_fn=tokenrig._process_fn,
        )
        dataloader = module.predict_dataloader()["articulation"]
        batch = next(iter(dataloader))
        batch = {k: v.to("cuda") if isinstance(v, Tensor) else v for k, v in batch.items()}

        if not use_skeleton:
            batch.pop("skeleton_tokens", None)
            batch.pop("skeleton_mask", None)

        batch["generate_kwargs"] = {
            "max_length": 2048,
            "top_k": int(top_k),
            "top_p": float(top_p),
            "temperature": float(temperature),
            "repetition_penalty": float(repetition_penalty),
            "num_return_sequences": 1,
            "num_beams": int(num_beams),
            "do_sample": True,
        }

        if "skeleton_tokens" in batch and "skeleton_mask" in batch:
            mask = batch["skeleton_mask"][0] == 1
            skeleton_tokens = batch["skeleton_tokens"][0][mask].cpu().numpy()
        else:
            skeleton_tokens = None

        preds = tokenrig.predict_step(
            batch,
            skeleton_tokens=[skeleton_tokens] if skeleton_tokens is not None else None,
            make_asset=True,
        )["results"]
        asset = preds[0].asset
        if asset is None:
            raise RuntimeError("SkinTokens did not produce a rigged asset")

        if use_postprocess:
            voxel = asset.voxel(resolution=196)
            asset.skin *= voxel_skin(
                grid=0,
                grid_coords=voxel.coords,
                joints=asset.joints,
                vertices=asset.vertices,
                faces=asset.faces,
                mode="square",
                voxel_size=voxel.voxel_size,
            )
            asset.normalize_skin()

        if use_transfer:
            transfer_rigging(
                source_asset=asset,
                target_path=asset.path,
                export_path=str(out_path),
                group_per_vertex=4,
            )
        else:
            BpyParser.export(
                asset=asset,
                filepath=str(out_path),
                group_per_vertex=4,
            )

        return (str(out_path),)
