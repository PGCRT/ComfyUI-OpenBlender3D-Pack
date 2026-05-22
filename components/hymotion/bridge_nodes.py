"""
Bridge nodes connecting HY-Motion HYMOTION_DATA with MotionCapture SMPL_PARAMS.
Self-contained - no dependency on nodes.py to avoid model_management import errors.
"""
import os
import sys
import uuid
import time
from typing import Dict, Any, List

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


class HYMotionData:
    def __init__(self, output_dict: Dict[str, Any], text: str, duration: float, seeds: List[int], device_info: str = "cpu"):
        self.output_dict = output_dict
        self.text = text
        self.duration = duration
        self.seeds = seeds
        self.device_info = device_info
        self.batch_size = output_dict["keypoints3d"].shape[0] if "keypoints3d" in output_dict else 1


def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def get_comfy_output_dir():
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except ImportError:
        return os.path.join(CURRENT_DIR, "..", "..", "output")

COMFY_OUTPUT_DIR = get_comfy_output_dir()

try:
    import folder_paths
except ImportError:
    folder_paths = None


class HYMotionNPZToSMPLParams:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "npz_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Path to NPZ file."}),
            },
        }

    RETURN_TYPES = ("SMPL_PARAMS",)
    RETURN_NAMES = ("smpl_params",)
    FUNCTION = "convert"
    CATEGORY = "OpenBlender/HY-Motion/utils"

    def convert(self, npz_path):
        resolved = npz_path
        if not os.path.isabs(resolved) and folder_paths:
            for d in [folder_paths.get_input_directory(), folder_paths.get_output_directory()]:
                p = os.path.join(d, resolved)
                if os.path.isfile(p):
                    resolved = p
                    break
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"NPZ not found: {npz_path}")

        data = np.load(resolved, allow_pickle=True)

        def to_tensor(arr):
            if isinstance(arr, np.ndarray):
                return torch.from_numpy(arr).float()
            if isinstance(arr, torch.Tensor):
                return arr.float()
            return arr

        global_params = {}
        for k in ["body_pose", "betas", "global_orient", "transl"]:
            if k in data:
                global_params[k] = to_tensor(data[k])
        if "poses" in data and "poses" not in global_params:
            global_params["poses"] = to_tensor(data["poses"])
        if "rot6d" in data and "poses" not in global_params:
            global_params["rot6d"] = to_tensor(data["rot6d"])

        if not global_params:
            for k in data.files:
                arr = data[k]
                if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                    global_params[k] = to_tensor(arr)

        first_arr = list(global_params.values())[0] if global_params else torch.zeros(1)
        nframes = first_arr.shape[0] if first_arr.ndim >= 1 else 1
        global_params.setdefault("body_pose", torch.zeros(nframes, 21 * 3))
        global_params.setdefault("global_orient", torch.zeros(nframes, 3))
        global_params.setdefault("transl", torch.zeros(nframes, 3))
        global_params.setdefault("betas", torch.zeros(1, 10))

        smpl_params = {
            "global": global_params,
            "incam": dict(global_params),
            "K_fullimg": to_tensor(data.get("K_fullimg", np.eye(3))),
            "R_w2c": to_tensor(data.get("R_w2c", np.eye(3))),
            "t_w2c": to_tensor(data.get("t_w2c", np.zeros(3))),
        }
        print(f"[HYMotionNPZToSMPLParams] Loaded {resolved} ({len(global_params)} keys)")
        return (smpl_params,)


class HYMotionSMPLToData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "smpl_params": ("SMPL_PARAMS", {"tooltip": "SMPL parameters from GVHMR or MotionCapture."}),
            },
            "optional": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("HYMOTION_DATA",)
    RETURN_NAMES = ("motion_data",)
    FUNCTION = "convert"
    CATEGORY = "OpenBlender/HY-Motion/utils"

    def convert(self, smpl_params, text="", duration=0.0):
        from .hymotion.utils.geometry import axis_angle_to_matrix, rotation_matrix_to_rot6d
        from .hymotion.pipeline.body_model import WoodenMesh

        try:
            duration = float(duration) if duration else 0.0
        except:
            duration = 0.0

        if isinstance(smpl_params, list):
            params = smpl_params[0]
        else:
            params = smpl_params
        if params is None:
            raise ValueError("smpl_params is None")

        inner = params.get("global", params.get("incam", params))
        body_pose = inner.get("body_pose")
        global_orient = inner.get("global_orient")
        trans = inner.get("trans", inner.get("transl"))
        poses = inner.get("poses")

        def to_tensor(x):
            if x is None: return None
            if isinstance(x, np.ndarray): return torch.from_numpy(x).float()
            return x.float() if isinstance(x, torch.Tensor) else x

        body_pose = to_tensor(body_pose)
        global_orient = to_tensor(global_orient)
        trans = to_tensor(trans)
        poses = to_tensor(poses)

        if poses is None and body_pose is not None and global_orient is not None:
            poses = torch.cat([global_orient, body_pose], dim=1)
        if poses is None:
            raise ValueError("No pose data found")

        if len(poses.shape) == 2:
            poses = poses.reshape(poses.shape[0], -1, 3)
        if len(poses.shape) == 3:
            poses = poses.unsqueeze(0)

        bs, nf, nj, _ = poses.shape
        device = poses.device

        if nj < 52:
            poses = torch.cat([poses, torch.zeros(bs, nf, 52 - nj, 3, device=device)], dim=2)
        elif nj > 52:
            poses = poses[:, :, :52, :]

        if trans is None:
            trans = torch.zeros(bs, nf, 3, device=device)
        else:
            if len(trans.shape) == 2:
                trans = trans.unsqueeze(0).expand(bs, -1, -1)
            if trans.shape[1] != nf:
                trans = trans[:, :nf, :] if trans.shape[1] > nf else torch.cat([trans, torch.zeros(bs, nf - trans.shape[1], 3, device=device)], dim=1)

        poses = torch.nan_to_num(poses)
        trans = torch.nan_to_num(trans)

        rot_mat = axis_angle_to_matrix(poses)
        rot6d = rotation_matrix_to_rot6d(rot_mat)

        if duration <= 0:
            duration = nf / float(params.get("mocap_framerate", params.get("fps", 30.0)))

        txt = text.strip() if text.strip() else "motion from smpl"

        try:
            wm = WoodenMesh().cpu().eval()
            with torch.no_grad():
                out = wm.forward_batch({"rot6d": rot6d.cpu(), "trans": trans.cpu()}, chunk_size=64)
                kp3d = out["keypoints3d"]
        except Exception:
            kp3d = torch.zeros_like(trans.cpu().unsqueeze(2).expand(-1, -1, 22, -1))

        md = HYMotionData(
            output_dict={"rot6d": rot6d.cpu(), "transl": trans.cpu(), "root_rotations_mat": rot_mat.cpu()[:, :, 0], "keypoints3d": kp3d},
            text=txt, duration=duration, seeds=[0] * bs, device_info="cpu",
        )
        print(f"[HYMotionSMPLToData] {nf} frames -> HY-Motion (batch={bs})")
        return (md,)


class HYMotionRetargetFBX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA", {}),
                "target_fbx": ("STRING", {"default": "", "multiline": False}),
                "output_dir": ("STRING", {"default": "hymotion_retarget"}),
                "filename_prefix": ("STRING", {"default": "retarget"}),
            },
            "optional": {
                "mapping_file": ("STRING", {"default": "", "multiline": False}),
                "yaw_offset": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "neutral_fingers": ("BOOLEAN", {"default": True}),
                "unique_names": ("BOOLEAN", {"default": True}),
                "in_place": ("BOOLEAN", {"default": False}),
                "in_place_x": ("BOOLEAN", {"default": False}),
                "in_place_y": ("BOOLEAN", {"default": False}),
                "in_place_z": ("BOOLEAN", {"default": False}),
                "preserve_position": ("BOOLEAN", {"default": False}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "target_pose_type": (["T-Pose", "A-Pose"], {"default": "T-Pose"}),
            }
        }

    RETURN_TYPES = ("STRING", "FBX")
    RETURN_NAMES = ("fbx_path", "fbx")
    FUNCTION = "retarget"
    CATEGORY = "OpenBlender/HY-Motion/view"
    OUTPUT_NODE = True

    def retarget(self, motion_data, target_fbx, output_dir="hymotion_retarget", filename_prefix="retarget",
                 mapping_file="", yaw_offset=0.0, neutral_fingers=True, unique_names=True,
                 in_place=False, in_place_x=False, in_place_y=False, in_place_z=False, preserve_position=False,
                 fps=30.0, target_pose_type="T-Pose"):
        from .hymotion.utils.retarget_fbx import (
            load_npz, load_fbx, load_bone_mapping, retarget_animation,
            apply_retargeted_animation, save_fbx,
        )
        from .hymotion.pipeline.body_model import construct_smpl_data_dict
        import fbx as _fbx

        resolved = target_fbx
        if not os.path.isabs(resolved) and folder_paths:
            for d in [folder_paths.get_input_directory(), folder_paths.get_output_directory()]:
                p = os.path.join(d, resolved)
                if os.path.isfile(p):
                    resolved = p
                    break
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Target FBX not found: {target_fbx}")

        full_out = os.path.join(COMFY_OUTPUT_DIR, output_dir)
        os.makedirs(full_out, exist_ok=True)

        ts = get_timestamp()
        uid = str(uuid.uuid4())[:8]
        out_files = []

        for bi in range(motion_data.batch_size):
            if unique_names:
                out_fbx = os.path.join(full_out, f"{filename_prefix}_{ts}_{uid}_{bi:03d}.fbx")
            else:
                out_fbx = os.path.join(full_out, f"{filename_prefix}_{bi:03d}.fbx")

            dd = {}
            od = motion_data.output_dict
            for k in ['keypoints3d', 'rot6d', 'transl', 'root_rotations_mat']:
                if k in od and od[k] is not None:
                    t = od[k][bi]
                    dd[k] = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)

            if "rot6d" in dd and "transl" in dd:
                sd = construct_smpl_data_dict(torch.from_numpy(dd["rot6d"]), torch.from_numpy(dd["transl"]))
                for k, v in sd.items():
                    if k not in dd:
                        dd[k] = v

            if "poses" not in dd:
                raise ValueError("Motion data missing poses")

            src_skel = load_npz(dd)
            tgt_man, tgt_scene, tgt_skel = load_fbx(resolved, use_bind_pose=False)
            mapping = load_bone_mapping(mapping_file, src_skel, tgt_skel)
            rots, locs, active = retarget_animation(
                src_skel, tgt_skel, mapping, force_scale=0.0, yaw_offset=yaw_offset,
                neutral_fingers=neutral_fingers, in_place=in_place, in_place_x=in_place_x,
                in_place_y=in_place_y, in_place_z=in_place_z, preserve_position=preserve_position,
                auto_stride=True, smart_arm_align=(target_pose_type == "A-Pose"),
            )
            src_tm = _fbx.FbxTime().ConvertFrameRateToTimeMode(fps)
            apply_retargeted_animation(tgt_scene, tgt_skel, rots, locs, src_skel.frame_start, src_skel.frame_end, src_tm)
            save_fbx(tgt_man, tgt_scene, out_fbx)
            out_files.append(out_fbx)

        rel = [os.path.relpath(p, COMFY_OUTPUT_DIR).replace("\\", "/") for p in out_files]
        result = "\n".join(rel)
        fbx_i = [{"filename": os.path.basename(p), "subfolder": output_dir, "type": "output"} for p in out_files]
        print(f"[HYMotionRetargetFBX] Done: {len(out_files)} file(s)")
        return {"ui": {"fbx": fbx_i, "fbx_url": [result]}, "result": (result, fbx_i)}