"""
SMPLViewer Node - Packs SMPL motion capture data for visualization/import.
Writes a single .npz containing motion params plus generated SMPL mesh frames.
"""

import logging
from pathlib import Path
import torch
import numpy as np
import smplx
import folder_paths

from comfy_api.latest import io

logger = logging.getLogger("SMPLViewer")

from .shared_utils import next_sequential_filename as _next_sequential_filename


class SMPLViewer(io.ComfyNode):
    """
    ComfyUI node for visualizing SMPL motion capture sequences in an interactive 3D viewer.
    Writes mesh and motion data to a single .npz file.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SMPLViewer",
            display_name="SMPL 3D Viewer",
            category="OpenBlender/MotionCapture/GVHMR",
            is_output_node=True,
            inputs=[
                io.String.Input("npz_path", default="", multiline=False,
                                tooltip="Path to .npz file with SMPL parameters (from GVHMR Inference)"),
                io.Int.Input("frame_skip", default=1, min=1, max=10, step=1,
                             tooltip="Skip every N frames to reduce data size (1 = no skip)",
                             optional=True),
                io.String.Input("mesh_color", default="#4a9eff",
                                tooltip="Hex color for the mesh (e.g. #4a9eff for blue)",
                                optional=True),
            ],
            outputs=[
                io.String.Output(display_name="rig_npz_path"),
            ],
        )

    @classmethod
    def execute(cls, npz_path="", frame_skip=1, mesh_color="#4a9eff") -> io.NodeOutput:
        """
        Generate 3D mesh data from SMPL parameters, pack it with the source
        motion data into one .npz file, and return that file path.
        """
        logger.info("[SMPLViewer] Generating 3D mesh data for visualization...")

        if not npz_path or not npz_path.strip():
            raise ValueError("npz_path is required")

        logger.info(f"[SMPLViewer] Loading SMPL parameters from: {npz_path}")
        file_path = Path(npz_path)
        if not file_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {file_path}")

        # Load npz file
        data = np.load(str(file_path))
        params = {}
        for key in data.files:
            params[key] = torch.from_numpy(data[key])

        # Extract SMPL parameters
        body_pose = params['body_pose']  # (F, 63)
        betas = params['betas']  # (F, 10)
        global_orient = params['global_orient']  # (F, 3)
        transl = params.get('transl', None)  # (F, 3) or None

        num_frames = body_pose.shape[0]
        logger.info(f"[SMPLViewer] Processing {num_frames} frames (skip={frame_skip})")

        try:
            import comfy.model_management
            device = comfy.model_management.get_torch_device()
        except Exception:
            device = torch.device("cpu")

        data_dir = Path(__file__).parent / "body_model"
        models_dir = Path(folder_paths.models_dir) / "motion_capture" / "body_models"

        # Initialize SMPL-X model
        smplx_model = smplx.create(
            model_path=str(models_dir),
            model_type='smplx',
            gender='neutral',
            num_pca_comps=12,
            flat_hand_mean=False,
        ).to(device)
        smplx_model.eval()

        # Load SMPL-X to SMPL vertex conversion matrix
        from .body_model.utils import load_sparse_tensor
        smplx2smpl = load_sparse_tensor(data_dir / "smplx2smpl_sparse.npz").to(device)

        # Get SMPL faces
        faces = np.load(str(data_dir / "smpl_faces.npy"))

        # Generate vertices
        import comfy.model_management
        vertices_list = []
        with torch.no_grad():
            for frame_idx in range(0, num_frames, frame_skip):
                comfy.model_management.throw_exception_if_processing_interrupted()
                bp = body_pose[frame_idx:frame_idx+1].to(device)
                b = betas[frame_idx:frame_idx+1].to(device)
                go = global_orient[frame_idx:frame_idx+1].to(device)
                t = transl[frame_idx:frame_idx+1].to(device) if transl is not None else None

                smplx_out = smplx_model(
                    body_pose=bp, betas=b, global_orient=go, transl=t,
                )
                smpl_verts = torch.matmul(smplx2smpl, smplx_out.vertices[0])
                vertices_list.append(smpl_verts.cpu().numpy())

        vertices_array = np.stack(vertices_list, axis=0).astype(np.float32)  # (F', V, 3)
        faces_u32 = faces.astype(np.uint32)
        fps = 30 // frame_skip

        logger.info(f"[SMPLViewer] Generated mesh: {vertices_array.shape[0]} frames, "
                     f"{vertices_array.shape[1]} vertices, {faces_u32.shape[0]} faces")

        output_dir = Path(folder_paths.get_output_directory())
        rig_filename = _next_sequential_filename(output_dir, "smpl_rig", ".npz")
        rig_filepath = output_dir / rig_filename

        packed = {key: data[key] for key in data.files}
        packed.update({
            "vertices": vertices_array,
            "faces": faces_u32,
            "fps": np.array([float(fps)], dtype=np.float32),
            "mesh_color": np.array([mesh_color]),
            "openblender_packed_rig": np.array([1], dtype=np.uint8),
        })
        np.savez_compressed(str(rig_filepath), **packed)

        size_mb = rig_filepath.stat().st_size / (1024 * 1024)
        logger.info(f"[SMPLViewer] Wrote packed rig {rig_filename} ({size_mb:.1f} MB)")

        return io.NodeOutput(str(rig_filepath), ui={
            "smpl_rig_file": [rig_filename]
        })


# Node registration
NODE_CLASS_MAPPINGS = {
    "SMPLViewer": SMPLViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLViewer": "SMPL 3D Viewer",
}
