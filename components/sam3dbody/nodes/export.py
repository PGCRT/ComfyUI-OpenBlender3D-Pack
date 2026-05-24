# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Export nodes for SAM 3D Body meshes.

Exports meshes with rigging data to various formats using bpy in isolated venv.
"""

import logging
import os
import json
import time
import tempfile
import sys
import subprocess
import numpy as np
import torch
import folder_paths
import glob
from comfy_api.latest import io

log = logging.getLogger("sam3dbody")


def _prepare_bpy_import_environment():
    """Harden environment before importing bpy in isolated worker."""
    os.environ["PYTHONNOUSERSITE"] = "1"

    usersite_markers = [
        "\\AppData\\Roaming\\Python\\",
        "/AppData/Roaming/Python/",
    ]
    sys.path[:] = [
        p for p in sys.path
        if p and not any(marker in p for marker in usersite_markers)
    ]

    prefix = sys.prefix
    dll_candidates = [
        os.path.join(prefix, "Library", "bin"),
        os.path.join(prefix, "DLLs"),
        os.path.join(prefix, "Lib", "site-packages", "bpy"),
    ]

    bpy_pkg_root = os.path.join(prefix, "Lib", "site-packages", "bpy")
    if os.path.isdir(bpy_pkg_root):
        for child in os.listdir(bpy_pkg_root):
            child_path = os.path.join(bpy_pkg_root, child)
            if os.path.isdir(child_path):
                dll_candidates.append(child_path)

    existing = [p for p in dll_candidates if os.path.isdir(p)]

    for dll_dir in existing:
        try:
            os.add_dll_directory(dll_dir)
        except (AttributeError, FileNotFoundError, OSError):
            pass

    current_path = os.environ.get("PATH", "")
    prepend = os.pathsep.join(existing)
    if prepend:
        os.environ["PATH"] = prepend + (os.pathsep + current_path if current_path else "")


def _safe_bone_name(base_name, index, prefix=""):
    name = str(base_name) if base_name else f"Joint_{index:03d}"
    name = name.replace(" ", "_")
    return f"{prefix}{name}"


class BpyFBXExporter:
    """Isolated bpy-based FBX exporter that runs in the sam3dbody venv."""

    FUNCTION = "export"

    def _export_in_subprocess(self, input_obj_path, output_fbx_path, skeleton_json_path=None, combined_json_path=None):
        result_path = tempfile.NamedTemporaryFile(suffix="_sam3d_export_result.json", delete=False).name
        payload = {
            "input_obj_path": input_obj_path,
            "output_fbx_path": output_fbx_path,
            "skeleton_json_path": skeleton_json_path,
            "combined_json_path": combined_json_path,
            "result_path": result_path,
        }

        with tempfile.NamedTemporaryFile(suffix="_sam3d_export_payload.json", delete=False, mode="w", encoding="utf-8") as fp:
            payload_path = fp.name
            json.dump(payload, fp)

        try:
            cmd = [sys.executable, __file__, "--sam3dbody-export", payload_path]
            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"
            env["SAM3DBODY_BPY_SUBPROC"] = "1"
            env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p])
            # Harden against Intel Fortran runtime crash on Windows:
            # forrtl: error (200) is triggered by console window-close signals.
            # Using DETACHED_PROCESS + pythonw.exe (if available) prevents this.
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                # Prefer pythonw.exe to avoid any console window at all
                pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
                if os.path.isfile(pythonw):
                    cmd[0] = pythonw
                # Also suppress MKL/Intel threading signals that can cause aborts
                env["MKL_NUM_THREADS"] = "1"
                env["OPENBLAS_NUM_THREADS"] = "1"
                env["OMP_NUM_THREADS"] = "1"
                env["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

            completed = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                creationflags=creationflags,
            )

            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                stdout = (completed.stdout or "").strip()
                msg = stderr if stderr else stdout
                raise RuntimeError(f"SAM3DBody bpy subprocess export failed (code {completed.returncode}): {msg}")

            if not os.path.exists(result_path):
                stderr = (completed.stderr or "").strip()
                stdout = (completed.stdout or "").strip()
                msg = stderr if stderr else stdout
                raise RuntimeError(f"SAM3DBody bpy subprocess returned no result file: {msg}")

            with open(result_path, "r", encoding="utf-8") as rf:
                return json.load(rf)
        finally:
            if os.path.exists(payload_path):
                try:
                    os.unlink(payload_path)
                except Exception:
                    pass
            if os.path.exists(result_path):
                try:
                    os.unlink(result_path)
                except Exception:
                    pass

    def export(self, input_obj_path, output_fbx_path, skeleton_json_path=None, combined_json_path=None):
        """Export OBJ mesh to FBX using bpy.

        If combined_json_path is provided, exports multiple people into a single FBX.
        Otherwise, exports a single person.
        """
        try:
            _prepare_bpy_import_environment()
            import bpy
        except ImportError:
            if os.environ.get("SAM3DBODY_BPY_SUBPROC") == "1":
                raise
            return self._export_in_subprocess(
                input_obj_path=input_obj_path,
                output_fbx_path=output_fbx_path,
                skeleton_json_path=skeleton_json_path,
                combined_json_path=combined_json_path,
            )
        from mathutils import Vector
        import numpy as np
        import json

        # Handle combined export mode
        if combined_json_path and os.path.exists(combined_json_path):
            return self._export_combined(combined_json_path, output_fbx_path)

        # Single person export mode - load skeleton data from JSON if provided
        joints = None
        num_joints = 0
        joint_parents_list = None
        skinning_weights = None
        global_rotations = None
        joint_names = None

        if skeleton_json_path and os.path.exists(skeleton_json_path):
            with open(skeleton_json_path, 'r') as f:
                skeleton_data = json.load(f)

            joint_positions = skeleton_data.get('joint_positions', [])
            num_joints = skeleton_data.get('num_joints', len(joint_positions))
            joint_parents_list = skeleton_data.get('joint_parents')
            skinning_weights = skeleton_data.get('skinning_weights')
            global_rotations_data = skeleton_data.get('global_rotations')
            joint_names = skeleton_data.get('joint_names')

            if joint_positions:
                joints = np.array(joint_positions, dtype=np.float32)

            if global_rotations_data:
                global_rotations = np.array(global_rotations_data, dtype=np.float32)
                log.info(f" Loaded global_rotations: shape {global_rotations.shape}")

        # Clean default scene
        for c in bpy.data.actions:
            bpy.data.actions.remove(c)
        for c in bpy.data.armatures:
            bpy.data.armatures.remove(c)
        for c in bpy.data.cameras:
            bpy.data.cameras.remove(c)
        for c in bpy.data.collections:
            bpy.data.collections.remove(c)
        for c in bpy.data.images:
            bpy.data.images.remove(c)
        for c in bpy.data.materials:
            bpy.data.materials.remove(c)
        for c in bpy.data.meshes:
            bpy.data.meshes.remove(c)
        for c in bpy.data.objects:
            bpy.data.objects.remove(c)
        for c in bpy.data.textures:
            bpy.data.textures.remove(c)

        # Create collection
        collection = bpy.data.collections.new('SAM3D_Export')
        bpy.context.scene.collection.children.link(collection)

        # Import OBJ mesh
        bpy.ops.wm.obj_import(filepath=input_obj_path)

        imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        if not imported_objects:
            raise RuntimeError("No mesh found after OBJ import")

        mesh_obj = imported_objects[0]
        mesh_obj.name = 'SAM3D_Character'

        # Move to our collection
        if mesh_obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(mesh_obj)
        collection.objects.link(mesh_obj)

        # Create armature from skeleton if provided
        if joints is not None and num_joints > 0:
            # Create armature in edit mode
            bpy.ops.object.armature_add(enter_editmode=True)
            armature = bpy.data.armatures.get('Armature')
            armature.name = 'SAM3D_Skeleton'
            armature_obj = bpy.context.active_object
            armature_obj.name = 'SAM3D_Skeleton'

            # Move to our collection
            if armature_obj.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(armature_obj)
            collection.objects.link(armature_obj)

            edit_bones = armature.edit_bones
            extrude_size = 0.05

            # Remove default bone
            default_bone = edit_bones.get('Bone')
            if default_bone:
                edit_bones.remove(default_bone)

            # Calculate skeleton center for root bone placement
            skeleton_center = joints.mean(axis=0)

            # Make positions relative to skeleton center
            rel_joints = joints - skeleton_center

            # Apply coordinate system correction to match mesh orientation
            rel_joints_corrected = np.zeros_like(rel_joints)
            rel_joints_corrected[:, 0] = rel_joints[:, 0]
            rel_joints_corrected[:, 1] = -rel_joints[:, 2]
            rel_joints_corrected[:, 2] = rel_joints[:, 1]

            # Build child map for orientation
            children_map = {i: [] for i in range(num_joints)}
            if joint_parents_list and len(joint_parents_list) == num_joints:
                for child_idx, parent_idx in enumerate(joint_parents_list):
                    if 0 <= parent_idx < num_joints and parent_idx != child_idx:
                        children_map[parent_idx].append(child_idx)

            # Create all bones
            bones_dict = {}
            for i in range(num_joints):
                source_name = joint_names[i] if joint_names and i < len(joint_names) else None
                bone_name = _safe_bone_name(source_name, i)
                bone = edit_bones.new(bone_name)
                head = Vector((rel_joints_corrected[i, 0], rel_joints_corrected[i, 1], rel_joints_corrected[i, 2]))

                tail_dir = None
                child_indices = children_map.get(i, [])
                if child_indices:
                    accum = Vector((0.0, 0.0, 0.0))
                    valid = 0
                    for cidx in child_indices:
                        cpos = Vector((
                            rel_joints_corrected[cidx, 0],
                            rel_joints_corrected[cidx, 1],
                            rel_joints_corrected[cidx, 2],
                        ))
                        d = cpos - head
                        if d.length > 1e-6:
                            accum += d.normalized()
                            valid += 1
                    if valid > 0 and accum.length > 1e-6:
                        tail_dir = accum.normalized()

                if tail_dir is None and joint_parents_list and len(joint_parents_list) == num_joints:
                    parent_idx = joint_parents_list[i]
                    if 0 <= parent_idx < num_joints and parent_idx != i:
                        ppos = Vector((
                            rel_joints_corrected[parent_idx, 0],
                            rel_joints_corrected[parent_idx, 1],
                            rel_joints_corrected[parent_idx, 2],
                        ))
                        d = head - ppos
                        if d.length > 1e-6:
                            tail_dir = d.normalized()

                if tail_dir is None:
                    tail_dir = Vector((0.0, 0.0, 1.0))

                bone.head = head
                bone.tail = head + tail_dir * extrude_size
                bones_dict[bone_name] = bone

            # Build hierarchical structure using joint parents if available
            if joint_parents_list and len(joint_parents_list) == num_joints:
                for i in range(num_joints):
                    parent_idx = joint_parents_list[i]
                    if parent_idx >= 0 and parent_idx < num_joints and parent_idx != i:
                        source_name = joint_names[i] if joint_names and i < len(joint_names) else None
                        source_parent = joint_names[parent_idx] if joint_names and parent_idx < len(joint_names) else None
                        bone_name = _safe_bone_name(source_name, i)
                        parent_bone_name = _safe_bone_name(source_parent, parent_idx)
                        bones_dict[bone_name].parent = bones_dict[parent_bone_name]
                        bones_dict[bone_name].use_connect = False
            else:
                # Fallback: create flat hierarchy with Joint_000 as root
                root_source = joint_names[0] if joint_names and len(joint_names) > 0 else None
                root_bone_name = _safe_bone_name(root_source, 0)
                for i in range(1, num_joints):
                    source_name = joint_names[i] if joint_names and i < len(joint_names) else None
                    bone_name = _safe_bone_name(source_name, i)
                    bones_dict[bone_name].parent = bones_dict[root_bone_name]
                    bones_dict[bone_name].use_connect = False

            # Switch to object mode
            bpy.ops.object.mode_set(mode='OBJECT')

            # Position armature at skeleton center
            skeleton_center_corrected = np.zeros(3)
            skeleton_center_corrected[0] = skeleton_center[0]
            skeleton_center_corrected[1] = -skeleton_center[2]
            skeleton_center_corrected[2] = skeleton_center[1]
            armature_obj.location = Vector((skeleton_center_corrected[0], skeleton_center_corrected[1], skeleton_center_corrected[2]))

            # Apply skinning weights if available
            if skinning_weights:
                # Create vertex groups for each bone
                for i in range(num_joints):
                    source_name = joint_names[i] if joint_names and i < len(joint_names) else None
                    bone_name = _safe_bone_name(source_name, i)
                    mesh_obj.vertex_groups.new(name=bone_name)

                # Assign weights to vertices
                num_vertices = len(mesh_obj.data.vertices)
                for vert_idx in range(min(num_vertices, len(skinning_weights))):
                    influences = skinning_weights[vert_idx]
                    if influences and len(influences) > 0:
                        for bone_idx, weight in influences:
                            if 0 <= bone_idx < num_joints and weight > 0.0001:
                                source_name = joint_names[bone_idx] if joint_names and bone_idx < len(joint_names) else None
                                bone_name = _safe_bone_name(source_name, bone_idx)
                                vertex_group = mesh_obj.vertex_groups.get(bone_name)
                                if vertex_group:
                                    vertex_group.add([vert_idx], weight, 'REPLACE')

            # Deselect all
            for obj in bpy.context.selected_objects:
                obj.select_set(False)

            # Parent mesh to armature
            mesh_obj.select_set(True)
            armature_obj.select_set(True)
            bpy.context.view_layer.objects.active = armature_obj

            if skinning_weights:
                bpy.ops.object.parent_set(type='ARMATURE')
            else:
                bpy.ops.object.parent_set(type='ARMATURE_NAME')

        # Make mesh double-sided AFTER skinning (so duplicated vertices inherit weights)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.duplicate()
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.mode_set(mode='OBJECT')

        # Export to FBX
        os.makedirs(os.path.dirname(output_fbx_path) if os.path.dirname(output_fbx_path) else '.', exist_ok=True)

        # Select all objects in our collection
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in collection.objects:
            obj.select_set(True)

        # Export file (FBX or GLB depending on extension)
        lower_path = output_fbx_path.lower()
        if lower_path.endswith('.glb'):
            bpy.ops.export_scene.gltf(
                filepath=output_fbx_path,
                check_existing=False,
                export_format='GLB',
                use_selection=True,
                export_yup=True,
                export_apply=False,
                export_animations=True,
                export_skins=True,
            )
        else:
            bpy.ops.export_scene.fbx(
                filepath=output_fbx_path,
                check_existing=False,
                use_selection=True,
                add_leaf_bones=False,
                path_mode='COPY',
                embed_textures=True,
            )

        return {"success": True, "output_path": output_fbx_path}

    def _export_combined(self, combined_json_path, output_fbx_path):
        """
        Export multiple people into a single combined FBX file.

        Args:
            combined_json_path: Path to JSON file containing list of people data
            output_fbx_path: Output path for the combined FBX file
        """
        _prepare_bpy_import_environment()
        import bpy
        from mathutils import Vector
        import numpy as np
        import json

        # Load the combined data from JSON
        with open(combined_json_path, 'r') as f:
            people_data = json.load(f)

        # Clean default scene
        for c in bpy.data.actions:
            bpy.data.actions.remove(c)
        for c in bpy.data.armatures:
            bpy.data.armatures.remove(c)
        for c in bpy.data.cameras:
            bpy.data.cameras.remove(c)
        for c in bpy.data.collections:
            bpy.data.collections.remove(c)
        for c in bpy.data.images:
            bpy.data.images.remove(c)
        for c in bpy.data.materials:
            bpy.data.materials.remove(c)
        for c in bpy.data.meshes:
            bpy.data.meshes.remove(c)
        for c in bpy.data.objects:
            bpy.data.objects.remove(c)
        for c in bpy.data.textures:
            bpy.data.textures.remove(c)

        # Create collection for all exported objects
        collection = bpy.data.collections.new('SAM3D_Export')
        bpy.context.scene.collection.children.link(collection)

        # Process each person
        for person in people_data:
            obj_path = person["obj_path"]
            skeleton_json_path = person.get("skeleton_json_path")
            idx = person["index"]

            # Load skeleton data from JSON if provided
            joints = None
            num_joints = 0
            joint_parents_list = None
            skinning_weights = None
            global_rotations = None
            joint_names = None

            if skeleton_json_path and os.path.exists(skeleton_json_path):
                with open(skeleton_json_path, 'r') as f:
                    skeleton_data = json.load(f)

                joint_positions = skeleton_data.get('joint_positions', [])
                num_joints = skeleton_data.get('num_joints', len(joint_positions))
                joint_parents_list = skeleton_data.get('joint_parents')
                skinning_weights = skeleton_data.get('skinning_weights')
                global_rotations_data = skeleton_data.get('global_rotations')
                joint_names = skeleton_data.get('joint_names')

                if joint_positions:
                    joints = np.array(joint_positions, dtype=np.float32)

                if global_rotations_data:
                    global_rotations = np.array(global_rotations_data, dtype=np.float32)
                    log.info(f" Person {idx}: Loaded global_rotations shape {global_rotations.shape}")

            # Import OBJ mesh
            bpy.ops.wm.obj_import(filepath=obj_path)

            imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not imported_objects:
                continue

            mesh_obj = imported_objects[0]
            mesh_obj.name = f'SAM3D_Character_{idx}'

            # Move to our collection
            if mesh_obj.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(mesh_obj)
            collection.objects.link(mesh_obj)

            # Create armature from skeleton if provided
            if joints is not None and num_joints > 0:
                # Create armature in edit mode
                bpy.ops.object.armature_add(enter_editmode=True)
                armature = bpy.context.active_object.data
                armature.name = f'SAM3D_Skeleton_{idx}'
                armature_obj = bpy.context.active_object
                armature_obj.name = f'SAM3D_Skeleton_{idx}'

                # Move to our collection
                if armature_obj.name in bpy.context.scene.collection.objects:
                    bpy.context.scene.collection.objects.unlink(armature_obj)
                collection.objects.link(armature_obj)

                edit_bones = armature.edit_bones
                extrude_size = 0.05

                # Remove default bone
                default_bone = edit_bones.get('Bone')
                if default_bone:
                    edit_bones.remove(default_bone)

                # Calculate skeleton center for root bone placement
                skeleton_center = joints.mean(axis=0)

                # Make positions relative to skeleton center
                rel_joints = joints - skeleton_center

                # Apply coordinate system correction to match mesh orientation
                rel_joints_corrected = np.zeros_like(rel_joints)
                rel_joints_corrected[:, 0] = rel_joints[:, 0]
                rel_joints_corrected[:, 1] = -rel_joints[:, 2]
                rel_joints_corrected[:, 2] = rel_joints[:, 1]

                # Build child map for orientation
                children_map = {j: [] for j in range(num_joints)}
                if joint_parents_list and len(joint_parents_list) == num_joints:
                    for child_idx, parent_idx in enumerate(joint_parents_list):
                        if 0 <= parent_idx < num_joints and parent_idx != child_idx:
                            children_map[parent_idx].append(child_idx)

                # Create all bones with unique names for this person
                bones_dict = {}
                for i in range(num_joints):
                    source_name = joint_names[i] if joint_names and i < len(joint_names) else None
                    bone_name = _safe_bone_name(source_name, i, prefix=f'P{idx}_')
                    bone = edit_bones.new(bone_name)
                    head = Vector((rel_joints_corrected[i, 0], rel_joints_corrected[i, 1], rel_joints_corrected[i, 2]))

                    tail_dir = None
                    child_indices = children_map.get(i, [])
                    if child_indices:
                        accum = Vector((0.0, 0.0, 0.0))
                        valid = 0
                        for cidx in child_indices:
                            cpos = Vector((
                                rel_joints_corrected[cidx, 0],
                                rel_joints_corrected[cidx, 1],
                                rel_joints_corrected[cidx, 2],
                            ))
                            d = cpos - head
                            if d.length > 1e-6:
                                accum += d.normalized()
                                valid += 1
                        if valid > 0 and accum.length > 1e-6:
                            tail_dir = accum.normalized()

                    if tail_dir is None and joint_parents_list and len(joint_parents_list) == num_joints:
                        parent_idx = joint_parents_list[i]
                        if 0 <= parent_idx < num_joints and parent_idx != i:
                            ppos = Vector((
                                rel_joints_corrected[parent_idx, 0],
                                rel_joints_corrected[parent_idx, 1],
                                rel_joints_corrected[parent_idx, 2],
                            ))
                            d = head - ppos
                            if d.length > 1e-6:
                                tail_dir = d.normalized()

                    if tail_dir is None:
                        tail_dir = Vector((0.0, 0.0, 1.0))

                    bone.head = head
                    bone.tail = head + tail_dir * extrude_size
                    bones_dict[bone_name] = bone

                # Build hierarchical structure using joint parents if available
                if joint_parents_list and len(joint_parents_list) == num_joints:
                    for i in range(num_joints):
                        parent_idx = joint_parents_list[i]
                        if parent_idx >= 0 and parent_idx < num_joints and parent_idx != i:
                            source_name = joint_names[i] if joint_names and i < len(joint_names) else None
                            source_parent = joint_names[parent_idx] if joint_names and parent_idx < len(joint_names) else None
                            bone_name = _safe_bone_name(source_name, i, prefix=f'P{idx}_')
                            parent_bone_name = _safe_bone_name(source_parent, parent_idx, prefix=f'P{idx}_')
                            bones_dict[bone_name].parent = bones_dict[parent_bone_name]
                            bones_dict[bone_name].use_connect = False
                else:
                    # Fallback: create flat hierarchy with Joint_000 as root
                    root_source = joint_names[0] if joint_names and len(joint_names) > 0 else None
                    root_bone_name = _safe_bone_name(root_source, 0, prefix=f'P{idx}_')
                    for i in range(1, num_joints):
                        source_name = joint_names[i] if joint_names and i < len(joint_names) else None
                        bone_name = _safe_bone_name(source_name, i, prefix=f'P{idx}_')
                        bones_dict[bone_name].parent = bones_dict[root_bone_name]
                        bones_dict[bone_name].use_connect = False

                # Switch to object mode
                bpy.ops.object.mode_set(mode='OBJECT')

                # Position armature at skeleton center
                skeleton_center_corrected = np.zeros(3)
                skeleton_center_corrected[0] = skeleton_center[0]
                skeleton_center_corrected[1] = -skeleton_center[2]
                skeleton_center_corrected[2] = skeleton_center[1]
                armature_obj.location = Vector((skeleton_center_corrected[0], skeleton_center_corrected[1], skeleton_center_corrected[2]))

                # Apply skinning weights if available
                if skinning_weights:
                    # Create vertex groups for each bone
                    for i in range(num_joints):
                        source_name = joint_names[i] if joint_names and i < len(joint_names) else None
                        bone_name = _safe_bone_name(source_name, i, prefix=f'P{idx}_')
                        mesh_obj.vertex_groups.new(name=bone_name)

                    # Assign weights to vertices
                    num_vertices = len(mesh_obj.data.vertices)
                    for vert_idx in range(min(num_vertices, len(skinning_weights))):
                        influences = skinning_weights[vert_idx]
                        if influences and len(influences) > 0:
                            for bone_idx, weight in influences:
                                if 0 <= bone_idx < num_joints and weight > 0.0001:
                                    source_name = joint_names[bone_idx] if joint_names and bone_idx < len(joint_names) else None
                                    bone_name = _safe_bone_name(source_name, bone_idx, prefix=f'P{idx}_')
                                    vertex_group = mesh_obj.vertex_groups.get(bone_name)
                                    if vertex_group:
                                        vertex_group.add([vert_idx], weight, 'REPLACE')

                # Deselect all
                for obj in bpy.context.selected_objects:
                    obj.select_set(False)

                # Parent mesh to armature
                mesh_obj.select_set(True)
                armature_obj.select_set(True)
                bpy.context.view_layer.objects.active = armature_obj

                if skinning_weights:
                    bpy.ops.object.parent_set(type='ARMATURE')
                else:
                    bpy.ops.object.parent_set(type='ARMATURE_NAME')

            # Make mesh double-sided AFTER skinning (so duplicated vertices inherit weights)
            bpy.context.view_layer.objects.active = mesh_obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.duplicate()
            bpy.ops.mesh.flip_normals()
            bpy.ops.object.mode_set(mode='OBJECT')

        # Export to FBX
        os.makedirs(os.path.dirname(output_fbx_path) if os.path.dirname(output_fbx_path) else '.', exist_ok=True)

        # Select all objects in our collection
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in collection.objects:
            obj.select_set(True)

        # Export FBX with all objects
        bpy.ops.export_scene.fbx(
            filepath=output_fbx_path,
            check_existing=False,
            use_selection=True,
            add_leaf_bones=False,
            path_mode='COPY',
            embed_textures=True,
        )

        return {"success": True, "output_path": output_fbx_path}

class BpyPoseApplier:
    """Isolated bpy-based pose applier that runs in the sam3dbody venv."""

    FUNCTION = "apply_pose"

    def _apply_pose_in_subprocess(self, input_fbx_path, output_fbx_path, transforms_json_path):
        result_path = tempfile.NamedTemporaryFile(suffix="_sam3d_pose_result.json", delete=False).name
        payload = {
            "input_fbx_path": input_fbx_path,
            "output_fbx_path": output_fbx_path,
            "transforms_json_path": transforms_json_path,
            "result_path": result_path,
        }

        with tempfile.NamedTemporaryFile(suffix="_sam3d_pose_payload.json", delete=False, mode="w", encoding="utf-8") as fp:
            payload_path = fp.name
            json.dump(payload, fp)

        try:
            cmd = [sys.executable, __file__, "--sam3dbody-pose", payload_path]
            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"
            env["SAM3DBODY_BPY_SUBPROC"] = "1"
            env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p])

            completed = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                stdout = (completed.stdout or "").strip()
                msg = stderr if stderr else stdout
                raise RuntimeError(f"SAM3DBody bpy subprocess pose failed (code {completed.returncode}): {msg}")

            if not os.path.exists(result_path):
                stderr = (completed.stderr or "").strip()
                stdout = (completed.stdout or "").strip()
                msg = stderr if stderr else stdout
                raise RuntimeError(f"SAM3DBody bpy subprocess pose returned no result file: {msg}")

            with open(result_path, "r", encoding="utf-8") as rf:
                return json.load(rf)
        finally:
            if os.path.exists(payload_path):
                try:
                    os.unlink(payload_path)
                except Exception:
                    pass
            if os.path.exists(result_path):
                try:
                    os.unlink(result_path)
                except Exception:
                    pass

    def apply_pose(self, input_fbx_path, output_fbx_path, transforms_json_path):
        """
        Load an FBX, apply bone transforms, and export to new FBX.

        Args:
            input_fbx_path: Path to input FBX file
            output_fbx_path: Path to output FBX file
            transforms_json_path: Path to JSON file containing bone transforms
        """
        try:
            _prepare_bpy_import_environment()
            import bpy
        except ImportError:
            if os.environ.get("SAM3DBODY_BPY_SUBPROC") == "1":
                raise
            return self._apply_pose_in_subprocess(
                input_fbx_path=input_fbx_path,
                output_fbx_path=output_fbx_path,
                transforms_json_path=transforms_json_path,
            )
        import mathutils
        import json

        log.info(f" Loading FBX: {input_fbx_path}")

        # Clear the scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Import the FBX
        bpy.ops.import_scene.fbx(filepath=input_fbx_path)

        # Load bone transforms from JSON
        with open(transforms_json_path, 'r') as f:
            bone_transforms = json.load(f)

        log.info(f" Loaded {len(bone_transforms)} bone transforms")

        # Find the armature
        armature = None
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                armature = obj
                break

        if not armature:
            return {"success": False, "error": "No armature found in FBX"}

        log.info(f" Found armature: {armature.name}")
        log.info(f" Armature has {len(armature.pose.bones)} pose bones")

        # Apply transforms to pose bones
        applied_count = 0
        for bone_name, transform in bone_transforms.items():
            if bone_name not in armature.pose.bones:
                log.info(f" WARNING: Bone '{bone_name}' not found in armature")
                continue

            pose_bone = armature.pose.bones[bone_name]

            # Apply position delta (offset from rest pose)
            pos_delta = transform.get('position', {})
            if pos_delta:
                pose_bone.location.x += pos_delta.get('x', 0)
                pose_bone.location.y += pos_delta.get('y', 0)
                pose_bone.location.z += pos_delta.get('z', 0)

            # Apply rotation delta (quaternion multiply)
            quat_delta = transform.get('quaternion', {})
            if quat_delta:
                delta_quat = mathutils.Quaternion((
                    quat_delta.get('w', 1.0),
                    quat_delta.get('x', 0.0),
                    quat_delta.get('y', 0.0),
                    quat_delta.get('z', 0.0)
                ))
                # Multiply current rotation by delta
                pose_bone.rotation_quaternion = pose_bone.rotation_quaternion @ delta_quat

            # Apply scale delta (multiply)
            scale_delta = transform.get('scale', {})
            if scale_delta:
                pose_bone.scale.x *= scale_delta.get('x', 1.0)
                pose_bone.scale.y *= scale_delta.get('y', 1.0)
                pose_bone.scale.z *= scale_delta.get('z', 1.0)

            applied_count += 1

        log.info(f" Applied transforms to {applied_count} bones")

        # Update the scene to apply transforms
        bpy.context.view_layer.update()

        # Export to FBX with current pose
        os.makedirs(os.path.dirname(output_fbx_path) if os.path.dirname(output_fbx_path) else '.', exist_ok=True)

        log.info(f" Exporting posed FBX: {output_fbx_path}")
        bpy.ops.export_scene.fbx(
            filepath=output_fbx_path,
            use_selection=False,
            apply_scale_options='FBX_SCALE_ALL',
            bake_anim=False,
            add_leaf_bones=False,
        )

        log.info(f" Export complete")
        return {"success": True, "output_path": output_fbx_path}


def find_mhr_model_path(mesh_data=None):
    """
    Find the MHR model path using multiple fallback strategies.

    Args:
        mesh_data: Optional mesh_data dict that may contain mhr_path

    Returns:
        str: Path to mhr_model.pt or None if not found
    """
    # Strategy 1: Check mesh_data for explicitly provided path
    if mesh_data and mesh_data.get("mhr_path"):
        mhr_path = mesh_data["mhr_path"]
        if os.path.exists(mhr_path):
            return mhr_path

    # Strategy 2: Check environment variable
    env_path = os.environ.get("SAM3D_MHR_PATH", "")
    if env_path and os.path.exists(env_path):
        return env_path

    # Strategy 3: Search ComfyUI models/sam3dbody/ folder
    sam3dbody_dir = os.path.join(folder_paths.models_dir, "sam3dbody", "assets", "mhr_model.pt")
    if os.path.exists(sam3dbody_dir):
        return sam3dbody_dir

    # Strategy 4 (legacy): Search HuggingFace cache for backwards compatibility
    hf_cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3")
    if os.path.exists(hf_cache_base):
        pattern = os.path.join(hf_cache_base, "snapshots", "*", "assets", "mhr_model.pt")
        matches = glob.glob(pattern)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]

    return None


class SAM3DBodyExportFBX(io.ComfyNode):
    """
    Export SAM3D Body mesh with skeleton to FBX format.

    Takes mesh data from SAM3D and exports it as a rigged FBX file
    using Blender for format conversion.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DBodyExportFBX",
            display_name="SAM 3D Body: Export FBX",
            category="OpenBlender/SAM3DBody/export",
            is_output_node=True,
            inputs=[
                io.Custom("SAM3D_OUTPUT").Input("mesh_data",
                    tooltip="Mesh data from SAM3D Body Process node"),
                io.String.Input("output_filename", default="sam3d_rigged.fbx",
                    tooltip="Output filename for the FBX file"),
            ],
            outputs=[
                io.String.Output(display_name="fbx_path"),
            ],
        )

    @classmethod
    def execute(cls, mesh_data, output_filename):
        """Export mesh with skeleton to FBX format."""

        # Extract mesh data
        vertices = mesh_data.get("vertices")
        faces = mesh_data.get("faces")
        joint_coords = mesh_data.get("joint_coords")  # 127 joints

        if vertices is None or faces is None:
            raise RuntimeError("Mesh vertices or faces not found in mesh_data")

        # Convert tensors to numpy if needed
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        if joint_coords is not None and isinstance(joint_coords, torch.Tensor):
            joint_coords = joint_coords.cpu().numpy()

        # Prepare output path
        output_dir = folder_paths.get_output_directory()
        if not output_filename.endswith('.fbx'):
            output_filename = output_filename + '.fbx'
        output_fbx_path = os.path.join(output_dir, output_filename)

        # Create a simple OBJ file first (Blender can import this easily)
        temp_dir = folder_paths.get_temp_directory()
        temp_obj_path = os.path.join(temp_dir, f"temp_mesh_{int(time.time())}.obj")

        # Write OBJ file
        cls._write_obj_file(temp_obj_path, vertices, faces)

        # Save skeleton data if available
        skeleton_json_path = None
        if joint_coords is not None:
            skeleton_json_path = os.path.join(temp_dir, f"skeleton_{int(time.time())}.json")

            # Convert mesh bounds to plain Python types (with coordinate transform applied)
            mesh_min = vertices.min(axis=0)
            mesh_max = vertices.max(axis=0)
            if isinstance(mesh_min, np.ndarray):
                mesh_min = [float(x) for x in mesh_min]
            if isinstance(mesh_max, np.ndarray):
                mesh_max = [float(x) for x in mesh_max]
            # Apply same transform as mesh: flip Y and Z axes
            mesh_min = [mesh_min[0], -mesh_min[1], -mesh_min[2]]
            mesh_max = [mesh_max[0], -mesh_max[1], -mesh_max[2]]
            # Ensure min < max after flipping (signs reverse order)
            mesh_min, mesh_max = [min(mesh_min[i], mesh_max[i]) for i in range(3)], [max(mesh_min[i], mesh_max[i]) for i in range(3)]

            # Start from predicted joints, may be overridden by canonical rest pose from MHR
            joint_coords_flipped = joint_coords.copy()
            joint_coords_flipped[:, 1] = -joint_coords_flipped[:, 1]
            joint_coords_flipped[:, 2] = -joint_coords_flipped[:, 2]

            skeleton_data = {
                "joint_positions": joint_coords_flipped.tolist(),
                "num_joints": len(joint_coords),
                "mesh_vertices_bounds_min": mesh_min,
                "mesh_vertices_bounds_max": mesh_max,
            }

            # Extract skinning weights from MHR model
            try:
                mhr_model_path = find_mhr_model_path(mesh_data)

                if mhr_model_path and os.path.exists(mhr_model_path):
                    mhr_model = torch.jit.load(mhr_model_path, map_location='cpu')
                    skeleton = mhr_model.character_torch.skeleton
                    skeleton_data["joint_names"] = list(skeleton.joint_names)
                    skeleton_data["joint_parents"] = skeleton.joint_parents.cpu().numpy().astype(int).tolist()

                    lbs = mhr_model.character_torch.linear_blend_skinning

                    vert_indices = lbs.vert_indices_flattened.cpu().numpy().astype(int)
                    skin_indices = lbs.skin_indices_flattened.cpu().numpy().astype(int)
                    skin_weights = lbs.skin_weights_flattened.cpu().numpy().astype(float)

                    vertex_weights = {}
                    for i in range(len(vert_indices)):
                        vert_idx = int(vert_indices[i])
                        bone_idx = int(skin_indices[i])
                        weight = float(skin_weights[i])

                        if vert_idx not in vertex_weights:
                            vertex_weights[vert_idx] = []
                        vertex_weights[vert_idx].append([bone_idx, weight])

                    skinning_data = []
                    num_vertices = len(vertices)
                    for vert_idx in range(num_vertices):
                        if vert_idx in vertex_weights:
                            skinning_data.append(vertex_weights[vert_idx])
                        else:
                            skinning_data.append([])

                    skeleton_data["skinning_weights"] = skinning_data
            except Exception:
                pass  # Skip skinning weights if extraction fails

            # Get joint parent hierarchy from mesh_data (override only if not present)
            joint_parents = None
            joint_rotations = mesh_data.get("joint_rotations")

            if isinstance(joint_rotations, dict) and "joint_parents" in joint_rotations:
                joint_parents_data = joint_rotations["joint_parents"]
            else:
                joint_parents_data = mesh_data.get("joint_parents")

            if joint_parents_data is not None:
                if isinstance(joint_parents_data, np.ndarray):
                    joint_parents = joint_parents_data.astype(int).tolist()
                elif isinstance(joint_parents_data, torch.Tensor):
                    joint_parents = joint_parents_data.cpu().numpy().astype(int).tolist()
                else:
                    joint_parents = [int(p) for p in joint_parents_data]
                skeleton_data["joint_parents"] = joint_parents
            else:
                # Load joint parents from MHR model if we have 127 joints
                if len(joint_coords) == 127:
                    try:
                        mhr_model_path = find_mhr_model_path(mesh_data)
                        if mhr_model_path and os.path.exists(mhr_model_path):
                            mhr_model = torch.jit.load(mhr_model_path, map_location='cpu')
                            joint_parents_tensor = mhr_model.character_torch.skeleton.joint_parents
                            joint_parents = joint_parents_tensor.cpu().numpy().astype(int).tolist()
                            skeleton_data["joint_parents"] = joint_parents
                    except Exception:
                        pass

            # Add camera and focal length if available
            camera = mesh_data.get("camera")
            focal_length = mesh_data.get("focal_length")
            if camera is not None:
                if isinstance(camera, torch.Tensor):
                    camera = camera.cpu().numpy()
                skeleton_data["camera"] = [float(x) for x in camera.flatten()] if isinstance(camera, np.ndarray) else camera
            if focal_length is not None:
                if isinstance(focal_length, (torch.Tensor, np.ndarray)):
                    focal_length = float(focal_length.item() if hasattr(focal_length, 'item') else focal_length)
                skeleton_data["focal_length"] = float(focal_length)

            with open(skeleton_json_path, 'w') as f:
                json.dump(skeleton_data, f)

        try:
            # Use isolated bpy exporter in sam3dbody venv
            exporter = BpyFBXExporter()
            result = exporter.export(
                input_obj_path=temp_obj_path,
                output_fbx_path=output_fbx_path,
                skeleton_json_path=skeleton_json_path
            )

            if not result.get("success"):
                raise RuntimeError(f"FBX export failed")

            if not os.path.exists(output_fbx_path):
                raise RuntimeError(f"Export completed but output file not found: {output_fbx_path}")

            return io.NodeOutput(output_fbx_path)

        finally:
            # Clean up temporary files
            if os.path.exists(temp_obj_path):
                os.unlink(temp_obj_path)
            if skeleton_json_path and os.path.exists(skeleton_json_path):
                os.unlink(skeleton_json_path)

    @staticmethod
    def _write_obj_file(filepath, vertices, faces):
        """Write mesh to OBJ file format."""
        with open(filepath, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]:.6f} {-v[1]:.6f} {-v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


class SAM3DBodyExportMultipleFBX(io.ComfyNode):
    """
    Export multiple SAM3D Body meshes with skeletons to a single FBX file.

    Takes multi-person mesh data and exports all meshes with their armatures
    into a single combined FBX file.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DBodyExportMultipleFBX",
            display_name="SAM 3D Body: Export Multiple FBX",
            category="OpenBlender/SAM3DBody/export",
            is_output_node=True,
            inputs=[
                io.Custom("SAM3D_MULTI_OUTPUT").Input("multi_mesh_data",
                    tooltip="Multi-person mesh data from SAM3D Body Process Multiple node"),
                io.String.Input("output_filename", default="sam3d_multi_rigged.fbx",
                    tooltip="Output filename for the combined FBX file"),
                io.Boolean.Input("combine", default=True,
                    tooltip="When enabled, exports all people into a single FBX file (works with Preview3D). When disabled, creates separate FBX files per person."),
            ],
            outputs=[
                io.String.Output(display_name="fbx_path"),
            ],
        )

    @classmethod
    def execute(cls, multi_mesh_data, output_filename, combine):
        """Export all meshes with skeletons to FBX file(s).

        Args:
            multi_mesh_data: Multi-person mesh data from SAM3D Body Process Multiple node
            output_filename: Output filename for the FBX file
            combine: If True, export all people into single FBX. If False, create separate FBX per person.
        """
        import comfy.model_management
        import comfy.utils

        num_people = multi_mesh_data.get("num_people", 0)
        people = multi_mesh_data.get("people", [])
        faces = multi_mesh_data.get("faces")

        log.info(f" num_people from data: {num_people}")
        log.info(f" actual people list length: {len(people)}")

        if num_people == 0 or len(people) == 0:
            raise RuntimeError("No mesh data to export")

        # Setup output path
        output_dir = folder_paths.get_output_directory()
        if not output_filename.endswith('.fbx'):
            output_filename = output_filename + '.fbx'
        output_fbx_path = os.path.join(output_dir, output_filename)

        # Find MHR model path for skinning weights
        mhr_model_path = find_mhr_model_path(multi_mesh_data)

        # Load skinning data once (same for all people)
        skinning_data = None
        joint_parents = None
        joint_names = None
        if mhr_model_path and os.path.exists(mhr_model_path):
            try:
                mhr_model = torch.jit.load(mhr_model_path, map_location='cpu')
                skeleton = mhr_model.character_torch.skeleton
                lbs = mhr_model.character_torch.linear_blend_skinning

                vert_indices = lbs.vert_indices_flattened.cpu().numpy().astype(int)
                skin_indices = lbs.skin_indices_flattened.cpu().numpy().astype(int)
                skin_weights = lbs.skin_weights_flattened.cpu().numpy().astype(float)

                vertex_weights = {}
                for j in range(len(vert_indices)):
                    vert_idx = int(vert_indices[j])
                    bone_idx = int(skin_indices[j])
                    weight = float(skin_weights[j])
                    if vert_idx not in vertex_weights:
                        vertex_weights[vert_idx] = []
                    vertex_weights[vert_idx].append([bone_idx, weight])

                # Get joint parents
                joint_parents = skeleton.joint_parents.cpu().numpy().astype(int).tolist()
                joint_names = list(skeleton.joint_names)
            except Exception:
                pass

        # Build combined data structure for all people
        temp_files = []
        combined_data = {
            "output_path": output_fbx_path,
            "people": [],
        }

        try:
            pbar = comfy.utils.ProgressBar(len(people))
            for i, person in enumerate(people):
                comfy.model_management.throw_exception_if_processing_interrupted()
                vertices = person.get("pred_vertices")
                joint_coords = person.get("pred_joint_coords")
                cam_t = person.get("pred_cam_t")  # Camera translation for world positioning
                global_rots = person.get("pred_global_rots")  # Global joint rotations for bone orientations

                if vertices is None:
                    pbar.update(1)
                    continue

                # Convert to numpy
                if isinstance(vertices, torch.Tensor):
                    vertices = vertices.cpu().numpy()
                if joint_coords is not None and isinstance(joint_coords, torch.Tensor):
                    joint_coords = joint_coords.cpu().numpy()
                if cam_t is not None and isinstance(cam_t, torch.Tensor):
                    cam_t = cam_t.cpu().numpy()
                if global_rots is not None and isinstance(global_rots, torch.Tensor):
                    global_rots = global_rots.cpu().numpy()

                # Apply world position offset from camera translation
                if cam_t is not None:
                    vertices = vertices + cam_t  # Broadcast adds cam_t to each vertex
                    if joint_coords is not None:
                        joint_coords = joint_coords + cam_t

                # Write OBJ file for this person
                temp_obj = tempfile.NamedTemporaryFile(suffix=f'_person{i}.obj', delete=False)
                temp_files.append(temp_obj.name)
                cls._write_obj_file(temp_obj.name, vertices, faces)

                # Prepare skeleton data
                skeleton_info = {}
                if joint_coords is not None:
                    # Apply coordinate transform to joint positions (flip Y and Z)
                    joint_coords_flipped = joint_coords.copy()
                    joint_coords_flipped[:, 1] = -joint_coords_flipped[:, 1]
                    joint_coords_flipped[:, 2] = -joint_coords_flipped[:, 2]

                    skeleton_info = {
                        "joint_positions": joint_coords_flipped.tolist(),
                        "num_joints": len(joint_coords),
                    }

                    if joint_names:
                        skeleton_info["joint_names"] = joint_names

                    # Add skinning weights (build per-vertex data)
                    if vertex_weights:
                        num_vertices = len(vertices)
                        skinning_list = []
                        for vert_idx in range(num_vertices):
                            if vert_idx in vertex_weights:
                                skinning_list.append(vertex_weights[vert_idx])
                            else:
                                skinning_list.append([])
                        skeleton_info["skinning_weights"] = skinning_list

                    # Add joint parents
                    if joint_parents:
                        skeleton_info["joint_parents"] = joint_parents

                    # Add global joint rotations for better bone orientations in FBX
                    if global_rots is not None:
                        skeleton_info["global_rotations"] = global_rots.tolist()
                        log.info(f" Person {i}: Including global_rots shape {global_rots.shape}")

                # Add person to combined data
                combined_data["people"].append({
                    "obj_path": temp_obj.name,
                    "skeleton": skeleton_info,
                    "index": i,
                })
                pbar.update(1)

            log.info(f" people added to combined_data: {len(combined_data['people'])}")
            log.info(f" combine: {combine}")

            if not combined_data["people"]:
                raise RuntimeError("No valid mesh data to export")

            exporter = BpyFBXExporter()

            if combine:
                # Combined mode: export all people into a single FBX file
                # Write skeleton JSON files for each person and build combined data
                people_data_for_export = []
                for person_data in combined_data["people"]:
                    idx = person_data["index"]
                    skeleton_info = person_data.get("skeleton", {})

                    # Write skeleton JSON for this person if available
                    person_skeleton_json = None
                    if skeleton_info:
                        person_skeleton_json = tempfile.NamedTemporaryFile(
                            suffix=f'_person{idx}_skeleton.json', delete=False, mode='w'
                        )
                        temp_files.append(person_skeleton_json.name)
                        json.dump(skeleton_info, person_skeleton_json)
                        person_skeleton_json.close()
                        person_skeleton_json = person_skeleton_json.name

                    people_data_for_export.append({
                        "obj_path": person_data["obj_path"],
                        "skeleton_json_path": person_skeleton_json,
                        "index": idx,
                    })

                # Write combined data JSON file
                combined_json = tempfile.NamedTemporaryFile(
                    suffix='_combined_export.json', delete=False, mode='w'
                )
                temp_files.append(combined_json.name)
                json.dump(people_data_for_export, combined_json)
                combined_json.close()

                # Export all people into single FBX via combined_json_path
                result = exporter.export(
                    input_obj_path=None,
                    output_fbx_path=output_fbx_path,
                    combined_json_path=combined_json.name
                )

                if not result.get("success"):
                    raise RuntimeError("Combined FBX export failed")

                log.info(f" Combined FBX created: {output_fbx_path}")
                return io.NodeOutput(output_fbx_path)

            else:
                # Separate mode: export each person to individual FBX files
                exported_files = []
                pbar_export = comfy.utils.ProgressBar(len(combined_data["people"]))

                for person_data in combined_data["people"]:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    obj_path = person_data["obj_path"]
                    idx = person_data["index"]
                    skeleton_info = person_data.get("skeleton", {})

                    # Create per-person FBX filename
                    if len(combined_data["people"]) == 1:
                        person_fbx_path = output_fbx_path
                    else:
                        person_fbx_path = output_fbx_path.replace('.fbx', f'_person{idx}.fbx')

                    # Write skeleton JSON for this person if available
                    person_skeleton_json = None
                    if skeleton_info:
                        person_skeleton_json = tempfile.NamedTemporaryFile(
                            suffix=f'_person{idx}_skeleton.json', delete=False, mode='w'
                        )
                        temp_files.append(person_skeleton_json.name)
                        json.dump(skeleton_info, person_skeleton_json)
                        person_skeleton_json.close()
                        person_skeleton_json = person_skeleton_json.name

                    # Export using isolated bpy
                    result = exporter.export(
                        input_obj_path=obj_path,
                        output_fbx_path=person_fbx_path,
                        skeleton_json_path=person_skeleton_json
                    )

                    if result.get("success"):
                        exported_files.append(person_fbx_path)
                    else:
                        raise RuntimeError(f"FBX export failed for person {idx}")
                    pbar_export.update(1)

                if not exported_files:
                    raise RuntimeError("No FBX files were exported")

                # Return the first exported file (separate mode returns first file for compatibility)
                output_fbx_path = exported_files[0]
                log.info(f" Separate FBX files created: {len(exported_files)} files")
                return io.NodeOutput(output_fbx_path)

        finally:
            # Clean up temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception:
                        pass

    @staticmethod
    def _write_obj_file(filepath, vertices, faces):
        """Write mesh to OBJ file format."""
        with open(filepath, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]:.6f} {-v[1]:.6f} {-v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


class SAM3DBodyExportNPZ(io.ComfyNode):
    """Export SAM3D mesh + rig data to NPZ without bpy."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DBodyExportNPZ",
            display_name="SAM 3D Body: Export NPZ",
            category="OpenBlender/SAM3DBody/export",
            is_output_node=True,
            inputs=[
                io.Custom("SAM3D_OUTPUT").Input(
                    "mesh_data",
                    tooltip="Mesh/rig data from SAM3D Body Process node",
                ),
                io.String.Input(
                    "output_filename",
                    default="sam3d_rigged.npz",
                    tooltip="Output NPZ filename",
                ),
            ],
            outputs=[
                io.String.Output(display_name="npz_path"),
            ],
        )

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            try:
                return np.asarray(value)
            except Exception:
                return np.array(value, dtype=object)
        if isinstance(value, dict):
            return np.array(value, dtype=object)
        return np.asarray(value)

    @staticmethod
    def _ensure_2d(value, width=None, default_rows=1):
        arr = SAM3DBodyExportNPZ._to_numpy(value)
        if arr is None:
            cols = width if width is not None else 0
            return np.zeros((default_rows, cols), dtype=np.float32)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        if width is not None:
            if arr.shape[1] < width:
                pad = np.zeros((arr.shape[0], width - arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=1)
            elif arr.shape[1] > width:
                arr = arr[:, :width]

        return arr

    @classmethod
    def execute(cls, mesh_data, output_filename):
        if not isinstance(mesh_data, dict):
            raise RuntimeError("mesh_data must be a dictionary")

        output_dir = folder_paths.get_output_directory()
        if not output_filename.endswith('.npz'):
            output_filename = output_filename + '.npz'
        output_npz_path = os.path.join(output_dir, output_filename)

        pose_params = mesh_data.get("pose_params") or {}

        body_pose = cls._ensure_2d(pose_params.get("body_pose"), width=63)
        betas = cls._ensure_2d(pose_params.get("shape"), width=10)
        global_orient = cls._ensure_2d(pose_params.get("global_rot"), width=3)

        transl_src = mesh_data.get("camera")
        transl = cls._ensure_2d(transl_src, width=3)
        # Fallback if camera translation is unavailable
        if transl.size == 0 or transl.shape[1] != 3:
            transl = np.zeros((body_pose.shape[0], 3), dtype=np.float32)

        # Keep frame counts aligned where possible
        n_frames = body_pose.shape[0]
        if global_orient.shape[0] != n_frames:
            global_orient = np.repeat(global_orient[:1], n_frames, axis=0)
        if transl.shape[0] != n_frames:
            transl = np.repeat(transl[:1], n_frames, axis=0)

        # SAM3D does not currently expose reliable in-camera pose directly in this path.
        # Mirror global values for compatibility with importers requiring *_incam keys.
        global_orient_incam = np.array(global_orient, copy=True)
        transl_incam = np.array(transl, copy=True)

        payload = {
            # SMPL/Kimodo-compatible keys
            "body_pose": body_pose,
            "betas": betas,
            "global_orient": global_orient,
            "transl": transl,
            "global_orient_incam": global_orient_incam,
            "transl_incam": transl_incam,
            # Dotted namespace variants used by some importers
            "global.body_pose": body_pose,
            "global.betas": betas,
            "global.global_orient": global_orient,
            "global.transl": transl,
            "incam.body_pose": body_pose,
            "incam.betas": betas,
            "incam.global_orient": global_orient_incam,
            "incam.transl": transl_incam,
            # Keep original rich data for debugging/advanced consumers
            "vertices": cls._to_numpy(mesh_data.get("vertices")),
            "faces": cls._to_numpy(mesh_data.get("faces")),
            "joints": cls._to_numpy(mesh_data.get("joints")),
            "joint_coords": cls._to_numpy(mesh_data.get("joint_coords")),
            "joint_rotations": cls._to_numpy(mesh_data.get("joint_rotations")),
            "joint_parents": cls._to_numpy(mesh_data.get("joint_parents")),
            "camera": cls._to_numpy(mesh_data.get("camera")),
            "focal_length": cls._to_numpy(mesh_data.get("focal_length")),
            "bbox": cls._to_numpy(mesh_data.get("bbox")),
            "pose_body": body_pose,
            "pose_hand": cls._to_numpy(pose_params.get("hand_pose")),
            "pose_global_rot": global_orient,
            "pose_shape": betas,
            "pose_scale": cls._to_numpy(pose_params.get("scale")),
            "pose_expr": cls._to_numpy(pose_params.get("expr")),
            "mhr_path": np.array(str(mesh_data.get("mhr_path", ""))),
        }

        for key, val in list(payload.items()):
            if val is None:
                payload[key] = np.array([], dtype=np.float32)

        os.makedirs(os.path.dirname(output_npz_path) if os.path.dirname(output_npz_path) else '.', exist_ok=True)
        np.savez_compressed(output_npz_path, **payload)
        log.info(" NPZ export complete: %s", output_npz_path)
        return io.NodeOutput(output_npz_path)


class SAM3DBodyExportGLB(io.ComfyNode):
    """Export SAM3D mesh as GLB without bpy."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3DBodyExportGLB",
            display_name="SAM 3D Body: Export GLB",
            category="OpenBlender/SAM3DBody/export",
            is_output_node=True,
            inputs=[
                io.Custom("SAM3D_OUTPUT").Input(
                    "mesh_data",
                    tooltip="Mesh data from SAM3D Body Process node",
                ),
                io.String.Input(
                    "output_filename",
                    default="sam3d_mesh.glb",
                    tooltip="Output GLB filename",
                ),
            ],
            outputs=[
                io.String.Output(display_name="glb_path"),
            ],
        )

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def execute(cls, mesh_data, output_filename):
        if not isinstance(mesh_data, dict):
            raise RuntimeError("mesh_data must be a dictionary")

        vertices = mesh_data.get("vertices")
        faces = mesh_data.get("faces")
        joint_coords = mesh_data.get("joint_coords")

        if vertices is None or faces is None:
            raise RuntimeError("Mesh vertices or faces not found in mesh_data")

        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        if joint_coords is not None and isinstance(joint_coords, torch.Tensor):
            joint_coords = joint_coords.cpu().numpy()

        if vertices.ndim == 3:
            vertices = vertices[0]
        if faces.ndim == 3:
            faces = faces[0]

        output_dir = folder_paths.get_output_directory()
        if not output_filename.endswith('.glb'):
            output_filename = output_filename + '.glb'
        output_glb_path = os.path.join(output_dir, output_filename)
        os.makedirs(os.path.dirname(output_glb_path) if os.path.dirname(output_glb_path) else '.', exist_ok=True)

        temp_dir = folder_paths.get_temp_directory()
        temp_obj_path = os.path.join(temp_dir, f"temp_mesh_{int(time.time())}.obj")
        SAM3DBodyExportFBX._write_obj_file(temp_obj_path, vertices, faces)

        skeleton_json_path = None
        if joint_coords is not None:
            skeleton_json_path = os.path.join(temp_dir, f"skeleton_{int(time.time())}.json")

            mesh_min = vertices.min(axis=0)
            mesh_max = vertices.max(axis=0)
            if isinstance(mesh_min, np.ndarray):
                mesh_min = [float(x) for x in mesh_min]
            if isinstance(mesh_max, np.ndarray):
                mesh_max = [float(x) for x in mesh_max]
            mesh_min = [mesh_min[0], -mesh_min[1], -mesh_min[2]]
            mesh_max = [mesh_max[0], -mesh_max[1], -mesh_max[2]]
            mesh_min, mesh_max = [min(mesh_min[i], mesh_max[i]) for i in range(3)], [max(mesh_min[i], mesh_max[i]) for i in range(3)]

            joint_coords_flipped = joint_coords.copy()
            joint_coords_flipped[:, 1] = -joint_coords_flipped[:, 1]
            joint_coords_flipped[:, 2] = -joint_coords_flipped[:, 2]

            skeleton_data = {
                "joint_positions": joint_coords_flipped.tolist(),
                "num_joints": len(joint_coords),
                "mesh_vertices_bounds_min": mesh_min,
                "mesh_vertices_bounds_max": mesh_max,
            }

            try:
                mhr_model_path = find_mhr_model_path(mesh_data)
                if mhr_model_path and os.path.exists(mhr_model_path):
                    mhr_model = torch.jit.load(mhr_model_path, map_location='cpu')
                    skeleton = mhr_model.character_torch.skeleton
                    skeleton_data["joint_names"] = list(skeleton.joint_names)
                    skeleton_data["joint_parents"] = skeleton.joint_parents.cpu().numpy().astype(int).tolist()
                    lbs = mhr_model.character_torch.linear_blend_skinning

                    vert_indices = lbs.vert_indices_flattened.cpu().numpy().astype(int)
                    skin_indices = lbs.skin_indices_flattened.cpu().numpy().astype(int)
                    skin_weights = lbs.skin_weights_flattened.cpu().numpy().astype(float)

                    vertex_weights = {}
                    for i in range(len(vert_indices)):
                        vert_idx = int(vert_indices[i])
                        bone_idx = int(skin_indices[i])
                        weight = float(skin_weights[i])
                        if vert_idx not in vertex_weights:
                            vertex_weights[vert_idx] = []
                        vertex_weights[vert_idx].append([bone_idx, weight])

                    skinning_data = []
                    num_vertices = len(vertices)
                    for vert_idx in range(num_vertices):
                        if vert_idx in vertex_weights:
                            skinning_data.append(vertex_weights[vert_idx])
                        else:
                            skinning_data.append([])
                    skeleton_data["skinning_weights"] = skinning_data
            except Exception:
                pass

            joint_parents = None
            joint_rotations = mesh_data.get("joint_rotations")
            if isinstance(joint_rotations, dict) and "joint_parents" in joint_rotations:
                joint_parents_data = joint_rotations["joint_parents"]
            else:
                joint_parents_data = mesh_data.get("joint_parents")

            if joint_parents_data is not None:
                if isinstance(joint_parents_data, np.ndarray):
                    joint_parents = joint_parents_data.astype(int).tolist()
                elif isinstance(joint_parents_data, torch.Tensor):
                    joint_parents = joint_parents_data.cpu().numpy().astype(int).tolist()
                else:
                    joint_parents = [int(p) for p in joint_parents_data]
                if "joint_parents" not in skeleton_data:
                    skeleton_data["joint_parents"] = joint_parents
            elif len(joint_coords) == 127:
                try:
                    mhr_model_path = find_mhr_model_path(mesh_data)
                    if mhr_model_path and os.path.exists(mhr_model_path):
                        mhr_model = torch.jit.load(mhr_model_path, map_location='cpu')
                        joint_parents_tensor = mhr_model.character_torch.skeleton.joint_parents
                        skeleton_data["joint_parents"] = joint_parents_tensor.cpu().numpy().astype(int).tolist()
                except Exception:
                    pass

            camera = mesh_data.get("camera")
            focal_length = mesh_data.get("focal_length")
            if camera is not None:
                if isinstance(camera, torch.Tensor):
                    camera = camera.cpu().numpy()
                skeleton_data["camera"] = [float(x) for x in camera.flatten()] if isinstance(camera, np.ndarray) else camera
            if focal_length is not None:
                if isinstance(focal_length, (torch.Tensor, np.ndarray)):
                    focal_length = float(focal_length.item() if hasattr(focal_length, 'item') else focal_length)
                skeleton_data["focal_length"] = float(focal_length)

            with open(skeleton_json_path, 'w') as f:
                json.dump(skeleton_data, f)

        try:
            exporter = BpyFBXExporter()
            result = exporter.export(
                input_obj_path=temp_obj_path,
                output_fbx_path=output_glb_path,
                skeleton_json_path=skeleton_json_path,
            )

            if not result.get("success"):
                raise RuntimeError("GLB export failed")
            if not os.path.exists(output_glb_path):
                raise RuntimeError(f"GLB export completed but output file not found: {output_glb_path}")
            log.info(" GLB export complete: %s", output_glb_path)
            return io.NodeOutput(output_glb_path)
        finally:
            if os.path.exists(temp_obj_path):
                os.unlink(temp_obj_path)
            if skeleton_json_path and os.path.exists(skeleton_json_path):
                os.unlink(skeleton_json_path)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyExportFBX": SAM3DBodyExportFBX,
    "SAM3DBodyExportMultipleFBX": SAM3DBodyExportMultipleFBX,
    "SAM3DBodyExportNPZ": SAM3DBodyExportNPZ,
    "SAM3DBodyExportGLB": SAM3DBodyExportGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyExportFBX": "SAM 3D Body: Export FBX",
    "SAM3DBodyExportMultipleFBX": "SAM 3D Body: Export Multiple FBX",
    "SAM3DBodyExportNPZ": "SAM 3D Body: Export NPZ",
    "SAM3DBodyExportGLB": "SAM 3D Body: Export GLB",
}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("--sam3dbody-export", "--sam3dbody-pose"):
        mode = sys.argv[1]
        payload_path = sys.argv[2]

        with open(payload_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if mode == "--sam3dbody-export":
            result = BpyFBXExporter().export(
                input_obj_path=payload.get("input_obj_path"),
                output_fbx_path=payload.get("output_fbx_path"),
                skeleton_json_path=payload.get("skeleton_json_path"),
                combined_json_path=payload.get("combined_json_path"),
            )
        else:
            result = BpyPoseApplier().apply_pose(
                input_fbx_path=payload.get("input_fbx_path"),
                output_fbx_path=payload.get("output_fbx_path"),
                transforms_json_path=payload.get("transforms_json_path"),
            )

        result_path = payload.get("result_path")
        if result_path:
            with open(result_path, "w", encoding="utf-8") as rf:
                json.dump(result, rf)
        else:
            print(json.dumps(result))
