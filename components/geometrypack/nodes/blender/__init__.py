NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    import bpy  # noqa: F401
    _BYP_AVAILABLE = True
except Exception:
    _BYP_AVAILABLE = False

if _BYP_AVAILABLE:
    # Import submodules only when bpy is available.
    from . import blender_io
    from . import boolean
    from . import remeshing
    from . import texture_remeshing
    from . import uv

    NODE_CLASS_MAPPINGS.update(blender_io.NODE_CLASS_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(boolean.NODE_CLASS_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(remeshing.NODE_CLASS_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(texture_remeshing.NODE_CLASS_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(uv.NODE_CLASS_MAPPINGS)

    NODE_DISPLAY_NAME_MAPPINGS.update(blender_io.NODE_DISPLAY_NAME_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(boolean.NODE_DISPLAY_NAME_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(remeshing.NODE_DISPLAY_NAME_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(texture_remeshing.NODE_DISPLAY_NAME_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(uv.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
