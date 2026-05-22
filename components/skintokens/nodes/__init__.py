from .skintokens_nodes import SkinTokensLoadModel, SkinTokensRigMesh

NODE_CLASS_MAPPINGS = {
    "SkinTokensLoadModel": SkinTokensLoadModel,
    "SkinTokensRigMesh": SkinTokensRigMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkinTokensLoadModel": "SkinTokens Load Model",
    "SkinTokensRigMesh": "SkinTokens Rig Mesh",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
