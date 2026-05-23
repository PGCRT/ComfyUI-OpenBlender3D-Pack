# Pixal3D submodules now import the wheel names directly (cumesh_vb,
# o_voxel_vb_ap, flex_gemm_ap) -- matching TRELLIS2's pattern. The previous
# sys.modules alias hack here was removed; it silently swallowed ImportError
# and masked real failures.

from . import models
from . import modules
from . import pipelines
from . import representations
from . import utils
