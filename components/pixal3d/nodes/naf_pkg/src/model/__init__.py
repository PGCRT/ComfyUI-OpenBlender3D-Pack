# Trimmed from upstream: only NAF is vendored. The other 11 upsampler variants
# (AnyUpsampler, Bilinear, FeatUp, IRCNN, JAFAR, JBF, JBU, Nearest, REDNet,
# Restormer, ...) are not used by Pixal3D and were skipped to keep the
# vendored surface minimal. See ../../VENDORED.md.
from .naf import NAF
