# Vendored NAF

Source: https://github.com/valeoai/NAF @ 37f2dfc180f2de53d98bd601109c0da0dd6b0f43
Date: 2026-05-20
Subset: minimum import chain for `from src.model.naf import NAF`:
  - src/__init__.py        (empty, added -- not present upstream)
  - src/layers/{__init__,attentions,convolutions,rope}.py
  - src/model/{__init__,base,naf}.py  (__init__.py trimmed to re-export only NAF)
  - LICENSE                (Apache 2.0)

Excluded: train.py, evaluation/, test/, notebooks/, hydra_plugins/, config/,
docs/, asset/, utils/, hubconf.py, denoising.py, src/loss.py, src/backbone/,
and the 11 other upsampler variants under src/model/.
