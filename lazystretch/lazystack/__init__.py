"""LazyStack — calibrate, measure/cull, register and integrate subs into a master.

A port of the ``STK`` engine embedded in LazyStretch.js (the Stack tab). PixInsight's native
processes are replaced by scientific-Python equivalents: calibration + integration in
numpy/astropy, sub-frame measurement via photutils, registration via ``astroalign`` when
installed (falling back to the translation-only FFT engine otherwise). The custom STK math
(MAD cull + reference ranking, edge-contract measurement) is ported directly.

Optional pro libraries lift fidelity where present: ``astroalign`` (distortion-free affine
registration), ``ccdproc`` (calibration), ``astroscrappy`` (cosmetic correction). Each is
feature-detected; without them LazyStack degrades gracefully and says so.
"""
from __future__ import annotations

__all__ = ["model", "calibrate", "integrate", "measure", "register", "contract", "run"]
