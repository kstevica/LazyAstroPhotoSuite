"""LazyStack parameters (STK tunables, LazyStretch.js:6900-6960)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LazyStackParams:
    # integration rejection (STK.integrateSet: WinsorizedSigmaClip)
    sigma_low: float = 4.0
    sigma_high: float = 3.0
    # cull (STK.CULL)
    max_reject_frac: float = 0.30      # never cull more than this fraction
    ecc_hard: float = 0.72             # hard eccentricity reject (trailed)
    fwhm_mads: float = 3.0             # soft FWHM reject (bloated) in MADs
    bg_mads: float = 6.0               # background-surge reject in MADs
    # pipeline toggles
    do_calibrate: bool = True
    do_cosmetic: bool = True
    do_debayer: bool = True
    do_register: bool = True
    normalize: bool = True         # match each frame's background+scale to the reference
    reuse_cache: bool = True
    stage_to_disk: bool = True     # stage frames to lazystack/work (low RAM); off = in-memory
