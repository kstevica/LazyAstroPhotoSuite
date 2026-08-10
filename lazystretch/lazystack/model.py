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
    fix_walking_noise: bool = True     # cross-frame static hot-pixel repair (darkless walking-noise fix)
    fix_banding: bool = True           # suppress column/row fixed-pattern banding on the master
    do_debayer: bool = True
    do_register: bool = True
    normalize: bool = True         # match each frame's background+scale to the reference (global)
    local_normalize: bool = False  # spatially-varying gradient match to the reference (LN)
    edge_crop: bool = True         # crop the master to the common fully-covered overlap region
    emit_snr_map: bool = True      # write a per-pixel noise/SNR companion (feeds the stretch's SNR mask)
    reuse_cache: bool = True
    stage_to_disk: bool = True     # stage frames to lazystack/work (low RAM); off = in-memory
