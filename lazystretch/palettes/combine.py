"""Narrowband channel combination (LazyStretch.js:804-861).

Mono SHO/HOS/HOO combine of separate Ha/OIII/SII masters, and the OSC narrowband
*simulation* (pseudo-Ha from red, pseudo-OIII from the green+blue mean). Pure channel
algebra — the exact PixelMath expressions, ported verbatim.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _as_mono(a: np.ndarray) -> np.ndarray:
    """Coerce a channel to 2-D float64 (accept (H,W) or (H,W,1))."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., 0]
    return a


def combine_narrowband(key: str, ha: np.ndarray, oiii: np.ndarray,
                       sii: Optional[np.ndarray] = None) -> np.ndarray:
    """Combine mono Ha/OIII/(SII) into a linear RGB image per palette (js:804-828).

    SHO: R=SII|Ha G=Ha  B=OIII;  HOS: R=Ha G=OIII B=SII|OIII;  HOO: R=Ha G=OIII B=OIII.
    SII is optional and falls back to Ha (SHO) / OIII (HOS).
    """
    ha = _as_mono(ha)
    oiii = _as_mono(oiii)
    sii_a = _as_mono(sii) if sii is not None else None

    k = (key or "").upper()
    if k == "SHO":
        r = sii_a if sii_a is not None else ha
        g = ha
        b = oiii
    elif k == "HOS":
        r = ha
        g = oiii
        b = sii_a if sii_a is not None else oiii
    else:  # HOO / Bicolor
        r = ha
        g = oiii
        b = oiii
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def combine_osc_narrowband(osc: np.ndarray, key: str) -> np.ndarray:
    """Narrowband *simulation* from an OSC RGB image (js:837-861).

    ha = R;  o3 = (G + B) / 2.
    SHO(sim): R=ha G=(ha+o3)/2 B=o3;  otherwise HOO: R=ha G=o3 B=o3.
    """
    a = np.asarray(osc, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("OSC narrowband simulation needs a 3-channel RGB image")
    ha = a[..., 0]
    o3 = (a[..., 1] + a[..., 2]) / 2.0
    if (key or "").upper() == "SHO":
        r, g, b = ha, (ha + o3) / 2.0, o3
    else:  # HOO (and HOS -> HOO, no real SII)
        r, g, b = ha, o3, o3
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
