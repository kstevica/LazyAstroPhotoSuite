"""Core transparency — large-scale local contrast masked to the bright nebulosity, so a
bright core reads as a *transparent* structured cloud (like PixInsight) instead of an opaque
veil.

Not a port of a specific PI process: a finishing **dial**. The comparison against PI's M42
showed the port's bright core is not too bright — its *dust lanes are not dark enough*
(veil floor ~0.13 vs PI ~0.085) and its detail contrast is lower, so the core looks filled
in. The fix is to raise the LARGE-SCALE local contrast inside the core: deepen the big dark
lanes and lift the gas around them, so structure shows *through* the veil.

Mechanism: an unsharp mask at a large (dust-lane) scale — ``L + amount*(L - localmean)`` —
applied only through a smooth mask on the local brightness, so only the bright nebulosity is
affected and the sky background is untouched. Luminance-based (R:G:B scaled together) so the
colour is preserved. Distinct from the highlight roll-off (which *dims* the veil); this adds
*structure* to it. Distinct from LHE (small-scale); the dust lanes are large-scale.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

_LUM = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# The dial (Parameters.transparency, 0..1) scales the unsharp amount up to this maximum.
_MAX_AMOUNT: float = 1.5
# Large-scale radius (fraction of the long edge) — the dust-lane scale, well above LHE's.
_SCALE_FRAC: float = 0.02
# Brightness mask: off below _MASK_LO (sky background protected), full above _MASK_LO+_MASK_SPAN.
_MASK_LO: float = 0.12
_MASK_SPAN: float = 0.25


def core_transparency(img, dial: float) -> np.ndarray:
    """Raise large-scale local contrast in the bright core so it reads as transparent.

    ``img`` float64 in [0, 1], mono or RGB. ``dial`` 0..1 (0 = off). Luminance-based, so
    colour ratios are preserved; masked to the bright nebulosity so the sky is untouched.
    Returns a new array.
    """
    a = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    amount = min(max(float(dial), 0.0), 1.0) * _MAX_AMOUNT
    if amount <= 1e-6:
        return a.copy()

    rgb = a.ndim == 3 and a.shape[-1] == 3
    lum = a @ _LUM if rgb else a
    sigma = float(np.clip(max(lum.shape) * _SCALE_FRAC, 4.0, 120.0))
    localmean = gaussian_filter(lum, sigma, mode="nearest")

    m = np.clip((localmean - _MASK_LO) / _MASK_SPAN, 0.0, 1.0)
    mask = m * m * (3.0 - 2.0 * m)                      # smoothstep: only bright nebulosity
    lum_c = np.clip(lum + amount * (lum - localmean) * mask, 0.0, 1.0)

    if not rgb:
        return lum_c
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(lum > 1e-6, lum_c / lum, 1.0)
    return np.clip(a * scale[..., None], 0.0, 1.0)
