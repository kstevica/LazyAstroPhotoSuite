"""Dim core — mask the large bright areas (the luminous core veil) and multiplicatively
lower their luminosity, so an over-bright core comes down toward the PixInsight look.

Not a port of a specific PI process: a finishing **dial**. The highlight roll-off can only
compress the top end toward a ceiling (~0.5 + 0.5*knee), so it cannot take a big luminous
core BELOW that — the veil stays bright. This step is unbounded: it builds a smooth mask of
the LARGE bright areas (large-scale luminance above a floor — sky, stars and faint nebula
stay out) and scales luminosity down inside it by ``strength * mask``.

Multiplicative (``L * (1 - strength*mask)``), so relative contrast — the core's structure —
is preserved while the overall level drops; luminance-based (R:G:B scaled together) so the
colour is preserved. Opt-in (0 = off) so it never dims a target that does not need it.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

_LUM = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# The dial (Parameters.dimCore, 0..1) scales the luminosity cut up to this maximum
# (dial 1 -> core luminosity * (1 - 0.80) inside the mask).
_MAX_STRENGTH: float = 0.80
# Large-scale radius (fraction of the long edge) that defines "large bright area".
_SIGMA_FRAC: float = 0.04
# Brightness mask on the large-scale luminance: off below _LO (sky/faint nebula protected),
# full above _HI. A bright core's large-scale luminance sits ~0.3-0.5 on a stretched frame.
_LO: float = 0.18
_HI: float = 0.42


def dim_bright_core(img, dial: float) -> np.ndarray:
    """Lower the luminosity of the large bright core via a smooth mask (see module docstring).

    ``img`` float64 in [0, 1], mono or RGB. ``dial`` 0..1 (0 = off). Colour preserved
    (R:G:B scaled together); sky/stars/faint nebula untouched (below the mask floor).
    Returns a new array; never brightens a pixel.
    """
    a = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    strength = min(max(float(dial), 0.0), 1.0) * _MAX_STRENGTH
    if strength <= 1e-6:
        return a.copy()

    rgb = a.ndim == 3 and a.shape[-1] == 3
    lum = a @ _LUM if rgb else a
    sigma = float(np.clip(max(lum.shape) * _SIGMA_FRAC, 4.0, 200.0))
    bright = gaussian_filter(lum, sigma, mode="nearest")     # large-scale brightness = the veil

    m = np.clip((bright - _LO) / (_HI - _LO), 0.0, 1.0)
    mask = m * m * (3.0 - 2.0 * m)                            # smoothstep: only big bright areas
    lum_new = lum * (1.0 - strength * mask)                   # multiplicative cut, keeps structure

    if not rgb:
        return np.clip(lum_new, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(lum > 1e-6, lum_new / lum, 1.0)
    return np.clip(a * scale[..., None], 0.0, 1.0)
