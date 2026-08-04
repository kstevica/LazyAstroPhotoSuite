"""Highlight roll-off — dial down blown highlights while REVEALING core structure.

Not a port of a specific PI process: a **calibration / finishing dial** to tame the port's
blown grey-white cores and star cores. The golden survey (calibration/survey.py) showed PI
keeps near-white ≈ 0% on every target — bright cores roll off just under white and keep
colour + structure — while the port's finishing steps push bright pixels toward (1,1,1).

Two design choices matter, both learned the hard way against real M42 data:

* **Luminance, not per-channel.** Compressing R, G, B independently pulls them toward each
  other and *desaturates* — a pink core greys to white. Instead the **luminance** L is the
  thing dialed down, and R:G:B are scaled together by ``L'/L`` so the colour ratios (hue +
  saturation) are preserved exactly: the core dims but stays pink.

* **Base/detail, not a flat squeeze.** Simply compressing L also *flattens* it (a squeezed
  range has less local contrast), so the blob dims but shows no structure. Instead L is
  split into a smooth **base** (the large-scale bright blob) and **detail** (``L - base``,
  the filaments); only the base is compressed and the detail is added back at full strength
  — so the blob dims AND the structure inside it is revealed. A final gentle top cap keeps
  bright points (stars) from blowing to pure white. Run last.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# Rec. 709 luminance weights (same as tone.py) — the "luminosity" that is dialed down.
_LUM = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# Knee where the roll-off begins. Tuned against the golden set so the brightest pixels
# land near PI's peak (~0.90-0.95) and near-white % approaches PI's ~0-0.1%.
_HIGHLIGHT_KNEE: float = 0.82

# The user-facing "Highlights" dial (Parameters.highlights, 0..1): LOW = highlights dialed
# DOWN (dim), HIGH = bright / untouched — the photographic-slider convention. dial 0 -> knee
# low (deep base compression: the bright core is dimmed hard and its structure revealed);
# dial 1 -> knee high (highlights nearly untouched). A lower knee starts earlier and
# (ceiling = 0.5 + 0.5*knee) caps the base lower. The 0.10 floor lets the low end bite HARD:
# dial 0 -> knee 0.10 (ceiling 0.55; near the whole nebula base is compressed = maximum
# structure, but the image loses overall brightness/punch), dial ~0.02 -> ~0.12.
_KNEE_AT_DIAL_0: float = 0.10   # dial 0 -> strongest dial-down (dim highlights, ceiling 0.55)
_KNEE_AT_DIAL_1: float = 0.90   # dial 1 -> gentlest (bright highlights, near-off)

# Base-smoothing scale as a fraction of the frame's long edge — large enough that the bright
# CORE lives in the base (so it is compressed) but filaments / stars stay in the detail (so
# they are preserved). Clamped so tiny previews and huge masters both behave.
_BASE_SIGMA_FRAC: float = 0.0027

# Fixed gentle top cap applied after the base/detail pass so bright DETAIL (star cores) still
# never reaches pure white — the base/detail pass alone leaves isolated stars (low base) hot.
_STAR_CAP_KNEE: float = 0.90


def knee_for_dial(dial: float) -> float:
    """Map the 0..1 ``highlights`` dial to a base-compression knee (LOWER dial -> lower knee -> dimmer)."""
    d = min(max(float(dial), 0.0), 1.0)
    return _KNEE_AT_DIAL_0 + d * (_KNEE_AT_DIAL_1 - _KNEE_AT_DIAL_0)


def _knee_compress(v: np.ndarray, knee: float) -> np.ndarray:
    """Soft top-end knee on a scalar field ``v`` in [0, 1]: identity below ``knee``, then
    ``t/(1+t)`` compression asymptoting below 1.0 (monotone)."""
    span = 1.0 - knee
    hi = v > knee
    t = np.where(hi, (v - knee) / span, 0.0)
    comp = knee + span * (t / (1.0 + t))
    return np.where(hi, comp, v)


def _rolloff_luminance(lum: np.ndarray, knee: float) -> np.ndarray:
    """The base/detail highlight roll-off on a luminance field. Compress the smooth base,
    keep detail at full strength, then a gentle top cap. Result <= input everywhere."""
    sigma = float(np.clip(max(lum.shape) * _BASE_SIGMA_FRAC, 2.0, 60.0))
    base = gaussian_filter(lum, sigma, mode="nearest")
    detail = lum - base
    # base_c <= base (compression), so base_c + detail <= base + detail = lum -> never brightens.
    hdr = _knee_compress(base, knee) + detail
    return np.clip(_knee_compress(np.clip(hdr, 0.0, 1.0), _STAR_CAP_KNEE), 0.0, 1.0)


def highlight_rolloff(img, knee: float = _HIGHLIGHT_KNEE) -> np.ndarray:
    """Dial down blown highlights while revealing core structure (see the module docstring).

    ``img`` float64 in [0, 1], mono or RGB. For RGB the luminance is rolled off (base/detail
    + top cap) and R:G:B are scaled together by ``L'/L`` so colour is preserved; mono rolls
    off the single channel directly. Returns a new array; never brightens a pixel.
    """
    a = np.asarray(img, dtype=np.float64)
    if not (0.0 < knee < 1.0):
        return np.clip(a, 0.0, 1.0)

    if not (a.ndim == 3 and a.shape[-1] == 3):        # mono / scalar: value == luminance
        return _rolloff_luminance(a, knee)

    lum = a @ _LUM
    lum_new = _rolloff_luminance(lum, knee)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(lum > 1e-6, lum_new / lum, 1.0)
    out = a * scale[..., None]
    return np.clip(out, 0.0, 1.0)
