"""Range/highlights masks + the portable masked step variants (LazyStretch.js:2504-2799).

Mask blending mirrors PI's ``setMask`` semantics as used in the source:
  * non-inverted mask -> process is applied where the mask is WHITE:
        out = orig*(1-mask) + processed*mask
  * inverted mask     -> process is applied where the mask is BLACK:
        out = orig*mask + processed*(1-mask)

``build_range_mask`` approximates ``RangeSelection`` (toLightness, fuzziness, smoothness) —
the exact falloff curves are undocumented PI internals, so this is a calibrated
approximation (soft ramps + a Gaussian smooth), flagged for tuning against PI output.

The star-protected saturation variant is NOT here: it needs ``buildStarMask`` (StarX, a
wall tool) and lives in ``external`` / falls back to global saturation.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .backgroundlevel import background_level
from .multiscale import noise_reduction_mmt
from .tone import local_contrast


def _lightness(a: np.ndarray) -> np.ndarray:
    """2-D lightness proxy (RangeSelection toLightness ~ CIE L*; approximated by the
    channel mean for RGB, the value for mono)."""
    a = np.asarray(a, dtype=np.float64)
    return a if a.ndim == 2 else a.mean(axis=2)


def _channel_avg_median(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 2:
        return float(np.median(a))
    return float(np.mean([np.median(a[..., c]) for c in range(a.shape[2])]))


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (x >= edge1).astype(np.float64)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_range_mask(img, low: float, high: float, fuzz: float = 0.15,
                     smooth: float = 12) -> np.ndarray:
    """White where lightness is within ``[low, high]``; soft edges + Gaussian smooth.

    Approximates ``LS.buildRangeMask`` (js:2508-2531). Returns a 2-D mask in [0,1].
    """
    L = _lightness(img)
    low = min(max(low, 0.0), 1.0)
    high = min(max(high, 0.0), 1.0)
    fuzz = max(fuzz, 0.0)
    if fuzz > 0.0:
        lo_edge = _smoothstep(low - fuzz, low, L)          # rises to 1 at `low`
        hi_edge = 1.0 - _smoothstep(high, high + fuzz, L)  # falls from 1 at `high`
        mask = lo_edge * hi_edge
    else:
        mask = ((L >= low) & (L <= high)).astype(np.float64)
    if smooth and smooth > 0:
        mask = gaussian_filter(mask, sigma=max(smooth / 3.0, 0.5))
    return np.clip(mask, 0.0, 1.0)


def build_highlights_mask(img, threshold: float) -> np.ndarray:
    """White on the brightest pixels (js:2770-2776) — softer falloff (fuzz 0.35/smooth 24)."""
    return build_range_mask(img, min(max(threshold, 0.0), 1.0), 1.0, 0.35, 24)


def apply_masked(orig: np.ndarray, processed: np.ndarray, mask2d: np.ndarray,
                 invert: bool = False) -> np.ndarray:
    """Composite ``processed`` over ``orig`` through a 2-D mask (broadcast over channels).

    ``invert=False`` -> process where the mask is WHITE; ``invert=True`` -> where BLACK.
    """
    o = np.asarray(orig, dtype=np.float64)
    p = np.asarray(processed, dtype=np.float64)
    m = mask2d if o.ndim == 2 else mask2d[..., None]
    if invert:
        return o * m + p * (1.0 - m)     # process where mask is BLACK
    return o * (1.0 - m) + p * m         # process where mask is WHITE


def background_masked(orig: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """Blend a fully-processed image in only over the sky (< channel-avg median)."""
    a = np.asarray(orig, dtype=np.float64)
    mask = build_range_mask(a, 0.0, _channel_avg_median(a))   # white on the sky background
    return apply_masked(a, processed, mask, invert=False)


def background_level_masked(img, target_level: float) -> np.ndarray:
    """Lower the background THROUGH a background mask — only the sky darkens (js:2726-2746)."""
    a = np.asarray(img, dtype=np.float64)
    return background_masked(a, background_level(a, target_level))


def local_contrast_masked(img, amount: float) -> np.ndarray:
    """Local contrast through an INVERTED highlights mask — protect bright cores (js:2781-2799)."""
    a = np.asarray(img, dtype=np.float64)
    mask = build_highlights_mask(a, 0.75)
    return apply_masked(a, local_contrast(a, amount), mask, invert=True)  # process where NOT highlights


def noise_reduction_masked(img) -> np.ndarray:
    """Denoise only the sky through a background mask (js:2751-2765); NR = MMT fallback."""
    a = np.asarray(img, dtype=np.float64)
    return background_masked(a, noise_reduction_mmt(a))
