"""Reference normalization (ImageIntegration additive+scale, LN-off equivalent).

Before integration, each registered frame is matched to the reference frame's background
level (additive) and signal scale (multiplicative), so frames taken as the sky brightness
and transparency drift across a session stack coherently. Without it, a merely-brighter
frame reads as a per-pixel outlier and sigma-clip rejection + gradients suffer.

This is the global (whole-frame) normalization; a full spatially-varying LocalNormalization
is a heavier future step. Stats are robust (median background, 90th-percentile signal) on
luminance, applied as scalars to every channel so per-frame colour balance is preserved.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

_SCALE_CLIP = (0.5, 2.0)          # guard against blow-ups from degenerate frames
_SIGNAL_Q = 0.90                  # signal-range proxy: (p90 - median)
_LF_TARGET = 256                  # long-edge size for the low-frequency (gradient) model


def frame_stats(img: np.ndarray) -> Tuple[float, float]:
    """Robust (background, signal-range) on the frame's luminance."""
    a = np.asarray(img, dtype=np.float64)
    lum = a[..., :3].mean(axis=2) if a.ndim == 3 else a
    med = float(np.median(lum))
    sig = float(np.quantile(lum, _SIGNAL_Q)) - med
    return med, max(1e-6, sig)


def normalize_to_ref(frame: np.ndarray, ref_med: float, ref_sig: float, *,
                     do_scale: bool = True) -> np.ndarray:
    """Match ``frame`` to the reference: ``(frame - bg)·scale + ref_bg``.

    Returns float (not clipped) so the downstream sigma-clip integration sees the true
    residuals; the final master is clipped on save.
    """
    a = np.asarray(frame, dtype=np.float64)
    med, sig = frame_stats(a)
    scale = float(np.clip(ref_sig / sig, *_SCALE_CLIP)) if do_scale else 1.0
    return (a - med) * scale + ref_med


def _low_frequency(plane: np.ndarray) -> np.ndarray:
    """Smooth low-frequency content of a 2-D plane (downsample → blur → upsample).

    Fast even on 40 MP frames: the gradient/background lives at very low spatial frequency,
    so it's modelled at ~256 px and resized back — stars/nebulosity (high frequency) average
    out and are left untouched.
    """
    H, W = plane.shape
    k = _LF_TARGET / max(H, W)
    if k < 1.0:
        small = zoom(plane, k, order=1)
        small = gaussian_filter(small, sigma=3.0, mode="nearest")
        return zoom(small, (H / small.shape[0], W / small.shape[1]), order=1)
    return gaussian_filter(plane, sigma=max(1.0, max(H, W) / 32.0), mode="nearest")


def local_normalize_to_ref(frame: np.ndarray, ref: np.ndarray, ref_med: float, ref_sig: float,
                           *, do_scale: bool = True) -> np.ndarray:
    """Full LocalNormalization: global background+scale, then a spatially-varying gradient
    match to the reference.

    After the global step (:func:`normalize_to_ref`), the frame's smooth low-frequency
    difference from the reference — the position-dependent gradient / vignetting residual —
    is estimated per channel and subtracted, so every frame's background surface matches the
    reference everywhere (not just on average). High-frequency signal is preserved.
    """
    f = normalize_to_ref(frame, ref_med, ref_sig, do_scale=do_scale)
    r = np.asarray(ref, dtype=np.float64)
    if f.ndim == 3:
        out = f.copy()
        for c in range(f.shape[2]):
            out[..., c] = f[..., c] - _low_frequency(f[..., c] - r[..., c])
        return out
    return f - _low_frequency(f - r)
