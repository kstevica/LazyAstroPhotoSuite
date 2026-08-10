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
    """Robust (background, signal-range) on the frame's luminance.

    NaN (registration no-data border) is ignored, so a frame's background level is measured
    over its *covered* pixels only — the fix for the normalization over-brightening large-border
    frames into visible edge strips.
    """
    a = np.asarray(img, dtype=np.float64)
    lum = a[..., :3].mean(axis=2) if a.ndim == 3 else a
    if not np.any(np.isfinite(lum)):
        return 0.0, 1.0
    med = float(np.nanmedian(lum))
    sig = float(np.nanquantile(lum, _SIGNAL_Q)) - med
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
    """Robust smooth low-frequency content of a 2-D plane (block-MEDIAN → blur → upsample).

    The gradient/background lives at very low spatial frequency, so it's modelled at ~256 px
    and resized back. Downsampling uses the per-block MEDIAN, not averaging/bilinear, so
    bright stars (and registration-residual spikes) are rejected as outliers rather than
    smeared into blobs — averaging them in was what punched dark/coloured holes at stars.
    """
    H, W = plane.shape
    bh = max(1, H // _LF_TARGET)
    bw = max(1, W // _LF_TARGET)
    if bh > 1 or bw > 1:
        Hc, Wc = (H // bh) * bh, (W // bw) * bw
        blocks = plane[:Hc, :Wc].reshape(Hc // bh, bh, Wc // bw, bw)
        with np.errstate(invalid="ignore"):
            small = np.nanmedian(blocks, axis=(1, 3))    # star- and no-data-robust downsample
    else:
        small = plane
    if not np.all(np.isfinite(small)):                   # fill no-data blocks so the blur/zoom stay finite
        fill = np.nanmedian(small) if np.any(np.isfinite(small)) else 0.0
        small = np.where(np.isfinite(small), small, fill)
    small = gaussian_filter(small, sigma=2.0, mode="nearest")
    if small.shape == plane.shape:
        return small
    return zoom(small, (H / small.shape[0], W / small.shape[1]), order=1)


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
