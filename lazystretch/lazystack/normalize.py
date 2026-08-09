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

_SCALE_CLIP = (0.5, 2.0)          # guard against blow-ups from degenerate frames
_SIGNAL_Q = 0.90                  # signal-range proxy: (p90 - median)


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
