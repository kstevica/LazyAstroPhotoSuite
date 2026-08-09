"""Sigma-clipped integration + master combination (ImageIntegration equivalent).

Pure numpy: an iterative sigma-clip about the per-pixel mean (the port's stand-in for PI's
WinsorizedSigmaClip), with optional per-frame weights (the PSFSignalWeight analogue). Used
both to build calibration masters and to integrate the final registered light stack.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def sigma_clip_mean(cube: np.ndarray, *, sigma_low: float = 4.0, sigma_high: float = 3.0,
                    iters: int = 3, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Per-pixel sigma-clipped (weighted) mean over axis 0 of ``cube`` (N, ...)."""
    data = np.asarray(cube, dtype=np.float64)
    n = data.shape[0]
    mask = np.ones(data.shape, dtype=bool)
    for _ in range(max(1, iters)):
        masked = np.where(mask, data, np.nan)
        m = np.nanmean(masked, axis=0)
        s = np.nanstd(masked, axis=0)
        lo = m - sigma_low * s
        hi = m + sigma_high * s
        mask = (data >= lo) & (data <= hi)
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=np.float64)
    wshape = (n,) + (1,) * (data.ndim - 1)
    wc = w.reshape(wshape) * mask
    wsum = wc.sum(axis=0)
    out = np.where(wsum > 0, (data * wc).sum(axis=0) / np.where(wsum > 0, wsum, 1.0),
                   data.mean(axis=0))
    return np.clip(out, 0.0, 1.0)


def combine_master(frames: Sequence[np.ndarray], *, sigma_low: float = 5.0,
                   sigma_high: float = 5.0) -> np.ndarray:
    """Build a calibration master (bias/dark/flat) by sigma-clipped mean."""
    if len(frames) == 0:
        raise ValueError("no frames to combine")
    if len(frames) == 1:
        return np.asarray(frames[0], dtype=np.float64)
    cube = np.stack([np.asarray(f, dtype=np.float64) for f in frames], axis=0)
    return sigma_clip_mean(cube, sigma_low=sigma_low, sigma_high=sigma_high)


def integrate(frames: Sequence[np.ndarray], *, weights: Optional[Sequence[float]] = None,
              sigma_low: float = 4.0, sigma_high: float = 3.0) -> np.ndarray:
    """Integrate the final registered light stack (weighted sigma-clipped mean)."""
    cube = np.stack([np.asarray(f, dtype=np.float64) for f in frames], axis=0)
    return sigma_clip_mean(cube, sigma_low=sigma_low, sigma_high=sigma_high, weights=weights)
