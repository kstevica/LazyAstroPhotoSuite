"""Sigma-clipped integration + master combination (ImageIntegration equivalent).

Pure numpy: an iterative sigma-clip about the per-pixel mean (the port's stand-in for PI's
WinsorizedSigmaClip), with optional per-frame weights (the PSFSignalWeight analogue). Used
both to build calibration masters and to integrate the final registered light stack.

``combine_files`` is the memory-bounded path: it memory-maps float32 ``.npy`` frames and
combines them in row bands, so a big burst never builds an N×H×W cube (holding a whole
30×40 MP stack at once is the OOM/bus-error that the in-RAM path hits on real data).
"""
from __future__ import annotations

from pathlib import Path
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


def combine_files(paths: Sequence["str | Path"], *, weights: Optional[Sequence[float]] = None,
                  sigma_low: float = 4.0, sigma_high: float = 3.0,
                  target_bytes: int = 200_000_000, log=None) -> np.ndarray:
    """Memory-bounded combine: mmap float32 ``.npy`` frames, sigma-clip in row bands.

    Never materialises the full N×H×W cube — only ``N × band × W × C`` at a time — so a large
    burst integrates in a bounded, few-hundred-MB footprint instead of tens of GB. Emits
    per-band progress through ``log`` (integration is otherwise a long silent step).
    """
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("no frames to combine")
    mm = [np.load(p, mmap_mode="r") for p in paths]
    shape = mm[0].shape
    H = shape[0]
    n = len(paths)
    row_bytes = int(np.prod(shape[1:])) * 4 * n            # float32 per output row × N frames
    band = int(max(1, min(H, target_bytes // max(1, row_bytes))))
    n_bands = (H + band - 1) // band
    out = np.empty(shape, dtype=np.float32)
    for bi, y0 in enumerate(range(0, H, band)):
        y1 = min(H, y0 + band)
        cube = np.stack([np.asarray(m[y0:y1], dtype=np.float64) for m in mm], axis=0)
        out[y0:y1] = sigma_clip_mean(cube, sigma_low=sigma_low, sigma_high=sigma_high,
                                     weights=weights).astype(np.float32)
        if log is not None:
            log(f"   integrating rows {y0}-{y1} ({bi + 1}/{n_bands}, "
                f"{100 * y1 // H}%)")
    return out
