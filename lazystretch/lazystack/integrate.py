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
                    iters: int = 3, weights: Optional[Sequence[float]] = None,
                    return_coverage: bool = False, return_noise: bool = False,
                    return_transient: bool = False):
    """Per-pixel sigma-clipped (weighted) mean over axis 0 of ``cube`` (N, ...).

    NaN samples are treated as **no-data** (registration marks the non-overlap border of an
    aligned frame with NaN, not 0): they are excluded from the clip statistics and from the
    divisor, so a partial-coverage pixel is the mean of the frames that actually cover it —
    never divided-by-N against silent zeros (the old bright/dark edge-strip bug).

    Optional extra maps (returned in this order after the mean): with ``return_coverage`` the
    per-pixel count of frames that had real data (drives the overlap crop); with ``return_noise``
    a per-pixel **standard error of the mean** (``std(survivors)/√N`` on luminance) — a directly
    measured noise map that is a far better SNR estimate than post-hoc single-image guesses.
    """
    data = np.asarray(cube, dtype=np.float64)
    n = data.shape[0]
    valid = np.isfinite(data)                      # no-data (NaN) is excluded everywhere
    mask = valid.copy()
    for _ in range(max(1, iters)):
        masked = np.where(mask, data, np.nan)
        with np.errstate(invalid="ignore"):
            # Robust center/scale (median + MAD), not mean/std: a hot value present in a
            # minority-to-slight-majority of frames (undithered walking noise dragged along the
            # drift) stays inside mean±kσ but not median±k·MAD — the audit's ineffective-rejection fix.
            m = np.nanmedian(masked, axis=0)
            mad = np.nanmedian(np.abs(masked - m), axis=0)
            s = np.maximum(1.4826 * mad, 1e-4)            # tiny floor: still clips a lone spike
            lo = m - sigma_low * s                        #             among near-identical inliers
            hi = m + sigma_high * s
            mask = valid & (data >= lo) & (data <= hi)
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=np.float64)
    wshape = (n,) + (1,) * (data.ndim - 1)
    wc = w.reshape(wshape) * mask
    wsum = wc.sum(axis=0)
    safe = np.where(mask, data, 0.0)               # NaN·0 would poison the sum
    with np.errstate(invalid="ignore"):
        fallback = np.nanmean(np.where(valid, data, np.nan), axis=0)   # NaN only if 0 coverage
    weighted = (safe * wc).sum(axis=0) / np.where(wsum > 0, wsum, 1.0)
    out = np.clip(np.where(wsum > 0, weighted, fallback), 0.0, 1.0)    # NaN survives clip
    if not (return_coverage or return_noise or return_transient):
        return out
    result = [out]
    if return_coverage:
        per_px = valid.all(axis=-1) if data.ndim == 4 else valid       # collapse colour
        result.append(per_px.sum(axis=0).astype(np.int32))
    if return_noise:
        if data.ndim == 4:
            lum_cube = data.mean(axis=-1)                              # (N, H, W) luminance
            surv_px = mask.all(axis=-1)                                # survivor in every channel
        else:
            lum_cube, surv_px = data, mask
        with np.errstate(invalid="ignore"):
            sd = np.nanstd(np.where(surv_px, lum_cube, np.nan), axis=0)
        cnt = surv_px.sum(axis=0)
        sem = sd / np.sqrt(np.maximum(cnt, 1))                         # standard error of the mean
        result.append(np.where(cnt > 0, sem, np.nan).astype(np.float32))
    if return_transient:
        # Meteor preservation: the brightest HIGH-SIDE REJECTED sample per pixel is the transient
        # (meteor/plane/satellite) light the clip is discarding. Keep its linear additive
        # contribution (frame − master) in RGB + which frame it came from (single-frame == meteor).
        with np.errstate(invalid="ignore"):
            hi_rej = valid & (~mask) & (data > out)                    # high-side rejected only
        excess = np.where(hi_rej, np.nan_to_num(data - out, nan=0.0), 0.0)
        contrib = np.max(excess, axis=0)                              # (H,W[,C]) brightest transient
        lum_ex = excess.mean(axis=-1) if data.ndim == 4 else excess    # (N,H,W)
        fidx = np.argmax(lum_ex, axis=0).astype(np.int16)             # source frame per pixel
        fidx = np.where(lum_ex.max(axis=0) > 0, fidx, -1).astype(np.int16)   # -1 == no transient
        result.append(np.clip(contrib, 0.0, 1.0).astype(np.float32))
        result.append(fidx)
    return tuple(result)


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
              sigma_low: float = 4.0, sigma_high: float = 3.0,
              return_coverage: bool = False, return_noise: bool = False,
              return_transient: bool = False):
    """Integrate the final registered light stack (weighted sigma-clipped mean).

    Optional maps follow the master in order: ``return_coverage`` (covered-frame count),
    ``return_noise`` (standard-error map), ``return_transient`` (meteor contribution + source frame).
    """
    cube = np.stack([np.asarray(f, dtype=np.float64) for f in frames], axis=0)
    return sigma_clip_mean(cube, sigma_low=sigma_low, sigma_high=sigma_high, weights=weights,
                           return_coverage=return_coverage, return_noise=return_noise,
                           return_transient=return_transient)


def combine_files(paths: Sequence["str | Path"], *, weights: Optional[Sequence[float]] = None,
                  sigma_low: float = 4.0, sigma_high: float = 3.0,
                  target_bytes: int = 200_000_000, log=None,
                  return_coverage: bool = False, return_noise: bool = False,
                  return_transient: bool = False):
    """Memory-bounded combine: mmap float32 ``.npy`` frames, sigma-clip in row bands.

    Never materialises the full N×H×W cube — only ``N × band × W × C`` at a time — so a large
    burst integrates in a bounded, few-hundred-MB footprint instead of tens of GB. Emits
    per-band progress through ``log`` (integration is otherwise a long silent step). Optional
    per-pixel maps, accumulated across bands, follow the master: ``return_coverage`` (covered-frame
    count, for the overlap crop) then ``return_noise`` (standard-error map, for the SNR mask).
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
    coverage = np.empty((H, shape[1]), dtype=np.int32) if return_coverage else None
    noise = np.empty((H, shape[1]), dtype=np.float32) if return_noise else None
    transient = np.empty(shape, dtype=np.float32) if return_transient else None
    tframe = np.empty((H, shape[1]), dtype=np.int16) if return_transient else None
    for bi, y0 in enumerate(range(0, H, band)):
        y1 = min(H, y0 + band)
        cube = np.stack([np.asarray(m[y0:y1], dtype=np.float64) for m in mm], axis=0)
        res = sigma_clip_mean(cube, sigma_low=sigma_low, sigma_high=sigma_high,
                              weights=weights, return_coverage=return_coverage,
                              return_noise=return_noise, return_transient=return_transient)
        if return_coverage or return_noise or return_transient:
            band_out = res[0]
            idx = 1
            if return_coverage:
                coverage[y0:y1] = res[idx]; idx += 1
            if return_noise:
                noise[y0:y1] = res[idx]; idx += 1
            if return_transient:
                transient[y0:y1] = res[idx]; idx += 1
                tframe[y0:y1] = res[idx]; idx += 1
        else:
            band_out = res
        out[y0:y1] = band_out.astype(np.float32)
        if log is not None:
            log(f"   integrating rows {y0}-{y1} ({bi + 1}/{n_bands}, "
                f"{100 * y1 // H}%)")
    if not (return_coverage or return_noise or return_transient):
        return out
    result = [out]
    if return_coverage:
        result.append(coverage)
    if return_noise:
        result.append(noise)
    if return_transient:
        result.append(transient)
        result.append(tframe)
    return tuple(result)
