"""Frame calibration (ImageCalibration equivalent) — pure numpy.

Master-dark subtraction (bias included, ``optimizeDarks`` off, matching STK.calibrate) and
master-flat division by the flat's own mean. Uses ``ccdproc`` when installed for closer
fidelity, else the direct arithmetic — which is what ImageCalibration does anyway for the
common case.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np


def _noop(_m: str) -> None:
    pass


def _local_median(plane: np.ndarray, size: int = 3) -> np.ndarray:
    from scipy.ndimage import median_filter
    return median_filter(plane, size=size, mode="nearest")


def calibrate_light(light: np.ndarray, *, bias: Optional[np.ndarray] = None,
                    dark: Optional[np.ndarray] = None,
                    flat: Optional[np.ndarray] = None) -> np.ndarray:
    """Calibrate one light: subtract dark (or bias), divide by the normalized flat."""
    out = np.asarray(light, dtype=np.float64)
    if dark is not None:
        out = out - np.asarray(dark, dtype=np.float64)      # dark includes the bias
    elif bias is not None:
        out = out - np.asarray(bias, dtype=np.float64)
    if flat is not None:
        f = np.asarray(flat, dtype=np.float64)
        norm = float(np.mean(f)) or 1.0
        fn = f / norm
        out = np.divide(out, fn, out=np.zeros_like(out), where=fn > 1e-6)
    return np.clip(out, 0.0, 1.0)


def bias_calibrate(frame: np.ndarray, bias: Optional[np.ndarray]) -> np.ndarray:
    """Subtract a master bias from a frame (used to calibrate flats before integrating)."""
    a = np.asarray(frame, dtype=np.float64)
    if bias is not None:
        a = a - np.asarray(bias, dtype=np.float64)
    return np.clip(a, 0.0, 1.0)


def cosmetic_correct(light: np.ndarray, dark: Optional[np.ndarray] = None,
                     sigma: float = 3.0) -> np.ndarray:
    """Repair hot/cold pixels: astroscrappy L.A.Cosmic if present, else a median-outlier map.

    Fallback: replace pixels that deviate from a 3x3 median by > ``sigma``·(robust noise)
    with that median — a light-touch stand-in for CosmeticCorrection (hot/cold only).
    """
    a = np.asarray(light, dtype=np.float64)
    try:
        import astroscrappy
        # Per-channel, not on the luminance mean: a hot pixel lives in one (demosaiced) channel,
        # so detecting on the mean and adding that delta to all three under-corrects the hot
        # channel and wrongly dims the other two (the audit's colour-cast bug).
        if a.ndim == 3:
            chans = [astroscrappy.detect_cosmics(a[..., c], sigclip=sigma)[1]
                     for c in range(a.shape[2])]
            return np.clip(np.stack(chans, axis=-1), 0.0, 1.0)
        return np.clip(astroscrappy.detect_cosmics(a, sigclip=sigma)[1], 0.0, 1.0)
    except Exception:
        pass
    from scipy.ndimage import median_filter
    def _fix(p):
        med = median_filter(p, size=3, mode="nearest")
        resid = p - med
        noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
        bad = np.abs(resid) > sigma * noise
        out = p.copy()
        out[bad] = med[bad]
        return out
    if a.ndim == 3:
        return np.clip(np.stack([_fix(a[..., c]) for c in range(a.shape[2])], axis=-1), 0, 1)
    return np.clip(_fix(a), 0.0, 1.0)


def static_hot_pixel_map(frames: Iterable[np.ndarray], *, sigma: float = 5.0,
                         presence: float = 0.65, log: Callable[[str], None] = _noop) -> np.ndarray:
    """Cross-frame **static** hot/cold pixel map in SENSOR space — the darkless walking-noise fix.

    A defective sensor site is a persistent local outlier at a FIXED sensor coordinate; a real
    star drifts across sensor coordinates frame-to-frame (sky vs. sensor), so it is NOT
    persistent at any one sensor pixel. Undithered hot pixels are exactly what sigma-clip cannot
    reject after registration drags them along the drift into short diagonal dashes (walking
    noise). Here — BEFORE registration — we flag pixels whose per-frame local residual exceeds
    ``sigma``·(robust frame noise) in at least ``presence`` of the frames, then repair them in
    every frame so the defect never enters the registered stack.

    Returns a 2-D boolean sensor mask (True = bad). Detection is on luminance so a demosaiced
    colour blob (a hot CFA site spread by the debayer) is caught as one object.
    """
    hot = cold = None
    n = 0
    for i, f in enumerate(frames):
        a = np.asarray(f, dtype=np.float64)
        lum = a[..., :3].mean(axis=2) if a.ndim == 3 else a
        resid = lum - _local_median(lum, size=3)
        noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) + 1e-9
        if hot is None:
            hot = np.zeros(lum.shape, dtype=np.int32)
            cold = np.zeros(lum.shape, dtype=np.int32)
        hot += (resid > sigma * noise)
        cold += (resid < -sigma * noise)
        n += 1
        log(f"   bad-pixel scan [{i + 1}]")
    if hot is None or n < 3:
        return np.zeros((1, 1), dtype=bool)
    thresh = max(2, int(np.ceil(presence * n)))       # persistent across most frames
    return (hot >= thresh) | (cold >= thresh)


def repair_bad_pixels(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked sensor pixels with the local (3×3) median of the same frame, per channel."""
    a = np.asarray(frame, dtype=np.float64)
    if mask is None or not mask.any() or mask.shape != a.shape[:2]:
        return a
    if a.ndim == 3:
        out = a.copy()
        for c in range(a.shape[2]):
            med = _local_median(a[..., c], size=3)
            out[..., c][mask] = med[mask]
        return out
    out = a.copy()
    out[mask] = _local_median(a, size=3)[mask]
    return out
