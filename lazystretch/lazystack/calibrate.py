"""Frame calibration (ImageCalibration equivalent) — pure numpy.

Master-dark subtraction (bias included, ``optimizeDarks`` off, matching STK.calibrate) and
master-flat division by the flat's own mean. Uses ``ccdproc`` when installed for closer
fidelity, else the direct arithmetic — which is what ImageCalibration does anyway for the
common case.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


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
        mono = a.mean(axis=2) if a.ndim == 3 else a
        _mask, clean = astroscrappy.detect_cosmics(mono, sigclip=sigma)
        if a.ndim == 3:
            corr = clean - mono
            return np.clip(a + corr[..., None], 0.0, 1.0)
        return np.clip(clean, 0.0, 1.0)
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
