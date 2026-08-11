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


def _same_shape(name: str, cal: np.ndarray, ref_shape: tuple) -> np.ndarray:
    """Return ``cal`` as float64, or raise a clear error if it doesn't match the light's shape."""
    a = np.asarray(cal, dtype=np.float64)
    if a.shape != ref_shape:
        raise ValueError(
            f"master {name} shape {a.shape} does not match the light frame {ref_shape} — the set "
            f"has inconsistent dimensions/orientation (commonly a mix of portrait/landscape EXIF "
            f"tags across lights vs. darks/flats). Use one camera in one orientation for the whole set."
        )
    return a


def calibrate_light(light: np.ndarray, *, bias: Optional[np.ndarray] = None,
                    dark: Optional[np.ndarray] = None,
                    flat: Optional[np.ndarray] = None) -> np.ndarray:
    """Calibrate one light: subtract dark (or bias), divide by the normalized flat."""
    out = np.asarray(light, dtype=np.float64)
    if dark is not None:
        out = out - _same_shape("dark", dark, out.shape)    # dark includes the bias
    elif bias is not None:
        out = out - _same_shape("bias", bias, out.shape)
    if flat is not None:
        f = _same_shape("flat", flat, out.shape)
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


def _protect_extended(original: np.ndarray, cleaned: np.ndarray, max_component: int) -> np.ndarray:
    """Only accept SMALL, compact cosmetic repairs — restore original over large/elongated changes.

    Hot pixels and cosmic rays are a few pixels; a **meteor / satellite trail** is a large
    connected feature. Repairing it would erase exactly what meteor-preservation wants to keep. So
    any run of changed pixels bigger than ``max_component`` is reverted to the original.
    """
    from scipy.ndimage import label
    a = np.asarray(original, dtype=np.float64)
    c = np.asarray(cleaned, dtype=np.float64)
    changed = np.any(np.abs(c - a) > 1e-6, axis=2) if a.ndim == 3 else (np.abs(c - a) > 1e-6)
    lbl, n = label(changed, structure=np.ones((3, 3)))
    if n == 0:
        return c
    sizes = np.bincount(lbl.ravel())
    small = np.zeros(sizes.size, dtype=bool)
    small[1:] = sizes[1:] <= max_component          # label 0 is unchanged background
    keep = small[lbl]                                # accept the small repairs only
    m = keep[..., None] if a.ndim == 3 else keep
    return np.where(m, c, a)


def cosmetic_correct(light: np.ndarray, dark: Optional[np.ndarray] = None,
                     sigma: float = 3.0, max_component: int = 12) -> np.ndarray:
    """Repair hot/cold pixels: astroscrappy L.A.Cosmic if present, else a median-outlier map.

    Fallback: replace pixels that deviate from a 3x3 median by > ``sigma``·(robust noise) with that
    median. Either way, only SMALL compact repairs are accepted (``_protect_extended``) so meteor /
    satellite trails survive to integration (where meteor-preservation catches them).
    """
    a = np.asarray(light, dtype=np.float64)
    cleaned = None
    try:
        import astroscrappy
        # Per-channel, not on the luminance mean: a hot pixel lives in one (demosaiced) channel,
        # so detecting on the mean and adding that delta to all three under-corrects the hot
        # channel and wrongly dims the other two (the audit's colour-cast bug).
        if a.ndim == 3:
            chans = [astroscrappy.detect_cosmics(a[..., c], sigclip=sigma)[1]
                     for c in range(a.shape[2])]
            cleaned = np.stack(chans, axis=-1)
        else:
            cleaned = astroscrappy.detect_cosmics(a, sigclip=sigma)[1]
    except Exception:
        from scipy.ndimage import median_filter
        def _fix(p):
            med = median_filter(p, size=3, mode="nearest")
            resid = p - med
            noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
            bad = np.abs(resid) > sigma * noise
            out = p.copy()
            out[bad] = med[bad]
            return out
        cleaned = (np.stack([_fix(a[..., c]) for c in range(a.shape[2])], axis=-1)
                   if a.ndim == 3 else _fix(a))
    return np.clip(_protect_extended(a, cleaned, max_component), 0.0, 1.0)


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


def suppress_banding(img: np.ndarray, *, strength: float = 1.0, smooth: int = 25,
                     do_columns: bool = True, do_rows: bool = True) -> np.ndarray:
    """Remove column/row fixed-pattern banding from the integrated master.

    Sensor column/row FPN (unavoidable with no darks) survives integration as a coherent
    per-column/per-row bias — the subtle vertical banding a stretch lifts into the shadows. For
    each line-direction: take the per-column (per-row) MEDIAN profile (robust to stars), keep only
    its HIGH-frequency part (profile minus a smooth version), and subtract that. Only the narrow
    column-to-column jumps go; real large-scale vertical/horizontal structure (the smooth part) is
    preserved. Self-limiting — if there's no coherent pattern the high-pass profile is ~flat and
    nothing is removed. Per channel on colour.
    """
    from scipy.ndimage import uniform_filter1d
    a = np.asarray(img, dtype=np.float64)
    win = max(3, int(smooth))
    s = float(np.clip(strength, 0.0, 1.0))

    def _correct(plane: np.ndarray) -> np.ndarray:
        out = plane
        if do_columns:
            col = np.median(out, axis=0)                      # one value per column
            out = out - s * (col - uniform_filter1d(col, win, mode="nearest"))[None, :]
        if do_rows:
            rowp = np.median(out, axis=1)
            out = out - s * (rowp - uniform_filter1d(rowp, win, mode="nearest"))[:, None]
        return out

    if a.ndim == 3:
        return np.clip(np.stack([_correct(a[..., c]) for c in range(a.shape[2])], axis=-1), 0.0, 1.0)
    return np.clip(_correct(a), 0.0, 1.0)


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
