"""SNR-protect mask — turn LazyStack's measured per-pixel noise map into a stretch mask.

The stacker writes a per-pixel **standard-error** map beside the master (``sigma_clip_mean``'s
``std(survivors)/√N``). Combined with the master, that gives a true, measured **SNR** per pixel.

Why this matters: a *luminance* mask cannot tell faint real signal from noise — both are dim. An
*SNR* mask can, because real signal is consistent frame-to-frame (high SNR) while noise is not.
So this mask lets the stretch hold the floor down on **pure-noise** pixels and back off the
noise-amplifying steps (local contrast / sharpening) there, while still lifting faint **real**
nebulosity — something the luminance masks structurally cannot do. (Reviewer 1's #1 ask, driven
by measured data instead of a single-image guess.)

This module is the *builder* — it does not yet modulate the pipeline; the stretch wires it in
once a re-stack confirms a companion is present. Falls back to ``None`` when no map exists
(imported masters), so callers degrade to today's single-image noise handling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

# Companion filenames written by lazystack.run (kept in sync with run.py).
_COMPANION_NAMES = ("lazystack_master_noise.npy",)
_COVERAGE_NAMES = ("lazystack_master_coverage.npy",)


def load_noise_map(master_path: "str | Path") -> Optional[np.ndarray]:
    """Find + load the per-pixel noise map written beside ``master_path`` (or ``None``)."""
    p = Path(master_path)
    candidates = [p.with_name(n) for n in _COMPANION_NAMES]
    candidates.append(p.with_name(p.stem + "_noise.npy"))
    for c in candidates:
        if c.exists():
            try:
                return np.load(str(c))
            except Exception:
                pass
    return None


def load_coverage_map(master_path: "str | Path") -> Optional[np.ndarray]:
    """Find + load the per-pixel frame-support (coverage) map beside ``master_path`` (or ``None``)."""
    p = Path(master_path)
    candidates = [p.with_name(n) for n in _COVERAGE_NAMES]
    candidates.append(p.with_name(p.stem + "_coverage.npy"))
    for c in candidates:
        if c.exists():
            try:
                return np.load(str(c))
            except Exception:
                pass
    return None


def snr_map(master: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Per-pixel SNR = luminance(master) / (noise + eps)."""
    a = np.asarray(master, dtype=np.float64)
    lum = a[..., :3].mean(axis=2) if a.ndim == 3 else a
    nz = np.asarray(noise, dtype=np.float64)
    if nz.shape != lum.shape:
        raise ValueError(f"noise map shape {nz.shape} != image {lum.shape}")
    return np.maximum(lum, 0.0) / (nz + 1e-6)


def snr_protect_mask(master: np.ndarray, noise: np.ndarray, strength: float = 0.5, *,
                     coverage: Optional[np.ndarray] = None,
                     lo_pct: float = 20.0, hi_pct: float = 70.0, smooth: float = 8.0) -> np.ndarray:
    """A ``0..strength`` mask: **high** where confidence is low, **0** where it is high.

    Confidence combines two stack-measured signals: **SNR** (self-scaling — full protection at/below
    the ``lo_pct`` SNR percentile, none at/above ``hi_pct``) and, when given, **frame support**
    (``coverage``) — pixels covered by fewer frames are less reliable and get protected more. The
    two are combined by taking the stronger protection. Smoothed to a low-frequency mask (the
    per-pixel estimates from ~tens of frames are themselves noisy). ``strength`` is the user's 0..1
    "ponder". Returns a float64 array in ``[0, strength]``.
    """
    snr = snr_map(master, noise)
    lo = float(np.nanpercentile(snr, lo_pct))
    hi = float(np.nanpercentile(snr, hi_pct))
    protect = np.clip((hi - snr) / (hi - lo + 1e-9), 0.0, 1.0)     # 1 at/below lo, 0 at/above hi
    if coverage is not None:
        cov = np.asarray(coverage, dtype=np.float64)
        if cov.shape == protect.shape:
            cmax = float(cov.max()) or 1.0
            protect = np.maximum(protect, np.clip(1.0 - cov / cmax, 0.0, 1.0))  # low support -> protect
    if smooth and smooth > 0:
        protect = gaussian_filter(protect, float(smooth))
    return float(np.clip(strength, 0.0, 1.0)) * np.clip(protect, 0.0, 1.0)
