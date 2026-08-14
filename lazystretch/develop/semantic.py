"""Semantic auto-masks — derive intent-level masks from the image itself.

``segment(img, snr=None)`` runs one analysis pass and returns a dict of named soft
masks that carry the *meaning* of the frame, so every Develop tool's gate becomes
smarter for free (Selective Color on Hα without touching star colour; De-veil that
avoids faint dust; saturation that protects stellar cores; …):

    Sky · Stars · Bright stars · Faint stars · Nebulosity · Faint nebulosity ·
    Dust · Cores · Hα (red)

This is the "carry knowledge forward" idea (cf. stack SNR-protect, meteor metadata)
applied inside Develop. Stars are isolated with a morphological top-hat (small bright
features), nebulosity is star-subtracted signal above the background, dust is local
darkening inside the subject, cores are near-clipped highlights, Hα is red excess. If a
LazyStack per-pixel SNR map is supplied it splits confident vs faint nebulosity;
otherwise brightness above background is used. Every mask is float32 in [0, 1].
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _luma(a3: np.ndarray) -> np.ndarray:
    return a3[..., 0] * _LUMA[0] + a3[..., 1] * _LUMA[1] + a3[..., 2] * _LUMA[2]


def _smoothstep(e0, e1, x):
    if e1 <= e0:
        return (np.asarray(x) >= e1).astype(np.float64)
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def segment(img: np.ndarray, snr: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Return semantic soft masks for ``img`` (float [0,1], mono or RGB)."""
    from scipy.ndimage import grey_opening, gaussian_filter, maximum_filter

    a = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    is_rgb = a.ndim == 3 and a.shape[2] == 3
    a3 = a if is_rgb else np.repeat(np.atleast_3d(a), 3, axis=2)
    L = _luma(a3)
    out: Dict[str, np.ndarray] = {}

    bg = float(np.median(L))
    noise = float(1.4826 * np.median(np.abs(L - gaussian_filter(L, 1.0)))) + 1e-6

    # --- stars: morphological top-hat isolates small bright features -----------
    tophat = np.clip(L - grey_opening(L, size=7), 0.0, 1.0)
    star = gaussian_filter(_smoothstep(4 * noise, 10 * noise, tophat), 1.0)
    star = np.clip(star, 0.0, 1.0)
    out["Stars"] = star.astype(np.float32)
    peak = maximum_filter(L, size=5)
    bright = _smoothstep(0.35, 0.75, peak)                 # bright vs faint by local peak
    out["Bright stars"] = np.clip(star * bright, 0.0, 1.0).astype(np.float32)
    out["Faint stars"] = np.clip(star * (1.0 - bright), 0.0, 1.0).astype(np.float32)

    # --- nebulosity: star-subtracted signal above the background ---------------
    signal = np.clip((L - bg) / max(1.0 - bg, 1e-6), 0.0, 1.0)
    neb_s = gaussian_filter(np.clip(signal * (1.0 - star), 0.0, 1.0), 2.0)
    neb = _smoothstep(0.015, 0.12, neb_s)
    out["Nebulosity"] = neb.astype(np.float32)
    if snr is not None and getattr(snr, "shape", None) == L.shape:
        s = np.clip(np.asarray(snr, dtype=np.float64), 0.0, None)
        s = s / (np.percentile(s, 99) + 1e-6)
        confident = _smoothstep(0.2, 0.6, s)
    else:
        confident = _smoothstep(0.10, 0.30, neb_s)
    out["Faint nebulosity"] = np.clip(neb * (1.0 - confident), 0.0, 1.0).astype(np.float32)

    # --- sky / background ------------------------------------------------------
    sky = np.clip((1.0 - _smoothstep(0.015, 0.10, neb_s)) * (1.0 - star), 0.0, 1.0)
    out["Sky"] = sky.astype(np.float32)

    # --- dust: local darkening inside the subject (not sky, not stars) ---------
    darkness = np.clip(gaussian_filter(L, 12.0) - L, 0.0, 1.0)
    dust = _smoothstep(3 * noise, 0.08, darkness) * (1.0 - sky) * (1.0 - star)
    out["Dust"] = np.clip(gaussian_filter(dust, 1.0), 0.0, 1.0).astype(np.float32)

    # --- saturated cores -------------------------------------------------------
    out["Cores"] = _smoothstep(0.90, 0.99, L).astype(np.float32)

    # --- Hα (red nebulosity) ---------------------------------------------------
    if is_rgb:
        red_excess = np.clip(a3[..., 0] - np.maximum(a3[..., 1], a3[..., 2]), 0.0, 1.0)
        out["Hα (red)"] = np.clip(neb * _smoothstep(0.02, 0.12, red_excess),
                                  0.0, 1.0).astype(np.float32)
    return out
