"""Mask generators for the Develop window.

Masks are 2-D float arrays in [0, 1] stored on the document by name; any tool can then
be gated by one (the Lighthouse blend ``orig + w·mask·(proc − orig)``). Three kinds:

* **Luminosity masks** — the Lighthouse model: extract luminance, normalise, raise to
  ``2**(depth-1)`` for Lights or the complement for Darks (deeper = more selective).
* **Range masks** — bright/dark/mid selection by lightness with soft edges (reuses
  ``processes.masks.build_range_mask``).
* **Painted masks** — a hand-painted 0/1 map from the canvas brush (built by the GUI).
"""
from __future__ import annotations

import numpy as np

from ..processes.masks import build_range_mask, build_highlights_mask

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# Human-facing kinds for luminosity masks.
LUM_LIGHTS = "lights"
LUM_DARKS = "darks"


def _luminance(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 2:
        return a
    return a[..., 0] * _LUMA[0] + a[..., 1] * _LUMA[1] + a[..., 2] * _LUMA[2]


def luminosity_mask(img: np.ndarray, kind: str = LUM_LIGHTS, depth: int = 1) -> np.ndarray:
    """Lighthouse luminosity mask. ``kind`` ∈ {lights, darks}; ``depth`` 1..5."""
    depth = int(np.clip(round(depth), 1, 5))
    power = 2 ** (depth - 1)
    L = _luminance(img)
    m = L.max()
    L = L / (m if m > 1e-6 else 1e-6)
    L = np.clip(L, 0.0, 1.0)
    base = (1.0 - L) if kind == LUM_DARKS else L
    return np.clip(base ** power, 0.0, 1.0).astype(np.float32)


def range_mask(img: np.ndarray, low: float = 0.0, high: float = 1.0,
               fuzz: float = 0.15, smooth: int = 12) -> np.ndarray:
    """Soft lightness-range selection (reuses the pipeline's range-mask builder)."""
    return build_range_mask(img, float(low), float(high), float(fuzz), int(smooth)).astype(np.float32)


def highlights_mask(img: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """Select the brightest pixels (stars / cores)."""
    return build_highlights_mask(img, float(threshold)).astype(np.float32)


def scribble_to_mask(scribbles: np.ndarray, *, positive: str = "sky") -> np.ndarray:
    """Turn the canvas brush map (+sky / −earth, |v| = strength) into a [0,1] mask.

    ``positive='sky'`` selects the painted-sky (>0) region; ``'earth'`` selects <0.
    Unpainted pixels are 0.5 (neutral) so a half-painted mask still reads sensibly.
    """
    s = np.asarray(scribbles, dtype=np.float32)
    if positive == "earth":
        s = -s
    return np.clip(0.5 + 0.5 * np.clip(s, -1.0, 1.0), 0.0, 1.0)
