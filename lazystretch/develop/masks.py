"""Mask generators for the Develop window.

Masks are 2-D float arrays in [0, 1] stored on the document by name; any tool can then
be gated by one (the masked blend ``orig + w·mask·(proc − orig)``). Three kinds:

* **Luminosity masks** — extract luminance, normalise, raise to
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

# Set operations for combining named masks into reusable composites.
MASK_OPS = ("intersect", "union", "subtract", "invert")
_OP_SYMBOL = {"intersect": "∩", "union": "∪", "subtract": "−", "invert": "¬"}


def _match_shape(m: np.ndarray, shape_hw) -> np.ndarray:
    """Nearest-resize a 2-D mask to (H, W) — masks in a doc are normally already equal."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if m.shape[0] == h and m.shape[1] == w:
        return m
    ys = np.clip(np.linspace(0, m.shape[0] - 1, h).round().astype(int), 0, m.shape[0] - 1)
    xs = np.clip(np.linspace(0, m.shape[1] - 1, w).round().astype(int), 0, m.shape[1] - 1)
    return m[ys][:, xs]


def combine_masks(a: np.ndarray, b=None, op: str = "intersect") -> np.ndarray:
    """Combine soft masks with fuzzy set algebra (min/max). ``op`` ∈ MASK_OPS.

    intersect = min(A,B) · union = max(A,B) · subtract = min(A, 1−B) (A and NOT B) ·
    invert = 1−A (B ignored). Returns a new float32 [0,1] mask.
    """
    a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    if op == "invert":
        return (1.0 - a).astype(np.float32)
    if b is None:
        return a
    b = _match_shape(np.clip(np.asarray(b, dtype=np.float32), 0.0, 1.0), a.shape[:2])
    if op == "union":
        return np.maximum(a, b)
    if op == "subtract":
        return np.minimum(a, 1.0 - b)
    return np.minimum(a, b)           # intersect (default)


def combined_name(name_a: str, op: str, name_b=None) -> str:
    """A readable default name for a composite, e.g. 'Nebula ∩ ¬Bright stars'."""
    sym = _OP_SYMBOL.get(op, op)
    if op == "invert" or name_b is None:
        return f"¬{name_a}" if op == "invert" else name_a
    if op == "subtract":
        return f"{name_a} ∩ ¬{name_b}"
    return f"{name_a} {sym} {name_b}"


def _luminance(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 2:
        return a
    return a[..., 0] * _LUMA[0] + a[..., 1] * _LUMA[1] + a[..., 2] * _LUMA[2]


def luminosity_mask(img: np.ndarray, kind: str = LUM_LIGHTS, depth: int = 1) -> np.ndarray:
    """Luminosity mask. ``kind`` ∈ {lights, darks}; ``depth`` 1..5 (deeper = more selective)."""
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
