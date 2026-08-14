"""Curve evaluation for the Develop Curves tool.

A curve is a list of control points ``[[x, y], ...]`` in [0, 1], monotone in x. We
build a 1024-entry lookup table with a shape-preserving monotone cubic (PCHIP) so the
curve never overshoots between points (the classic S-curve artefact), then map pixels
through it. Channel modes mirror Lightroom / PixInsight CurvesTransformation.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

_LUT_N = 1024


def _clean_points(points: Sequence) -> np.ndarray:
    """Sort/dedup control points and clamp to [0,1]; guarantee the two endpoints."""
    pts = [(float(x), float(y)) for x, y in points] if points else []
    pts = [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for x, y in pts]
    pts.sort(key=lambda t: t[0])
    # Drop points with duplicate x (keep the first), keep it strictly increasing.
    out: List = []
    for x, y in pts:
        if out and x <= out[-1][0] + 1e-6:
            continue
        out.append((x, y))
    if not out or out[0][0] > 1e-6:
        out.insert(0, (0.0, out[0][1] if out else 0.0))
    if out[-1][0] < 1.0 - 1e-6:
        out.append((1.0, out[-1][1]))
    return np.array(out, dtype=np.float64)


def curve_lut(points: Sequence, n: int = _LUT_N) -> np.ndarray:
    """Build a length-``n`` lookup table (float32 [0,1]) for a control-point curve."""
    pts = _clean_points(points)
    xs = np.linspace(0.0, 1.0, n)
    if len(pts) <= 2:
        lut = np.interp(xs, pts[:, 0], pts[:, 1])
    else:
        try:
            from scipy.interpolate import PchipInterpolator
            lut = PchipInterpolator(pts[:, 0], pts[:, 1])(xs)
        except Exception:
            lut = np.interp(xs, pts[:, 0], pts[:, 1])
    return np.clip(lut, 0.0, 1.0).astype(np.float32)


def _map_lut(chan: np.ndarray, lut: np.ndarray) -> np.ndarray:
    idx = np.clip(chan, 0.0, 1.0) * (lut.shape[0] - 1)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, lut.shape[0] - 1)
    f = (idx - i0).astype(np.float32)
    return lut[i0] * (1.0 - f) + lut[i1] * f


def _luma(img: np.ndarray) -> np.ndarray:
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])


def apply_curve(img: np.ndarray, points: Sequence, channel: str = "rgb") -> np.ndarray:
    """Apply a control-point curve. ``channel`` ∈ {rgb, lum, r, g, b}."""
    lut = curve_lut(points)
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        return _map_lut(a, lut)
    channel = (channel or "rgb").lower()
    if channel == "rgb":
        return np.stack([_map_lut(a[..., c], lut) for c in range(3)], axis=-1)
    if channel in ("r", "g", "b"):
        c = {"r": 0, "g": 1, "b": 2}[channel]
        out = a.copy()
        out[..., c] = _map_lut(a[..., c], lut)
        return out
    # luminance: map L, scale RGB by L'/L to preserve hue/saturation.
    L = _luma(a)
    Lp = _map_lut(L, lut)
    scale = np.where(L > 1e-6, Lp / np.maximum(L, 1e-6), 1.0).astype(np.float32)
    return a * scale[..., None]
