"""Depth relief, starless background, and star extraction from a single still.

None of this is a general monocular-depth network (those are trained on
terrestrial scenes and hallucinate on a nebula). It is a set of *astro* cues:

* **Brightness relief** — the bright body of the object is modelled as nearer
  than the faint outskirts, so a dolly-in makes the core billow toward you.
* **Hα / red-excess** — emission is pulled forward (it is what glows).
* **Fine structure** — high-pass filaments add relief so the depth is not a
  smooth dome.
* **Dust** (optional mask) — dark lanes recede.

``starless`` removes point sources with a grey-opening so the depth sheets carry
only the nebula, and ``detect_stars`` returns the stars as an independent point
cloud (position + flux + colour) that the renderer parallaxes on its own — the
single strongest 3D cue.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _luma(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.float64)
    return (rgb[..., 0] * _LUMA[0] + rgb[..., 1] * _LUMA[1]
            + rgb[..., 2] * _LUMA[2])


def _as_rgb(img: np.ndarray) -> np.ndarray:
    a = np.clip(np.nan_to_num(np.asarray(img, dtype=np.float64),
                              nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    return a


def _norm(a: np.ndarray, lo: float = 1.0, hi: float = 99.5) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a, dtype=np.float64))
    p0, p1 = np.percentile(a, [lo, hi])
    if p1 <= p0:
        return np.zeros_like(a)
    return np.clip((a - p0) / (p1 - p0), 0.0, 1.0)


def depth_field(img: np.ndarray, masks: Optional[Dict[str, np.ndarray]] = None,
                *, relief: float = 0.18, smooth: Optional[float] = None
                ) -> np.ndarray:
    """Return a depth relief ``Z`` in ``[0, 1]`` — ``1`` = near, ``0`` = far.

    ``masks`` may be a :func:`lazystretch.develop.semantic.segment` dict; its
    ``Dust`` / ``Hα (red)`` layers refine the relief when present. ``relief``
    weights the fine-structure term; ``smooth`` overrides the base blur radius
    (defaults to a fraction of the frame size).
    """
    from scipy.ndimage import gaussian_filter

    rgb = _as_rgb(img)
    h, w = rgb.shape[:2]
    lum = _luma(rgb)
    s = float(smooth) if smooth is not None else max(h, w) / 70.0
    s = max(s, 1.0)

    base = gaussian_filter(lum, s)                       # broad shape → nearness
    red = np.clip(rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]), 0.0, None)
    red = gaussian_filter(red, s * 0.8)                  # emission pulled forward
    struct = np.clip(gaussian_filter(lum, s * 0.4)
                     - gaussian_filter(lum, s * 1.8), 0.0, None)  # bright filaments

    z = 0.70 * _norm(base) + 0.45 * _norm(red) + relief * _norm(struct)

    if masks:
        dust = masks.get("Dust")
        if dust is not None and np.shape(dust) == lum.shape:
            z = z - 0.35 * np.clip(np.asarray(dust, dtype=np.float64), 0.0, 1.0)
        ha = masks.get("Hα (red)")
        if ha is not None and np.shape(ha) == lum.shape:
            z = z + 0.25 * np.clip(np.asarray(ha, dtype=np.float64), 0.0, 1.0)

    z = gaussian_filter(z, max(1.0, s * 0.9))            # smooth → coherent bulk parallax
    return _norm(z, 0.5, 99.5).astype(np.float32)


def starless(img: np.ndarray, radius: int = 4,
             mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Approximate a starless copy by opening away small bright point sources.

    A grey-opening removes bright features smaller than the structuring element
    (stars) while preserving the larger nebula; the opened result is blended in
    only where stars actually are, so the nebula elsewhere is untouched.
    """
    from scipy.ndimage import grey_opening, gaussian_filter

    rgb = _as_rgb(img)
    size = 2 * int(radius) + 1
    opened = np.stack([grey_opening(rgb[..., c], size=size)
                       for c in range(rgb.shape[2])], axis=-1)

    if mask is None:
        lum = _luma(rgb)
        tophat = np.clip(lum - grey_opening(lum, size=size), 0.0, 1.0)
        m = np.clip(tophat / (np.percentile(tophat, 99.5) + 1e-6), 0.0, 1.0)
        m = gaussian_filter(m, 1.0)
    else:
        m = np.clip(np.asarray(mask, dtype=np.float64), 0.0, 1.0)

    m3 = m[..., None]
    return (rgb * (1.0 - m3) + opened * m3).astype(np.float32)


def detect_stars(img: np.ndarray, max_stars: int = 1200,
                 thresh_sigma: float = 6.0) -> np.ndarray:
    """Return a ``(N, 6)`` array of star records: ``[y, x, flux, r, g, b]``.

    Point sources are isolated with a morphological top-hat (same idea as the
    semantic ``Stars`` mask), thresholded a few noise sigmas above the local
    background, reduced to local maxima, and kept brightest-first.
    """
    from scipy.ndimage import grey_opening, maximum_filter

    rgb = _as_rgb(img)
    lum = _luma(rgb)
    tophat = np.clip(lum - grey_opening(lum, size=7), 0.0, 1.0)
    med = float(np.median(tophat))
    noise = 1.4826 * float(np.median(np.abs(tophat - med))) + 1e-6
    thr = max(thresh_sigma * noise, float(np.percentile(tophat, 99.0)))

    peaks = (tophat >= maximum_filter(tophat, size=5)) & (tophat > thr)
    ys, xs = np.nonzero(peaks)
    if ys.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    flux = tophat[ys, xs]
    order = np.argsort(flux)[::-1][:max_stars]
    ys, xs, flux = ys[order], xs[order], flux[order]
    cols = rgb[ys, xs, :]
    return np.column_stack([ys, xs, flux, cols]).astype(np.float32)
