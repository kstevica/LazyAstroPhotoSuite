"""Diffraction spikes for stars.

Two pure pieces the GUI drives:

* :func:`detect_stars` — find the brightest point sources (morphological top-hat +
  local maxima) and return them as normalised records the editor can mark.
* :func:`render_spikes` — composite N diffraction spikes onto each star. Spikes are
  additive light: a thin bright core that tapers from the star to a per-star length,
  tinted by the star's own colour (or white). Number of spikes (3-32), base rotation,
  thickness and intensity are global; length is per star (each record carries ``len``).

A "star" is a dict: ``{"x", "y"}`` in [0,1] frame coords, ``"len"`` the spike length as a
fraction of the image diagonal, plus ``"flux"`` and ``"col"`` (RGB) captured at detection.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _as_rgb(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    return a if a.ndim == 3 else np.repeat(a[..., None], 3, axis=2)


def detect_stars(img: np.ndarray, *, max_stars: int = 40, thresh_sigma: float = 6.0,
                 min_sep_frac: float = 0.01) -> List[Dict]:
    """Return up to ``max_stars`` bright stars, brightest first, as normalised records.

    Point sources are isolated with a top-hat, thresholded a few noise sigmas over the
    local background, reduced to local maxima and kept brightest-first; near-duplicates
    within ``min_sep_frac`` of the diagonal are merged so one star yields one marker.
    """
    from scipy.ndimage import grey_opening, maximum_filter

    rgb = _as_rgb(img)
    h, w = rgb.shape[:2]
    lum = rgb.mean(axis=2)
    tophat = np.clip(lum - grey_opening(lum, size=7), 0.0, 1.0)
    med = float(np.median(tophat))
    noise = 1.4826 * float(np.median(np.abs(tophat - med))) + 1e-6
    thr = max(thresh_sigma * noise, float(np.percentile(tophat, 99.5)))

    peaks = (tophat >= maximum_filter(tophat, size=5)) & (tophat > thr)
    ys, xs = np.nonzero(peaks)
    if ys.size == 0:
        return []
    flux = tophat[ys, xs]
    order = np.argsort(flux)[::-1]
    ys, xs, flux = ys[order], xs[order], flux[order]

    min_sep = min_sep_frac * float(np.hypot(h, w))
    kept_y: List[float] = []
    kept_x: List[float] = []
    kept: List[Dict] = []
    fmax = float(flux[0]) if flux.size else 1.0
    for y, x, f in zip(ys, xs, flux):
        if any((y - ky) ** 2 + (x - kx) ** 2 < min_sep ** 2 for ky, kx in zip(kept_y, kept_x)):
            continue
        kept_y.append(float(y)); kept_x.append(float(x))
        col = rgb[int(y), int(x), :]
        kept.append({
            "x": float(x) / max(w - 1, 1), "y": float(y) / max(h - 1, 1),
            "flux": float(f / max(fmax, 1e-6)),
            "col": [float(col[0]), float(col[1]), float(col[2])],
        })
        if len(kept) >= max_stars:
            break
    return kept


def _default_len_for(star: Dict, base_len: float) -> float:
    """A per-star length that already varies with brightness (brighter → longer)."""
    return base_len * (0.6 + 0.8 * float(star.get("flux", 0.5)))


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hash01(x, y, k):
    """Deterministic pseudo-random in [0,1) from a star position + arm index (stable
    across preview/apply — no RNG state, so the same star always jitters the same way)."""
    return float((np.sin(x * 127.1 + y * 311.7 + k * 74.7) * 43758.5453) % 1.0)


def _spectrum(t):
    """A warm→cool visible-spectrum ramp over the array ``t`` in [0,1] (orange→violet),
    for chromatic diffraction fringes. Returns ``(len(t), 3)`` RGB in [0,1]."""
    h = 0.08 + 0.67 * np.asarray(t, dtype=np.float64)    # hue: ~orange to ~violet
    s = np.full_like(h, 0.9)
    v = np.ones_like(h)
    i = np.floor(h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1 - s); q = v * (1 - f * s); tt = v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, tt, v])
    g = np.choose(i, [tt, v, v, q, p, p])
    b = np.choose(i, [p, p, tt, v, v, q])
    return np.stack([r, g, b], axis=-1)


def render_spikes(img: np.ndarray, stars: List[Dict], *, count: int = 4,
                  angle_deg: float = 0.0, thickness: float = 1.0, intensity: float = 1.0,
                  colored: bool = True, base_len: float = 0.06, fringe: float = 0.0,
                  jitter: float = 0.0, core_boost: float = 0.6) -> np.ndarray:
    """Composite diffraction spikes for ``stars`` onto ``img`` (screen blend).

    ``count`` spikes per star (3-32) at ``angle_deg`` + k·360/count; each tapers from the
    star out to ``star['len']`` (fraction of the diagonal, defaulting from brightness).
    ``thickness`` widens the core, ``intensity`` scales brightness, ``colored`` tints by the
    star's colour (else white). Two opt-in looks: ``fringe`` (0-1) blends a warm→cool
    spectral gradient into the outer part of each arm (chromatic diffraction fringes), and
    ``jitter`` (0-1) varies each arm's length deterministically. A small glint is added at
    each star centre.
    """
    base = np.asarray(img, dtype=np.float64)
    rgb = base if base.ndim == 3 else np.repeat(base[..., None], 3, axis=2)
    h, w = rgb.shape[:2]
    if not stars or intensity <= 0:
        return np.clip(base, 0.0, 1.0)
    diag = float(np.hypot(h, w))
    buf = np.zeros((h, w, 3), np.float64)
    n = int(np.clip(count, 3, 32))
    fringe = float(np.clip(fringe, 0.0, 1.0))
    jitter = float(np.clip(jitter, 0.0, 1.0))

    for star in stars:
        cx = float(star.get("x", 0.5)) * (w - 1)
        cy = float(star.get("y", 0.5)) * (h - 1)
        sx, sy = float(star.get("x", 0.5)), float(star.get("y", 0.5))
        star_n = int(np.clip(star.get("count", n), 3, 32))
        length = float(star.get("len", _default_len_for(star, base_len))) * diag
        if length < 1.0:
            continue
        a0 = np.radians(angle_deg + float(star.get("angle", 0.0)))
        col = np.array(star.get("col", [1.0, 1.0, 1.0]), dtype=np.float64) if colored \
            else np.ones(3)
        col = col / max(float(col.max()), 1e-6)          # normalise hue, keep it bright
        amp = intensity * (0.35 + 0.65 * float(star.get("flux", 0.6)))

        for k in range(star_n):
            ang = a0 + k * 2.0 * np.pi / star_n
            Lk = length * (1.0 + jitter * (2.0 * _hash01(sx, sy, k) - 1.0) * 0.5) \
                if jitter > 0 else length
            if Lk < 1.0:
                continue
            steps = max(int(Lk), 8)
            t = np.linspace(0.0, 1.0, steps)
            taper = (1.0 - t) ** 1.6                      # bright at the star, fades to the tip
            px = cx + np.cos(ang) * Lk * t
            py = cy + np.sin(ang) * Lk * t
            xi = np.round(px).astype(np.intp); yi = np.round(py).astype(np.intp)
            m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            if not np.any(m):
                continue
            if fringe > 0:                                # colour shifts toward the tip
                fmask = fringe * _smoothstep(0.2, 0.9, t)
                col_t = col[None, :] * (1.0 - fmask[:, None]) + _spectrum(t) * fmask[:, None]
                cc = col_t[m]
            wgt = (amp * taper)[m]
            for c in range(3):
                vals = wgt * (cc[:, c] if fringe > 0 else col[c])
                np.add.at(buf[..., c], (yi[m], xi[m]), vals)
        # central glint
        if 0 <= int(round(cx)) < w and 0 <= int(round(cy)) < h:
            buf[int(round(cy)), int(round(cx))] += amp * core_boost * col

    from scipy.ndimage import gaussian_filter
    sigma = max(0.6 * float(thickness), 0.4)
    for c in range(3):
        buf[..., c] = gaussian_filter(buf[..., c], sigma)

    spikes = np.clip(buf, 0.0, 1.0)
    out = 1.0 - (1.0 - np.clip(rgb, 0.0, 1.0)) * (1.0 - spikes)   # screen blend
    if base.ndim == 2:
        out = out.mean(axis=2)
    return np.clip(out, 0.0, 1.0)
