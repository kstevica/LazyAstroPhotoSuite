"""FFT phase-correlation registration primitives — the LazyMoonSun engine crux.

A faithful numpy/scipy port of the self-contained PJSR math in ``LazyMoonSun.js``
(``SUN.gradient``/``fftOf``/``measureAgainst`` :352-419, ``gradPrep`` :1874-1897). No
PixInsight processes, no star detection: every frame is reduced to a tapered Kroon-gradient
image, cross-correlated against a reference spectrum with a Gaussian spectral apodization,
and the peak is refined to sub-pixel with a log-parabolic fit. Validated against synthetic
sub-pixel shifts to the probe's <0.3 px bar (see tests/test_moonsun_register.py).

Convention (matching the .js): coordinates are (dx, dy) = (column, row). ``measure_against``
returns the shift of ``grad`` relative to the reference, such that shifting the target by
(-dx, -dy) aligns it to the reference — exactly how the stacker consumes it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import convolve, shift as nd_shift, zoom

WORKDIM = 1024          # SUN.WORKDIM — coarse working size
REFDIM = 2048           # SUN.REFDIM — full-res refine window

# SUN.kroon (LazyMoonSun.js:296-301) — a 5x5 vertical-derivative kernel; its transpose
# is the horizontal derivative. Squared before summing, so kernel sign/flip is irrelevant.
_KROON = np.array([
    [+0.0007, +0.0052, +0.0370, +0.0052, +0.0007],
    [+0.0037, +0.1187, +0.2589, +0.1187, +0.0037],
    [0.0,     0.0,     0.0,     0.0,     0.0],
    [-0.0037, -0.1187, -0.2589, -0.1187, -0.0037],
    [-0.0007, -0.0052, -0.0370, -0.0052, -0.0007],
], dtype=np.float64)


# --------------------------------------------------------------------------- basics


def intensity(img: np.ndarray) -> np.ndarray:
    """PI ``getIntensity``: mono passes through, RGB collapses to the mean of channels."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        return a[..., :3].mean(axis=2)
    return a


def _rescale(a: np.ndarray) -> np.ndarray:
    """PI ``Image.rescale``: linear stretch to [0, 1] (no-op if flat)."""
    lo = float(a.min())
    hi = float(a.max())
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def working(img: np.ndarray, workdim: int = WORKDIM) -> Tuple[np.ndarray, float]:
    """Intensity, downsampled so the long edge is <= ``workdim``. Returns (wk, k), k<=1."""
    I = intensity(img)
    long_edge = max(I.shape[0], I.shape[1])
    k = 1.0
    if long_edge > workdim:
        k = workdim / long_edge
        I = zoom(I, k, order=1)
    return I, k


def _taper(a: np.ndarray, T: int) -> np.ndarray:
    """SUN.taper: raised-cosine edge window of width T on all four sides (in place)."""
    H, W = a.shape
    T = int(min(T, H // 2, W // 2))
    if T <= 0:
        return a
    w = 0.5 * (1.0 - np.cos(np.pi * np.arange(T) / T))   # w[0]=0 .. w[T-1]->1
    # rows
    a[:T, :] *= w[:, None]
    a[H - T:, :] *= w[::-1][:, None]
    # columns
    a[:, :T] *= w[None, :]
    a[:, W - T:] *= w[::-1][None, :]
    return a


def gradient(intensity_img: np.ndarray, taper: int = 24) -> np.ndarray:
    """SUN.gradient: Kroon gradient magnitude, rescaled to [0,1] and edge-tapered.

    Zero pixels are replaced by the median first (PI ``1+x==1`` guard) so black borders
    don't skew the rescale.
    """
    g = np.array(intensity_img, dtype=np.float64, copy=True)
    med = float(np.median(g))
    g[g == 0.0] = med
    ix = convolve(g, _KROON, mode="nearest")
    iy = convolve(g, _KROON.T, mode="nearest")
    mag = np.sqrt(ix * ix + iy * iy)
    mag = _rescale(mag)
    return _taper(mag, taper)


def grad_prep(intensity_img: np.ndarray, taper: int = 24) -> dict:
    """SUN.gradPrep: like ``gradient`` but also returns the per-AP quality metrics.

    ``q`` = std-dev of the raw (pre-rescale) gradient magnitude (feature richness);
    ``ex``/``ey`` = sqrt(mean) of the squared per-axis derivatives (2D-structure gate).
    """
    g = np.array(intensity_img, dtype=np.float64, copy=True)
    med = float(np.median(g))
    g[g == 0.0] = med
    ix = convolve(g, _KROON, mode="nearest")
    iy = convolve(g, _KROON.T, mode="nearest")
    ix2 = ix * ix
    iy2 = iy * iy
    ex = float(np.sqrt(ix2.mean()))
    ey = float(np.sqrt(iy2.mean()))
    mag = np.sqrt(ix2 + iy2)
    q = float(mag.std())
    mag = _taper(_rescale(mag), taper)
    return {"g": mag, "q": q, "ex": ex, "ey": ey}


# --------------------------------------------------------------------- correlation

_WEIGHT_CACHE: dict = {}


def weight_image(size: int) -> np.ndarray:
    """SUN.weightImage: DC-cornered Gaussian spectral apodization, exp(-(fx^2+fy^2)/s2)."""
    cached = _WEIGHT_CACHE.get(size)
    if cached is not None:
        return cached
    idx = np.arange(size)
    f = np.minimum(idx, size - idx)           # wrap-around distance from DC (corner)
    fx2 = (f * f)[None, :]
    fy2 = (f * f)[:, None]
    s2 = 2.0 * (0.15 * size) ** 2
    W = np.exp(-(fx2 + fy2) / s2)
    _WEIGHT_CACHE[size] = W
    return W


def fft_of(grad: np.ndarray, sz: int, w: int, h: int) -> np.ndarray:
    """SUN.fftOf: centre-pad the (h,w) gradient into an sz×sz canvas and FFT it."""
    canvas = np.zeros((sz, sz), dtype=np.float64)
    oy = (sz - h) >> 1
    ox = (sz - w) >> 1
    canvas[oy:oy + h, ox:ox + w] = grad
    return np.fft.fft2(canvas)


def _parabola(fm: float, fc: float, fp: float) -> float:
    """Log-parabolic sub-pixel offset from three samples (minus, centre, plus)."""
    fm = np.log(max(fm, 1.0e-12))
    fc = np.log(max(fc, 1.0e-12))
    fp = np.log(max(fp, 1.0e-12))
    d = fm - 2.0 * fc + fp
    if abs(d) < 1.0e-12:
        return 0.0
    return float(min(0.5, max(-0.5, 0.5 * (fm - fp) / d)))


def make_ref(grad: np.ndarray, w: int, h: int) -> dict:
    """Prepare a reference spectrum from a tapered gradient (square-padded to its long edge)."""
    size = max(w, h)
    return {"c0": fft_of(grad, size, w, h), "size": size}


def measure_against(ref: dict, grad: np.ndarray, w: int, h: int) -> Tuple[float, float]:
    """SUN.measureAgainst: sub-pixel shift (dx, dy) of ``grad`` vs the reference spectrum."""
    size = ref["size"]
    c1 = fft_of(grad, size, w, h)
    # PI's crossPowerSpectrumMatrix(A, B) conjugates the first argument, so the recovered
    # peak is the shift of B (target) relative to A (reference) with the .js sign — such
    # that translating the target by (-dx, -dy) aligns it. (conj(A)·B, not A·conj(B).)
    cc = np.conj(ref["c0"]) * c1
    mag = np.abs(cc)
    R = cc / np.where(mag > 0.0, mag, 1.0)         # normalized cross-power spectrum
    R = R * weight_image(size)
    corr = _rescale(np.real(np.fft.ifft2(R)))
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)

    x0 = px - 1 if px > 0 else size - 1
    x1 = px + 1 if px < size - 1 else 0
    y0 = py - 1 if py > 0 else size - 1
    y1 = py + 1 if py < size - 1 else 0
    dx = px + _parabola(corr[py, x0], corr[py, px], corr[py, x1])
    dy = py + _parabola(corr[y0, px], corr[py, px], corr[y1, px])
    if dx >= size / 2:
        dx -= size
    if dy >= size / 2:
        dy -= size
    return float(dx), float(dy)


def apply_shift(img: np.ndarray, dx: float, dy: float, order: int = 3) -> np.ndarray:
    """PI ``Image.translate``: shift content by (dx, dy) = (col, row), zero-filled borders."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        out = np.empty_like(a)
        for c in range(a.shape[2]):
            out[..., c] = nd_shift(a[..., c], (dy, dx), order=order, mode="constant", cval=0.0)
        return out
    return nd_shift(a, (dy, dx), order=order, mode="constant", cval=0.0)


def disc_center(wk: np.ndarray) -> Optional[Tuple[float, float]]:
    """SUN.discCenterWk: centroid of the bright mask (> 0.5·p99.9) on a working image."""
    p999 = float(np.quantile(wk, 0.999))
    thr = 0.5 * p999
    ys, xs = np.nonzero(wk > thr)
    if xs.size < 50:
        return None
    return float(xs.mean()), float(ys.mean())
