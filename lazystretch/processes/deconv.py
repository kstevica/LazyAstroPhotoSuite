"""Classical Richardson-Lucy deconvolution — a weak, opt-in BlurX substitute (PLAN §7).

No open CLI equivalent of BlurXTerminator exists (GraXpert has no deconvolution CLI),
so this is the honest fallback: RL with an assumed Gaussian PSF. It is markedly weaker
than BlurX (prone to ringing and noise amplification), so it is **off by default** and
must be explicitly enabled. Pure numpy + scipy (no external tool).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def _gaussian_psf(sigma: float, radius: int) -> np.ndarray:
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def _rl_channel(chan: np.ndarray, psf: np.ndarray, iterations: int) -> np.ndarray:
    psf_mirror = psf[::-1, ::-1]
    est = np.clip(chan, 1e-6, 1.0)
    for _ in range(iterations):
        conv = fftconvolve(est, psf, mode="same")
        conv = np.clip(conv, 1e-9, None)
        est = est * fftconvolve(chan / conv, psf_mirror, mode="same")
        est = np.clip(est, 0.0, 1.0)
    return est


def richardson_lucy(img, sigma: float = 1.2, iterations: int = 10,
                    radius: int = 4) -> np.ndarray:
    """RL deconvolution with a Gaussian PSF; per channel. Returns a new [0,1] array."""
    a = np.asarray(img, dtype=np.float64)
    psf = _gaussian_psf(sigma, radius)
    if a.ndim == 2:
        return _rl_channel(a, psf, iterations)
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        out[..., c] = _rl_channel(a[..., c], psf, iterations)
    return out
