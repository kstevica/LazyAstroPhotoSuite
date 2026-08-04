"""P1 HDR calibration guard — HDR must compress the large-scale core AND preserve/reveal
fine detail (the HDRMultiscaleTransform signature). Calibrated against PI's M42 core;
this synthetic test locks the behaviour so the detail-destroying regression can't return.
"""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, uniform_filter

from lazystretch.processes.multiscale import hdr_core


def _scene(H=200, W=280):
    """A bright large-scale 'core' blob + fine sinusoidal detail (a stand-in for a nebula
    core with structure)."""
    yy, xx = np.mgrid[0:H, 0:W]
    blob = 0.70 * np.exp(-(((xx - W / 2) / 45.0) ** 2 + ((yy - H / 2) / 38.0) ** 2))
    fine = 0.06 * np.sin(xx / 2.5) * np.sin(yy / 2.5)
    L = np.clip(0.05 + blob + fine, 0.0, 1.0)
    return np.stack([L, L * 0.9, L * 0.8], axis=-1)


def _detail_rms(L):
    return float((L - uniform_filter(L, size=7)).std())


def test_hdr_compresses_core_and_reveals_detail():
    img = _scene()
    L0 = img.mean(axis=2)
    out = hdr_core(img, 6)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0

    L1 = out.mean(axis=2)
    center = (slice(80, 120), slice(120, 160))
    # (a) the bright large-scale core is compressed (its coarse level drops)
    assert gaussian_filter(L1, 8)[center].mean() < gaussian_filter(L0, 8)[center].mean()
    # (b) fine detail is PRESERVED or REVEALED, never destroyed (the anti-regression:
    #     the old log/multiplicative reconstruction dropped detail to ~0.73x).
    assert _detail_rms(L1) >= _detail_rms(L0) * 0.95, (_detail_rms(L1), _detail_rms(L0))


def test_hdr_detail_boost_is_monotonic():
    img = _scene()
    d_lo = _detail_rms(hdr_core(img, 6, detail_boost=1.0).mean(axis=2))
    d_hi = _detail_rms(hdr_core(img, 6, detail_boost=1.4).mean(axis=2))
    assert d_hi > d_lo


def test_hdr_compression_controls_core_level():
    img = _scene()
    center = (slice(80, 120), slice(120, 160))
    strong = gaussian_filter(hdr_core(img, 6, compression=0.4).mean(axis=2), 8)[center].mean()
    gentle = gaussian_filter(hdr_core(img, 6, compression=0.9).mean(axis=2), 8)[center].mean()
    assert strong < gentle          # lower compression => flatter/darker core


def test_hdr_mono_in_range():
    out = hdr_core(_scene()[..., 0], 6)
    assert out.ndim == 2 and out.min() >= 0.0 and out.max() <= 1.0
