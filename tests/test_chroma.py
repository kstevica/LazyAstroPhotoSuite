"""Chromatic-aberration correction — aligns R/G to B from star-measured offsets."""
import numpy as np
from scipy.ndimage import map_coordinates

from lazystretch.processes import chroma


def _starfield(H=500, W=500, n=110, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.full((H, W), 0.05)
    for _ in range(n):
        y, x = rng.uniform(30, H - 30), rng.uniform(30, W - 30)
        base += 0.9 * np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.6 ** 2)))
    return np.clip(base, 0, 1), (yy, xx)


def _radial_warp(ch, k, yy, xx):
    H, W = ch.shape
    cy, cx = H / 2, W / 2
    r = np.hypot(yy - cy, xx - cx) + 1e-6
    amp = k * r / (H / 2)
    return map_coordinates(ch, [yy - amp * (yy - cy) / r, xx - amp * (xx - cx) / r],
                           order=1, mode="nearest")


def test_ca_correction_removes_radial_offset():
    base, (yy, xx) = _starfield()
    img = np.stack([base, base, base], -1).astype(np.float64)
    img[..., 0] = _radial_warp(img[..., 0], 2.4, yy, xx)        # R offset radially
    img[..., 1] = _radial_warp(img[..., 1], 1.2, yy, xx)        # G offset radially
    before = chroma.measure_offsets(img)["median_px"]
    fixed = chroma.correct_chromatic_aberration(img)
    after = chroma.measure_offsets(fixed)["median_px"]
    assert before > 1.0
    assert after < 0.3 * before                                 # large reduction


def test_ca_strength_scales_correction():
    base, (yy, xx) = _starfield(seed=5)
    img = np.stack([base, base, base], -1).astype(np.float64)
    img[..., 0] = _radial_warp(img[..., 0], 2.4, yy, xx)
    full = chroma.measure_offsets(chroma.correct_chromatic_aberration(img, strength=1.0))["median_px"]
    half = chroma.measure_offsets(chroma.correct_chromatic_aberration(img, strength=0.5))["median_px"]
    zero = chroma.measure_offsets(chroma.correct_chromatic_aberration(img, strength=0.0))["median_px"]
    assert full < half < zero                                   # more strength = more correction


def test_ca_mono_and_too_few_stars_unchanged():
    mono = _starfield()[0]
    assert np.array_equal(chroma.correct_chromatic_aberration(mono), mono)   # mono no-op
    blank = np.stack([np.full((80, 80), 0.1)] * 3, -1)          # no stars
    out = chroma.correct_chromatic_aberration(blank, min_stars=12)
    assert np.array_equal(out, blank)
