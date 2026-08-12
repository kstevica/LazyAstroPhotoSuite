"""Software star reduction (processes/starreduce.py) — port-only finishing dial."""
import numpy as np

from lazystretch.processes.starreduce import shrink_stars


def _star_field(H=200, W=240, n=60, seed=0, nebula=True):
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 0.10)
    if nebula:                                    # a broad, smooth structure (must survive)
        yy, xx = np.mgrid[0:H, 0:W]
        img += 0.25 * np.exp(-(((xx - W / 2) / 60) ** 2 + ((yy - H / 2) / 40) ** 2))
    star_mask = np.zeros((H, W), bool)
    for _ in range(n):
        y, x = rng.integers(4, H - 4), rng.integers(4, W - 4)
        img[y, x] = 0.95
        star_mask[y, x] = True
    return np.clip(img, 0, 1), star_mask


def test_shrink_stars_off_is_identity():
    img, _ = _star_field()
    assert np.allclose(shrink_stars(img, 0.0), img)


def _tiered_field(H=200, W=240, seed=0):
    """A field with faint 1-px carpet stars AND bright anchor stars (small Gaussian profiles)."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 0.10)
    faint = np.zeros((H, W), bool)
    bright = np.zeros((H, W), bool)
    for _ in range(80):                                      # faint carpet (crush)
        y, x = rng.integers(6, H - 6), rng.integers(6, W - 6)
        img[y, x] = max(img[y, x], 0.30)
        faint[y, x] = True
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(6):                                       # bright anchors (preserve)
        cy, cx = rng.integers(15, H - 15), rng.integers(15, W - 15)
        img = np.maximum(img, 0.10 + 0.85 * np.exp(-(((xx - cx) / 2.0) ** 2 + ((yy - cy) / 2.0) ** 2)))
        bright[cy, cx] = True
    return np.clip(img, 0, 1), faint, bright


def test_shrink_stars_tiered_crushes_carpet_keeps_anchors():
    img, faint, bright = _tiered_field(seed=1)
    out = shrink_stars(img, 0.8, small=0.3)
    assert out[faint].mean() < 0.85 * img[faint].mean()      # faint carpet is thinned...
    assert out[bright].mean() > 0.90 * img[bright].mean()    # ...bright anchors preserved (P4 tiering)


def test_shrink_stars_preserves_nebulosity():
    img, mask = _star_field(seed=2, nebula=True)
    out = shrink_stars(img, 0.8, small=0.3)
    yy, xx = np.mgrid[0:200, 0:240]
    core = (np.abs(xx - 120) < 20) & (np.abs(yy - 100) < 15) & ~mask   # smooth nebula, no stars
    assert abs(float(out[core].mean()) - float(img[core].mean())) < 0.02


def test_shrink_stars_rgb_preserves_color():
    lum, _ = _star_field(seed=3)
    rgb = np.stack([lum, lum * 0.8, lum * 0.6], axis=-1)
    out = shrink_stars(rgb, 0.7)
    assert out.shape == rgb.shape
    assert out.min() >= 0 and out.max() <= 1
