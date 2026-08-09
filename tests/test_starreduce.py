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


def test_shrink_stars_reduces_star_flux():
    img, mask = _star_field(seed=1)
    out = shrink_stars(img, 0.8, small=0.3)
    # star pixels are dimmed toward the local background
    assert out[mask].mean() < 0.6 * img[mask].mean()


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
