"""Local-contrast (LHE) overshoot guard — the unsharp stand-in must NOT ring into dark
halos around bright stars (models PI LHE's slopeLimit). It should still enhance nebula-
scale contrast; only extreme star overshoot is soft-clipped.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from lazystretch.processes import tone


def _scene(H=200, W=200):
    """A uniform mid background (room for the dark undershoot to show without clipping to 0)
    with a bright saturated star, whose steep edge makes the unsharp overshoot a dark ring."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.clip(0.40 + rng.normal(0, 0.008, (H, W)), 0.0, 1.0)
    out[((yy - 100) ** 2 + (xx - 100) ** 2) < 9 ** 2] = 1.0
    return out


def _edge_mask(H=200, W=200):
    yy, xx = np.mgrid[0:H, 0:W]
    ring = (yy - 100) ** 2 + (xx - 100) ** 2
    return (ring > 10 ** 2) & (ring < 18 ** 2)                   # just outside the star


def test_lhe_overshoot_clamp_reduces_dark_undershoot():
    img = _scene()
    edge = _edge_mask()
    old = tone._LHE_OVERSHOOT
    try:
        tone._LHE_OVERSHOOT = 999.0                 # effectively off (pre-fix behaviour)
        before = tone.local_contrast(img, 0.30)
        tone._LHE_OVERSHOOT = 3.0                    # the shipped clamp
        after = tone.local_contrast(img, 0.30)
    finally:
        tone._LHE_OVERSHOOT = old
    # the clamp never darkens the ring further, and strictly lifts the deepest undershoot
    assert after[edge].min() >= before[edge].min()
    assert after[edge].min() > before[edge].min() + 1e-3


def test_lhe_still_enhances_contrast_away_from_stars():
    img = _scene()
    out = tone.local_contrast(img, 0.30)
    edge = _edge_mask()
    hp_in = img - gaussian_filter(img, 8)
    hp_out = out - gaussian_filter(out, 8)
    m = ~edge                                                    # away from the star overshoot
    m[95:106, 95:106] = False
    assert np.abs(hp_out[m]).mean() >= 0.9 * np.abs(hp_in[m]).mean()


def test_lhe_mono_and_rgb_run_and_clip():
    mono = _scene()
    assert tone.local_contrast(mono, 0.2).shape == mono.shape
    rgb = np.stack([mono, mono * 0.9, mono * 0.8], axis=-1)
    out = tone.local_contrast(rgb, 0.2)
    assert out.shape == rgb.shape and out.min() >= 0.0 and out.max() <= 1.0
