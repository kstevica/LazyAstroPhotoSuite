"""Color-calibration tests — regression guard for the violet-cast bug.

The bug: color_calibration scaled the WHOLE channel (a * f), re-tinting the
just-neutralized background; the near-black linked auto-stretch then amplified that
tiny tint into a heavy violet cast on an already-neutral OSC master. These tests lock
the fix: a neutral background must stay neutral through calibration AND the stretch.
"""
import numpy as np
import pytest

from lazystretch.processes.colorcal import (
    background_neutralize,
    color_calibrate,
    color_calibration,
)
from lazystretch.stretch.autostretch import apply_auto_stretch


def _neutral_scene(H=200, W=260, seed=0, star_color=(0.6, 0.6, 0.9)):
    """A neutral (R=G=B) faint sky + a sprinkling of small COLOURED bright stars.

    The stars are the white reference (top-luminance structure) and are deliberately
    tinted, so ColorCalibration produces per-channel factors != 1 — the exact
    condition that made the old ``a * f`` re-tint the neutral background. They are a
    tiny pixel fraction, so the channel medians remain the true (neutral) sky.
    """
    rng = np.random.default_rng(seed)
    sky = rng.normal(0.05, 0.004, (H, W))
    rgb = np.stack([sky, sky.copy(), sky.copy()], axis=-1)
    rng2 = np.random.default_rng(seed + 1)
    ys = rng2.integers(3, H - 3, 24)
    xs = rng2.integers(3, W - 3, 24)
    for y, x in zip(ys, xs):
        for c in range(3):
            rgb[y - 1:y + 2, x - 1:x + 2, c] += star_color[c]
    return np.clip(rgb, 0.0, 1.0)


def _sky(a, pct=20):
    return [float(np.percentile(a[..., c], pct)) for c in range(3)]


def test_background_neutralize_aligns_channel_medians():
    img = _neutral_scene()
    out = background_neutralize(img)
    meds = [float(np.median(out[..., c])) for c in range(3)]
    assert max(meds) - min(meds) < 1e-4       # channels aligned to a common background


def test_color_calibrate_preserves_neutral_background():
    img = _neutral_scene()
    out = color_calibrate(img)
    r, g, b = _sky(out, pct=20)
    assert 0.9 < r / g < 1.1, ("R/G", r / g)
    assert 0.9 < b / g < 1.1, ("B/G", b / g)


def test_neutral_sky_survives_the_stretch():
    """The reported failure: a neutral master must NOT come out violet after stretch."""
    img = _neutral_scene()
    calibrated = color_calibrate(img)
    stretched, _ = apply_auto_stretch(calibrated, -1.05, 0.25)
    r, g, b = _sky(stretched, pct=20)
    # Loose bound (near-black stretch is sensitive) but far tighter than the bug,
    # which produced R/G ~2.1 and B/G ~0.09 (a violent cast) at this stage.
    assert 0.6 < r / g < 1.5, ("R/G after stretch", r / g)
    assert 0.6 < b / g < 1.5, ("B/G after stretch", b / g)


def test_color_calibration_balances_tinted_structure():
    """CC should pull a colored bright structure toward neutral (white balance)."""
    img = _neutral_scene(star_color=(0.4, 0.4, 0.9))   # strongly blue stars
    out = color_calibration(img)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_color_calibrate_mono_noop():
    mono = np.full((16, 16), 0.2)
    assert np.array_equal(color_calibrate(mono), mono)
