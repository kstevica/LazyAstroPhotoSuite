"""Tests for the auto-stretch (stretch/autostretch.py)."""
import numpy as np
import pytest

from lazystretch.stats.measure import median_channel_avg
from lazystretch.stretch.autostretch import apply_auto_stretch, solve_stretch


def _noise(median, dev, shape=(200, 200), seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(median, dev, size=shape), 0.0, 1.0)


def test_output_in_range_and_original_untouched():
    img = _noise(0.08, 0.02)
    orig = img.copy()
    out, res = apply_auto_stretch(img, -1.25, 0.25)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.array_equal(img, orig), "input array must not be modified"


@pytest.mark.parametrize("target", [0.12, 0.15, 0.20, 0.25])
def test_background_lands_near_target(target):
    # median commutes with the monotonic stretch, so median(out) ~= target.
    img = _noise(0.06, 0.015, seed=1)
    out, res = apply_auto_stretch(img, -1.25, target)
    got = median_channel_avg(out)
    assert abs(got - target) < 5e-3, (target, got)


def test_solve_stretch_from_raw_stats_matches_image_path():
    img = _noise(0.07, 0.02, seed=3)
    out, res = apply_auto_stretch(img, -1.30, 0.20)
    from lazystretch.stats.measure import avg_dev_channel_avg
    med = median_channel_avg(img)
    dev = avg_dev_channel_avg(img)
    res2 = solve_stretch(med, dev, -1.30, 0.20)
    assert abs(res.c0 - res2.c0) < 1e-12
    assert abs(res.m - res2.m) < 1e-12
    assert res.knee_ran == res2.knee_ran


def test_monotonic_mapping():
    img = _noise(0.07, 0.02, seed=2)
    out, res = apply_auto_stretch(img, -1.25, 0.25)
    order = np.argsort(img, axis=None)
    y = out.flatten()[order]
    assert np.all(np.diff(y) >= -1e-9), "stretch must be order-preserving"


def test_knee_runs_on_typical_dark_stack():
    img = _noise(0.08, 0.02, seed=4)
    out, res = apply_auto_stretch(img, -1.25, 0.25)
    assert res.knee_ran is True
    assert res.avgDev > 0.0
    assert 0.0 <= res.c0 <= 1.0
    assert 0.0 < res.m < 1.0


def test_rgb_linked_stretch():
    rgb = np.stack([_noise(0.05, 0.015, seed=i) for i in range(3)], axis=-1)
    out, res = apply_auto_stretch(rgb, -1.05, 0.25)
    assert out.shape == rgb.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert abs(median_channel_avg(out) - 0.25) < 1e-2
