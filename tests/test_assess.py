"""Tests for the self-calibrating assessments (stats/assess.py)."""
import numpy as np
import pytest

from lazystretch.stats.assess import (
    adaptive_floor_raise,
    auto_assess,
    measure_dust,
)


def test_adaptive_floor_raise_clamps():
    assert adaptive_floor_raise(-0.1) == 0.0
    assert abs(adaptive_floor_raise(0.01) - 0.01) < 1e-12
    assert adaptive_floor_raise(0.5) == 0.03  # capped at raiseCap


def test_measure_dust_zero_on_clean_field():
    a = np.full((400, 600), 0.2)  # uniform -> cleanSky == typical -> lift 0
    d = measure_dust(a)
    assert d.lift == 0.0


def test_measure_dust_positive_on_gradient():
    H, W = 400, 600
    ramp = np.linspace(0.0, 0.1, W)
    a = np.tile(0.1 + ramp, (H, 1))  # one side brighter -> typical > cleanSky
    d = measure_dust(a)
    assert d.lift > 0.0
    assert d.typical >= d.cleanSky


def test_auto_assess_small_frame_returns_none():
    assert auto_assess(np.full((64, 64), 0.1)) is None


def test_auto_assess_clean_frame_minimal_crop():
    a = np.full((500, 700), 0.1)  # uniform -> gradient 0 -> crop 3, ABE
    r = auto_assess(a)
    assert r is not None
    assert r.cropPercent == 3
    assert r.useGC is False
    assert r.gradient == 0.0


def test_auto_assess_strong_gradient_picks_gc():
    H, W = 500, 700
    a = np.tile(np.linspace(0.0, 0.5, W), (H, 1))  # strong horizontal gradient
    r = auto_assess(a)
    assert r is not None
    assert r.cropPercent >= 6
    assert r.useGC is True
    assert r.gradient >= 0.20


def test_auto_assess_ignores_single_bright_corner():
    # A lone bright object clipping ONE perimeter patch must not read as a gradient
    # (dropBrightest=1 uses the 2nd-brightest as the reference).
    a = np.full((500, 700), 0.1)
    a[10:60, 10:60] = 0.9  # bright top-left corner only
    r = auto_assess(a)
    assert r is not None
    assert r.cropPercent == 3
    assert r.useGC is False
