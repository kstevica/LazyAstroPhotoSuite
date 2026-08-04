"""Tests for the pinned MTF primitives (stats/mtf.py)."""
import numpy as np
import pytest

from lazystretch.stats.mtf import (
    FIND_MIDTONES_EPS,
    find_midtones_balance,
    mtf,
    mtf_scalar,
    range_clip,
)

MS = [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]


def test_mtf_endpoints():
    for m in MS:
        assert float(mtf(m, 0.0)) == 0.0
        assert float(mtf(m, 1.0)) == 1.0
        assert abs(float(mtf(m, m)) - 0.5) < 1e-12  # mtf(m, m) == 0.5 by definition


def test_mtf_identity_at_half():
    x = np.linspace(0.0, 1.0, 101)
    assert np.allclose(mtf(0.5, x), x, atol=1e-12)


def test_mtf_monotonic_and_bounded():
    x = np.linspace(0.0, 1.0, 257)
    for m in MS:
        y = mtf(m, x)
        assert np.all(np.diff(y) >= -1e-12), f"not monotonic at m={m}"
        assert np.all(y >= -1e-12) and np.all(y <= 1.0 + 1e-12)


def test_mtf_closed_form_values():
    # Independent of the implementation: computed directly from the pinned formula.
    cases = [
        (0.25, 0.5, 0.75),
        (0.75, 0.5, 0.25),
        (0.10, 0.5, 0.90),
        (0.40, 0.2, 3.0 / 11.0),
        (0.50, 0.37, 0.37),
        (0.25, 0.25, 0.50),
    ]
    for m, x, expected in cases:
        assert abs(float(mtf(m, x)) - expected) < 1e-12, (m, x)


def test_mtf_clamps_outside_unit_interval():
    assert float(mtf(0.3, -0.5)) == 0.0
    assert float(mtf(0.3, 1.5)) == 1.0


def test_mtf_scalar_matches_vectorized():
    for m in MS:
        for x in np.linspace(0.0, 1.0, 51):
            assert abs(mtf_scalar(m, float(x)) - float(mtf(m, x))) < 1e-12


def test_find_midtones_endpoints():
    assert find_midtones_balance(0.25, 0.0) == 0.0
    assert find_midtones_balance(0.25, 1.0) == 1.0
    assert find_midtones_balance(0.25, -0.1) == 0.0
    assert find_midtones_balance(0.25, 1.1) == 1.0


def test_find_midtones_roundtrip():
    for v1 in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95]:
        for v0 in [0.05, 0.1, 0.25, 0.4, 0.6, 0.9]:
            m = find_midtones_balance(v0, v1)
            assert 0.0 <= m <= 1.0
            assert abs(mtf_scalar(m, v1) - v0) < FIND_MIDTONES_EPS + 1e-9


def test_find_midtones_brightens_dark_background():
    # Classic auto-stretch: a dark median (0.05) lifted to 0.25 needs m < 0.5.
    m = find_midtones_balance(0.25, 0.05)
    assert 0.0 < m < 0.5


def test_range_clip():
    assert float(range_clip(1.5, 0.0, 1.0)) == 1.0
    assert float(range_clip(-1.0, 0.0, 1.0)) == 0.0
    assert float(range_clip(0.5, 0.0, 1.0)) == 0.5
