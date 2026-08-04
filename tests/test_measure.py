"""Tests for the statistics primitives (stats/measure.py)."""
import numpy as np
import pytest

from lazystretch.stats.measure import (
    avg_dev,
    avg_dev_channel_avg,
    mean_channel_avg,
    median_channel_avg,
    region_median_channel_avg,
)


def test_avg_dev_two_point():
    x = np.array([0.0, 0.0, 1.0, 1.0])  # median 0.5 -> mean|dev| = 0.5
    assert abs(avg_dev(x) - 0.5) < 1e-12


def test_avg_dev_symmetric_ramp():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # median 3 -> (2+1+0+1+2)/5 = 1.2
    assert abs(avg_dev(x) - 1.2) < 1e-12


def test_avg_dev_is_not_std():
    x = np.array([0.0, 0.0, 0.0, 10.0])  # median 0 -> avgDev 2.5, std ~4.33
    assert abs(avg_dev(x) - 2.5) < 1e-12


def test_channel_avg_mono_and_rgb():
    mono = np.full((10, 10), 0.3)
    assert abs(median_channel_avg(mono) - 0.3) < 1e-12
    rgb = np.stack(
        [np.full((10, 10), 0.1), np.full((10, 10), 0.2), np.full((10, 10), 0.6)],
        axis=-1,
    )
    assert abs(median_channel_avg(rgb) - 0.3) < 1e-12  # (0.1+0.2+0.6)/3
    assert abs(mean_channel_avg(rgb) - 0.3) < 1e-12
    assert abs(avg_dev_channel_avg(rgb) - 0.0) < 1e-12  # each channel constant


def test_region_median():
    a = np.zeros((10, 10))
    a[2:5, 3:6] = 0.7
    assert abs(region_median_channel_avg(a, 3, 2, 6, 5) - 0.7) < 1e-12


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        median_channel_avg(np.zeros((2, 2, 3, 1)))
