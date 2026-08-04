"""P1 saturation — ColorSaturation ported as a CIE Lab chroma scale (perceptually uniform),
replacing the RGB distance-from-gray scale that over-boosted saturated colours (clipping)
and under-boosted subtle ones. Guards the colour space + behaviour."""
import numpy as np
import pytest

from lazystretch.processes.tone import _lab_to_srgb, _srgb_to_lab, saturation


def _chroma(a):
    return a.max(axis=2) - a.min(axis=2)


def test_lab_roundtrip_is_exact():
    rng = np.random.default_rng(0)
    img = np.clip(rng.random((32, 40, 3)), 0.005, 0.995)
    L, a, b = _srgb_to_lab(img)
    back = _lab_to_srgb(L, a, b)
    assert np.max(np.abs(back - img)) < 1e-9


def test_saturation_amount_zero_is_identity():
    rng = np.random.default_rng(1)
    img = np.clip(rng.random((20, 25, 3)), 0.01, 0.99)
    assert np.max(np.abs(saturation(img, 0.0) - img)) < 1e-9


def test_saturation_mono_noop():
    mono = np.full((8, 8), 0.4)
    assert np.array_equal(saturation(mono, 0.6), mono)


def test_saturation_increases_chroma():
    rng = np.random.default_rng(2)
    img = np.clip(rng.random((30, 30, 3)), 0.05, 0.95)
    assert _chroma(saturation(img, 0.5)).mean() > _chroma(img).mean()


def test_saturation_boosts_subtle_colour_perceptually():
    # the RGB version under-boosted subtle colours; the Lab scale must give a real lift.
    subtle = np.array([[[0.55, 0.50, 0.52]]])          # chroma 0.05
    out = saturation(subtle, 0.85)
    assert _chroma(out)[0, 0] / _chroma(subtle)[0, 0] > 1.4


def test_saturation_preserves_hue_order():
    px = np.array([[[0.6, 0.45, 0.3]]])                # warm: R > G > B
    out = saturation(px, 0.5)[0, 0]
    assert out[0] > out[1] > out[2]
