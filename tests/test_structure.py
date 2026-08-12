"""Scale-separated mid-scale structure enhancement (P1) — multiscale.structure_boost."""
import numpy as np

from lazystretch.processes import multiscale as ms


def test_structure_boost_enhances_medium_feature():
    # a medium-scale blob (sigma 8 ≈ 20 px) on a bright subject → its contrast should rise
    yy, xx = np.mgrid[0:160, 0:160].astype(float)
    blob = 0.2 * np.exp(-(((xx - 80) ** 2 + (yy - 80) ** 2) / (2 * 8.0 ** 2)))
    L = np.clip(0.4 + blob, 0.0, 1.0)
    out = ms.structure_boost(L, 0.8)
    c_in = L[80, 80] - L[80, 40]                       # blob peak vs surround
    c_out = out[80, 80] - out[80, 40]
    assert c_out > 1.15 * c_in                         # medium structure enhanced
    _, rin = ms._atrous_planes(L, 6)
    _, rout = ms._atrous_planes(out, 6)
    assert abs(rout.mean() - rin.mean()) < 0.02        # large-scale base preserved (core/halo)


def test_structure_boost_does_not_amplify_a_fine_star():
    # a 1-px star on a bright subject must NOT be amplified (fine scale protected)
    L = np.full((128, 128), 0.4)
    L[64, 64] = 0.9
    out = ms.structure_boost(L, 0.8)
    assert out[64, 64] <= L[64, 64] + 0.02             # star peak not boosted


def test_structure_boost_protects_dark_background():
    rng = np.random.default_rng(0)
    L = np.clip(0.02 + rng.normal(0, 0.01, (128, 128)), 0.0, 1.0)   # dark sky, base < bg_lo
    out = ms.structure_boost(L, 1.0)
    assert np.allclose(out, L, atol=1e-3)             # subject mask ≈ 0 → noise not amplified


def test_structure_boost_preserves_hue():
    xx = np.mgrid[0:128, 0:128][1].astype(float)
    L = np.clip(0.5 + 0.05 * np.sin(2 * np.pi * xx / 16.0), 0.0, 1.0)
    rgb = np.stack([L, 0.6 * L, 0.4 * L], axis=-1)
    out = ms.structure_boost(rgb, 0.8)
    m = out[..., 0] > 1e-3
    assert np.allclose(out[..., 1][m] / out[..., 0][m], 0.6, atol=1e-3)
    assert np.allclose(out[..., 2][m] / out[..., 0][m], 0.4, atol=1e-3)


def test_structure_boost_amount_zero_and_no_mutation():
    L = np.full((64, 64), 0.5)
    L[::4] += 0.05
    before = L.copy()
    assert np.array_equal(ms.structure_boost(L, 0.0), L)   # off = identity
    ms.structure_boost(L, 1.0)
    assert np.array_equal(L, before)                       # input never mutated


def test_galaxy_profile_enables_structure():
    from lazystretch.objects.model import Parameters
    assert Parameters.for_object("galaxy").structure == 0.6
    assert Parameters.for_object("generic").structure == 0.0
