"""De-veil — colour-preserving background-floor deepener (processes/deveil.py)."""
import numpy as np

from lazystretch.processes.deveil import deepen_background


def _milky(H=200, W=260):
    """A brownish milky background (R>B>G) with a bright pinkish nebula blob."""
    img = np.zeros((H, W, 3)) + np.array([0.16, 0.13, 0.14])   # elevated warm veil
    yy, xx = np.mgrid[0:H, 0:W]
    blob = 0.5 * np.exp(-(((xx - W / 2) / 25.0) ** 2 + ((yy - H / 2) / 20.0) ** 2))
    img = img + blob[..., None] * np.array([0.9, 0.6, 0.7])     # pink nebula
    return np.clip(img, 0.0, 1.0)


def test_deepen_background_deepens_milky_floor_keeps_subject():
    img = _milky()
    out = deepen_background(img, 0.8)
    lum = img.mean(2)
    bg = lum < np.percentile(lum, 40)
    neb = lum > np.percentile(lum, 98)
    assert out.mean(2)[bg].mean() < 0.6 * lum[bg].mean()        # milky floor pulled down
    assert out.mean(2)[neb].mean() > 0.75 * lum[neb].mean()     # bright nebula largely preserved


def test_deepen_background_preserves_hue():
    img = _milky()
    out = deepen_background(img, 0.8)
    m = out[..., 1] > 1e-3
    r_in = img[..., 0][m] / np.clip(img[..., 1][m], 1e-6, None)
    r_out = out[..., 0][m] / np.clip(out[..., 1][m], 1e-6, None)
    assert np.allclose(r_out, r_in, rtol=2e-3)                  # ratio-scaled → hue/real colour held


def test_deepen_background_self_limiting_on_dark_field():
    rng = np.random.default_rng(0)
    img = np.clip(0.03 + rng.normal(0, 0.005, (140, 160, 3)), 0, 1)   # already dark sky
    out = deepen_background(img, 1.0)
    assert np.max(np.abs(out - img)) < 0.02                     # no-op (floor already below target)


def test_deepen_background_amount_zero_and_no_mutation():
    img = _milky()
    before = img.copy()
    assert np.array_equal(deepen_background(img, 0.0), img)     # off = identity
    deepen_background(img, 1.0)
    assert np.array_equal(img, before)                         # input never mutated


def test_deepen_background_mono():
    img = np.full((140, 160), 0.15)
    img[50:70, 60:90] = 0.6
    out = deepen_background(img, 0.8)
    assert out.ndim == 2
    assert out[0, 0] < img[0, 0]                               # background deepened
    assert out[60, 75] > 0.7 * img[60, 75]                     # bright region preserved


def test_deepenbg_default_and_recipe_roundtrip():
    from lazystretch.objects.model import Parameters
    from lazystretch.io import recipes as R
    assert Parameters.for_object("emission").deepenBackground == 0.0
    assert Parameters.for_object("galaxy").deepenBackground == 0.0
    p = Parameters(); p.deepenBackground = 0.5
    p2 = Parameters(); R.apply_recipe(p2, R.recipe_from_params(p))
    assert p2.deepenBackground == 0.5
