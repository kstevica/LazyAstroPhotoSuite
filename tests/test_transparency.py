"""Core-transparency dial — large-scale local contrast masked to the bright nebulosity so
dust lanes deepen and the core reads transparent. Colour preserved, sky background untouched."""
import numpy as np

from lazystretch.objects.model import Parameters
from lazystretch.pipeline.runcore import run_pipeline
from lazystretch.processes.transparency import core_transparency


def _nebula(H=160, W=200, seed=0):
    """A bright structured core (gas + a dark lane) on a dark sky."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    gas = 0.35 * np.exp(-(((xx - W / 2) / 40) ** 2 + ((yy - H / 2) / 34) ** 2))
    lane = 0.18 * np.exp(-(((xx - W / 2) / 6) ** 2))            # a dark dust lane through it
    L = np.clip(0.05 + gas - lane + rng.normal(0, 0.004, (H, W)), 0, 1)
    return np.stack([L, L * 0.85, L * 0.78], axis=-1)


def _core_contrast(a):
    """Local-contrast proxy in the bright core region."""
    from scipy.ndimage import uniform_filter
    L = a.mean(2) if a.ndim == 3 else a
    core = L[L.shape[0] // 3: 2 * L.shape[0] // 3, L.shape[1] // 3: 2 * L.shape[1] // 3]
    return float((core - uniform_filter(core, 5)).std())


def test_amount_zero_is_noop():
    img = _nebula()
    assert np.array_equal(core_transparency(img, 0.0), img)


def test_raises_core_local_contrast():
    img = _nebula()
    out = core_transparency(img, 0.6)
    assert out.shape == img.shape and out.min() >= 0.0 and out.max() <= 1.0
    assert _core_contrast(out) > _core_contrast(img) * 1.1       # dust lane / gas contrast up


def test_preserves_colour_ratio():
    img = _nebula()
    out = core_transparency(img, 0.6)
    # scaling R:G:B together preserves hue: G/R and B/R unchanged where R>0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    ro, go, bo = out[..., 0], out[..., 1], out[..., 2]
    m = r > 0.02                                                # ro = r*scale > 0 there too
    assert np.allclose(go[m] / ro[m], g[m] / r[m], atol=1e-6)
    assert np.allclose(bo[m] / ro[m], b[m] / r[m], atol=1e-6)


def test_background_untouched():
    img = _nebula()
    out = core_transparency(img, 0.8)
    L = img.mean(2)
    dark = L < 0.10                                              # sky background (below mask floor)
    assert np.allclose(out[dark], img[dark], atol=5e-3)


def test_mono_supported():
    mono = _nebula()[..., 0]
    out = core_transparency(mono, 0.6)
    assert out.shape == mono.shape and out.min() >= 0.0 and out.max() <= 1.0


def test_pipeline_wires_transparency():
    img = _nebula()
    on = run_pipeline(img, Parameters.for_object("emission", transparency=0.8), preview=True)
    off = run_pipeline(img, Parameters.for_object("emission", transparency=0.0), preview=True)
    assert any("Core transparency" in s for s in on.steps_run)
    assert not any("Core transparency" in s for s in off.steps_run)
