"""Dim-core dial — mask the large bright veil and multiplicatively lower its luminosity.
Colour preserved, sky/faint areas untouched, never brightens."""
import numpy as np

from lazystretch.objects.model import Parameters
from lazystretch.pipeline.runcore import run_pipeline
from lazystretch.processes.brightcore import dim_bright_core


def _nebula(H=200, W=260, seed=0):
    """A big bright core on a dark sky (large-scale luminance well above the mask floor)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    core = 0.55 * np.exp(-(((xx - W / 2) / 45) ** 2 + ((yy - H / 2) / 40) ** 2))
    L = np.clip(0.05 + core + rng.normal(0, 0.004, (H, W)), 0, 1)
    return np.stack([L, L * 0.85, L * 0.78], axis=-1)


def test_amount_zero_is_noop():
    img = _nebula()
    assert np.array_equal(dim_bright_core(img, 0.0), img)


def test_lowers_bright_core_luminosity():
    img = _nebula()
    out = dim_bright_core(img, 0.6)
    assert out.shape == img.shape and out.min() >= 0.0 and out.max() <= 1.0
    L, Lo = img.mean(2), out.mean(2)
    bright = L > 0.4                                    # the core
    assert Lo[bright].mean() < L[bright].mean() * 0.9   # core dimmed significantly


def test_never_brightens():
    img = _nebula()
    out = dim_bright_core(img, 0.8)
    assert np.all(out <= img + 1e-9)


def test_sky_untouched():
    img = _nebula()
    out = dim_bright_core(img, 0.8)
    dark = img.mean(2) < 0.10                            # sky background, below mask floor
    assert np.allclose(out[dark], img[dark], atol=5e-3)


def test_preserves_colour_ratio():
    img = _nebula()
    out = dim_bright_core(img, 0.7)
    r, g = img[..., 0], img[..., 1]
    ro, go = out[..., 0], out[..., 1]
    m = r > 0.05
    assert np.allclose(go[m] / ro[m], g[m] / r[m], atol=1e-6)   # hue preserved


def test_mono_supported():
    mono = _nebula()[..., 0]
    out = dim_bright_core(mono, 0.6)
    assert out.shape == mono.shape and out.max() <= 1.0


def test_pipeline_wires_dim_core():
    img = _nebula()
    on = run_pipeline(img, Parameters.for_object("emission", dimCore=0.6), preview=True)
    off = run_pipeline(img, Parameters.for_object("emission", dimCore=0.0), preview=True)
    assert any("Dim core" in s for s in on.steps_run)
    assert not any("Dim core" in s for s in off.steps_run)
