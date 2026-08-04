"""Integration tests: narrowband combine + the headless pipeline spine."""
import numpy as np
import pytest

from lazystretch.objects.model import Parameters
from lazystretch.palettes.combine import combine_narrowband, combine_osc_narrowband
from lazystretch.pipeline.runcore import run_pipeline


def _synthetic_rgb(H=180, W=240, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    blob = 0.15 * np.exp(-(((xx - W * 0.6) / 40) ** 2 + ((yy - H / 2) / 35) ** 2))
    base = np.clip(rng.normal(0.05, 0.008, (H, W)) + blob, 0, 1)
    return np.stack([base, base * 0.8, base * 0.7], axis=-1)


# --- narrowband combine ------------------------------------------------------

def test_combine_narrowband_channel_assignment():
    ha = np.full((8, 8), 0.4)
    o3 = np.full((8, 8), 0.2)
    s2 = np.full((8, 8), 0.6)
    sho = combine_narrowband("SHO", ha, o3, s2)   # R=SII G=Ha B=OIII
    assert sho[0, 0, 0] == pytest.approx(0.6)
    assert sho[0, 0, 1] == pytest.approx(0.4)
    assert sho[0, 0, 2] == pytest.approx(0.2)
    hoo = combine_narrowband("HOO", ha, o3)        # R=Ha G=OIII B=OIII
    assert hoo[0, 0, 0] == pytest.approx(0.4) and hoo[0, 0, 2] == pytest.approx(0.2)
    # SII missing in SHO -> falls back to Ha in the red slot.
    sho2 = combine_narrowband("SHO", ha, o3)
    assert sho2[0, 0, 0] == pytest.approx(0.4)


def test_combine_osc_narrowband():
    osc = np.zeros((4, 4, 3))
    osc[..., 0] = 0.5   # R -> Ha
    osc[..., 1] = 0.2   # G
    osc[..., 2] = 0.4   # B  -> o3 = (0.2+0.4)/2 = 0.3
    hoo = combine_osc_narrowband(osc, "HOO")
    assert hoo[0, 0, 0] == pytest.approx(0.5)
    assert hoo[0, 0, 1] == pytest.approx(0.3)
    assert hoo[0, 0, 2] == pytest.approx(0.3)


# --- parameters --------------------------------------------------------------

def test_for_object_applies_profile_toggles():
    g = Parameters.for_object("galaxy")
    assert g.doHDR is True and g.doStarReduce is True      # galaxy profile
    o = Parameters.for_object("open")
    assert o.doHDR is False and o.doSCNR is False           # open profile
    e = Parameters.for_object("emission", satAdj=0.2)
    assert e.satAdj == 0.2                                  # override applied


def test_for_object_rejects_unknown_override():
    with pytest.raises(TypeError):
        Parameters.for_object("galaxy", nonsense=1)


# --- pipeline ----------------------------------------------------------------

def test_pipeline_rgb_preview_runs_clean():
    rgb = _synthetic_rgb()
    r = run_pipeline(rgb, Parameters.for_object("emission"), preview=True)
    assert r.steps_skipped == []
    assert r.image.ndim == 3 and r.image.shape[2] == 3
    assert 0.0 <= r.image.min() and r.image.max() <= 1.0
    assert r.stretch is not None and r.stretch.knee_ran in (True, False)
    # background lifted from a dark linear median toward the class look.
    assert np.median(r.image) > np.median(rgb)


def test_pipeline_mono_skips_color_steps():
    mono = _synthetic_rgb()[..., 0]
    r = run_pipeline(mono, Parameters.for_object("galaxy"), preview=True)
    assert r.steps_skipped == []
    assert r.image.ndim == 2
    # color-only steps must not appear in the run list on a mono image.
    joined = " ".join(r.steps_run).lower()
    assert "scnr" not in joined and "saturation" not in joined
    assert any("stretch" in s.lower() for s in r.steps_run)


def test_pipeline_input_stretched_gates_off_stretch_and_colorcal():
    rgb = _synthetic_rgb()
    r = run_pipeline(rgb, Parameters.for_object("emission", inputStretched=True), preview=True)
    joined = " ".join(r.steps_run).lower()
    assert "auto-stretch" not in joined
    assert "color calibration" not in joined
    assert r.stretch is None


def test_preview_and_execute_both_complete():
    rgb = _synthetic_rgb()
    for preview in (True, False):
        r = run_pipeline(rgb, Parameters.for_object("emission"), preview=preview)
        assert r.steps_skipped == [], (preview, r.steps_skipped)
        assert np.isfinite(r.image).all()


def test_narrowband_pipeline_from_mono_channels():
    ha = _synthetic_rgb(seed=1)[..., 0]
    o3 = _synthetic_rgb(seed=2)[..., 0] * 0.6
    s2 = _synthetic_rgb(seed=3)[..., 0] * 0.8
    p = Parameters.for_object("emission", palette="SHO (Hubble)")
    p.ha, p.oiii, p.sii = ha, o3, s2
    r = run_pipeline(None, p, preview=True)   # image=None: built from channels
    assert r.image.ndim == 3 and r.image.shape[2] == 3
    assert r.steps_skipped == []
