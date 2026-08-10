"""P7 part 2 — Deepen second-pass + Advisor / Analyze-Frame."""
import numpy as np
import pytest

from lazystretch.data.loader import get_data
from lazystretch.objects.analyze import AnalyzeResult, analyze_view, measure_noise
from lazystretch.objects.model import Parameters
from lazystretch.pipeline.runcore import run_pipeline
from lazystretch.processes import deepen

D = get_data()


def _stretched_rgb(H=160, W=220, seed=0):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(0.2, 0.05, (H, W, 3)), 0, 1)
    img[40:46, 60:66, :] = 0.95   # a bright star
    return img


# --- Deepen -----------------------------------------------------------------

def test_deepen_lifts_and_protects_highlights():
    img = _stretched_rgb()
    out = deepen.deepen_stretch(img, 0.6)
    assert np.median(out) > np.median(img)                       # background lifted
    star_lift = out[43, 63].mean() - img[43, 63].mean()
    bg_lift = np.median(out) - np.median(img)
    assert star_lift < bg_lift                                   # highlight rides far less
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_deepen_noop_and_mono():
    img = _stretched_rgb()
    assert np.array_equal(deepen.deepen_stretch(img, 0.0), img)
    mono = deepen.deepen_stretch(img[..., 0], 0.5)
    assert mono.ndim == 2 and np.median(mono) > np.median(img[..., 0])


def test_restore_geometry_mismatch_returns_target():
    tgt = _stretched_rgb()
    src = _stretched_rgb()[:100]
    assert np.array_equal(deepen.restore_source_highlights(tgt, src, 0.0, 0.6), tgt)


def test_restore_pulls_highlights_toward_source():
    src = _stretched_rgb()
    tgt = deepen.deepen_stretch(src, 0.6)     # deepened (star slightly lifted)
    out = deepen.restore_source_highlights(tgt, src, 0.0, 0.6)
    # the bright star should move back toward the (ridden) source, away from the lifted target
    assert abs(out[43, 63].mean() - src[43, 63].mean()) < abs(tgt[43, 63].mean() - src[43, 63].mean())
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_deepen_pipeline_polish_mode():
    img = _stretched_rgb()
    p = Parameters.for_object("emission", inputStretched=True, deepen=0.6)
    r = run_pipeline(img, p, preview=True)
    assert r.steps_skipped == []
    assert any("Deepen" in s for s in r.steps_run)
    assert any("Restore source highlights" in s for s in r.steps_run)
    assert not any("Auto-stretch" in s for s in r.steps_run)   # skipped in polish mode


# --- Advisor ----------------------------------------------------------------

def _lp_scene(H=360, W=520, seed=0):
    rng = np.random.default_rng(seed)
    ramp = np.tile(np.linspace(0, 0.5, W), (H, 1))
    return np.stack([np.clip(rng.normal(0.05, 0.02, (H, W)) + ramp, 0, 1) for _ in range(3)], axis=-1)


def test_measure_noise_on_white_noise():
    rng = np.random.default_rng(1)
    img = np.clip(0.3 + rng.normal(0, 0.02, (256, 256)), 0, 1)
    sig = measure_noise(img)
    assert len(sig) == 1
    assert abs(sig[0] - 0.02) < 0.004        # MAD-based estimate near the true sigma


def test_analyze_fires_expected_recommendations():
    p = Parameters.for_object("emission")
    res = analyze_view(_lp_scene(), p, D)
    assert isinstance(res, AnalyzeResult) and res.lines
    keys = {r["key"] for r in res.recommendations}
    assert {"gradientCleanup", "reduceCast", "darkLaneGC", "dehaze", "chromaNR"}.issubset(keys)
    # params must NOT be mutated by analysis (emission seeds a modest chroma-NR default)
    assert p.dehaze == 0.0 and p.chromaNR == 0.30 and p.reduceCast is False


def test_analyze_mono_suppresses_colour_recs():
    p = Parameters.for_object("emission")
    res = analyze_view(_lp_scene()[..., 0], p, D)
    keys = {r["key"] for r in res.recommendations}
    assert "dehaze" not in keys and "chromaNR" not in keys       # colour-only recs skipped on mono


def test_analyze_input_stretched_suppresses_dust_noise():
    p = Parameters.for_object("emission", inputStretched=True)
    res = analyze_view(_lp_scene(), p, D)
    text = " ".join(res.lines)
    assert "Dust lift" not in text and "Noise:" not in text      # readings taken as-is
