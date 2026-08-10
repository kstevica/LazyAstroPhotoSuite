"""Background-neutrality fixes (review-driven).

Both external reviewers flagged a magenta/green-deficient cast and an unmodelled gradient on the
Milky Way widefield. Fixes: (1) neutralize colour on the DARKEST-patch sky, not the whole-frame
signal median; (2) the milkyway class uses the purpose-built darkLaneGC (deg-2, dark-lane
anchored) instead of the ill-posed ABE, and re-neutralizes the background after SCNR (reduceCast).
"""
import numpy as np

from lazystretch.objects.model import Parameters
from lazystretch.pipeline.runcore import run_pipeline
from lazystretch.processes.colorcal import background_neutralize


def _osc(H=140, W=170, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    gas = 0.30 * np.exp(-(((xx - W / 2) / 45) ** 2 + ((yy - H / 2) / 34) ** 2))
    L = np.clip(0.06 + gas + rng.normal(0, 0.006, (H, W)), 0, 1)
    return np.stack([L, L * 0.82, L * 0.7], axis=-1)


def test_milkyway_profile_gradient_and_cast_defaults():
    p = Parameters.for_object("milkyway")
    assert p.doBgExtract is False        # ill-posed ABE off for wall-to-wall MW
    assert p.darkLaneGC is True          # purpose-built gradient model on
    assert p.reduceCast is True          # post-SCNR re-neutralization on
    e = Parameters.for_object("emission")
    assert e.doBgExtract is True and e.darkLaneGC is False and e.reduceCast is False


def test_darkest_patch_bn_neutralizes_sky_under_colored_signal():
    # Magenta sky (R,B > G) with a sharp GREEN block over 60% of the frame that dominates the
    # whole-frame median. Whole-frame-median BN keys on the green signal and leaves the sky cast;
    # darkest-patch BN keys on the clean 40% sky and neutralizes it.
    H, W = 120, 150
    img = np.stack([np.full((H, W), 0.06), np.full((H, W), 0.04), np.full((H, W), 0.06)], axis=-1)
    img[:, :90, 1] += 0.30                        # sharp green block, 60% of columns, clean edge
    out = background_neutralize(np.clip(img, 0, 1))
    sky = np.zeros((H, W), bool)
    sky[:, 90:] = True                            # the clean sky region (no signal)
    meds = [float(np.median(out[..., c][sky])) for c in range(3)]
    assert max(meds) - min(meds) < 5e-3          # sky neutral despite signal-dominated frame


def test_milkyway_pipeline_uses_darklane_and_reducecast_not_abe():
    res = run_pipeline(_osc(), Parameters.for_object("milkyway"), preview=True)
    steps = " | ".join(res.steps_run)
    assert "Dark-lane gradient model" in steps
    assert "Reduce background color cast" in steps
    assert "ABE" not in steps                     # generic background extraction disabled for MW


def test_debug_background_captures_the_model():
    img = _osc()
    on = run_pipeline(img, Parameters.for_object("emission", debugBackground=True), preview=False)
    assert on.background_model is not None                       # ABE model captured for inspection
    assert on.background_model.shape[:2] == on.image.shape[:2]
    off = run_pipeline(img, Parameters.for_object("emission"), preview=False)
    assert off.background_model is None                          # off by default
