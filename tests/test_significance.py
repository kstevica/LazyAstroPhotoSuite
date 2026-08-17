"""Significance stretch — the display holds every region to what its SNR supports."""
from __future__ import annotations

import numpy as np

from lazystretch.processes import snrmask


def _linear_scene(h=120, w=160, sky=0.020, sigma=0.001, blob_amp=0.010, seed=1):
    """Linear master: flat sky + noise, one broad blob at blob_amp/σ significance."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = blob_amp * np.exp(-(((yy - 40.0) ** 2 + (xx - 40.0) ** 2) / (2 * 10.0 ** 2)))
    lum = sky + blob + rng.normal(0, sigma, (h, w))
    img = np.stack([lum, lum, lum], axis=-1)
    noise = np.full((h, w), sigma, np.float32)
    return np.clip(img, 0, 1), noise


def test_significance_weight_separates_proof_from_noise():
    img, noise = _linear_scene()                       # blob crest at 10σ
    w = snrmask.significance_weight(img, noise)
    assert w[40, 40] > 0.85                            # proven structure: full weight
    assert float(np.median(w[90:, 100:])) < 0.15       # pure sky: below proof
    assert 0.0 <= w.min() and w.max() <= 1.0


def test_significance_weight_coverage_demotes():
    img, noise = _linear_scene()
    cov = np.full(img.shape[:2], 40, np.int32)
    cov[:, :40] = 4                                    # poorly covered strip
    w_full = snrmask.significance_weight(img, noise)
    w_cov = snrmask.significance_weight(img, noise, coverage=cov)
    assert w_cov[40, 20] < w_full[40, 20]              # blob edge in the weak strip demoted
    assert abs(w_cov[40, 80] - w_full[40, 80]) < 0.05  # fully covered side unchanged


def test_runcore_significance_stretch_eases_noise_keeps_structure():
    from lazystretch.objects.model import Parameters
    from lazystretch.pipeline.runcore import run_pipeline
    img, noise = _linear_scene(h=160, w=200)
    p_off = Parameters.for_object("generic")
    p_off.snr_noise_map = noise
    base = run_pipeline(img, p_off, preview=False)
    p_on = Parameters.for_object("generic", significance=1.0)
    p_on.snr_noise_map = noise
    res = run_pipeline(img, p_on, preview=False)
    assert any("Significance map" in s for s in res.steps_run)
    assert any("Significance stretch" in s for s in res.steps_run)
    lum_off = np.asarray(base.image)[..., :3].mean(axis=2)
    lum_on = np.asarray(res.image)[..., :3].mean(axis=2)
    # pure-noise corner: spread about the local floor shrinks under the significance gate
    off_c, on_c = lum_off[120:, 140:], lum_on[120:, 140:]
    assert np.std(on_c - np.median(on_c)) < np.std(off_c - np.median(off_c))
    # the proven blob keeps (nearly) its stretched contrast against the sky
    def contrast(l):
        return float(l[40, 40] - np.median(l[120:, 140:]))
    assert contrast(lum_on) > 0.7 * contrast(lum_off)


def test_runcore_significance_off_by_default_and_needs_map():
    from lazystretch.objects.model import Parameters
    from lazystretch.pipeline.runcore import run_pipeline
    img, noise = _linear_scene()
    p = Parameters.for_object("generic")               # significance defaults 0
    p.snr_noise_map = noise
    res = run_pipeline(img, p, preview=False)
    assert not any("Significance" in s for s in res.steps_run)
    p2 = Parameters.for_object("generic", significance=0.8)   # dial on, but no noise map
    res2 = run_pipeline(img, p2, preview=False)
    assert not any("Significance map" in s for s in res2.steps_run)
