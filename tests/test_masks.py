"""Tests for range/highlights masks + the masked step variants (processes/masks.py)."""
import numpy as np
import pytest

from lazystretch.objects.model import Parameters
from lazystretch.pipeline.runcore import run_pipeline
from lazystretch.processes.masks import (
    background_level_masked,
    build_highlights_mask,
    build_range_mask,
    local_contrast_masked,
    noise_reduction_masked,
)


def _scene(H=160, W=200, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    blob = 0.4 * np.exp(-(((xx - W * 0.6) / 30) ** 2 + ((yy - H / 2) / 25) ** 2))
    base = np.clip(rng.normal(0.05, 0.01, (H, W)) + blob, 0, 1)
    return np.stack([base, base * 0.85, base * 0.7], axis=-1)


def test_build_range_mask_selects_band():
    L = np.tile(np.linspace(0, 1, 100), (10, 1))
    img = np.stack([L, L, L], axis=-1)
    mask = build_range_mask(img, 0.0, 0.3, fuzz=0.0, smooth=0)
    assert mask.shape == (10, 100)
    # white on the dark half (<=0.3), black on the bright end.
    assert mask[:, :25].mean() > 0.9
    assert mask[:, 60:].mean() < 0.1


def test_build_highlights_mask_selects_bright():
    L = np.tile(np.linspace(0, 1, 100), (10, 1))
    img = np.stack([L, L, L], axis=-1)
    mask = build_highlights_mask(img, 0.75)
    assert mask[:, -10:].mean() > 0.5   # bright end white
    assert mask[:, :40].mean() < 0.2    # dark end black


def test_masked_variants_run_and_stay_in_range():
    img = _scene()
    for fn in (lambda a: background_level_masked(a, 0.15),
               lambda a: local_contrast_masked(a, 0.15),
               noise_reduction_masked):
        out = fn(img)
        assert out.shape == img.shape
        assert np.isfinite(out).all() and out.min() >= -1e-9 and out.max() <= 1 + 1e-9
        assert not np.array_equal(out, img)   # something changed


def test_background_level_masked_protects_object_vs_global():
    from lazystretch.processes.backgroundlevel import background_level

    img = _scene()
    target = 0.02   # below the ~0.05 sky, so the background is pulled DOWN
    global_out = background_level(img, target)
    masked_out = background_level_masked(img, target)

    sky = (slice(0, 20), slice(0, 20))
    obj = (slice(70, 90), slice(110, 130))

    # the sky is darkened (global pull works).
    assert global_out[sky].mean() < img[sky].mean()
    # the masked variant preserves the bright object more than the global variant does.
    obj_orig = img[obj].mean()
    assert abs(masked_out[obj].mean() - obj_orig) < abs(global_out[obj].mean() - obj_orig)


def test_pipeline_masked_darkening_path():
    img = _scene()
    p = Parameters.for_object("emission", useMask=True)   # emission is in maskedDarkenAppliesTo
    r = run_pipeline(img, p, preview=True)
    assert r.steps_skipped == []
    assert any("masked" in s.lower() for s in r.steps_run)


def test_pipeline_core_protected_local_contrast():
    img = _scene()
    p = Parameters.for_object("emission", protectCores=True)
    r = run_pipeline(img, p, preview=True)
    assert r.steps_skipped == []
    assert any("core-protected" in s.lower() for s in r.steps_run)


def test_pipeline_masked_nr_on_execute():
    img = _scene()
    p = Parameters.for_object("emission", useNRMask=True)
    r = run_pipeline(img, p, preview=False)   # NR only runs in execute
    assert r.steps_skipped == []
    assert any("noise reduction" in s.lower() for s in r.steps_run)
    assert any("background-masked" in line.lower() for line in r.log)
