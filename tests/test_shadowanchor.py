"""P1 shadow-anchor calibration guard — the black-end mirror of the highlight roll-off.

Pulls a milky sky floor toward black WITHOUT touching cores (identity at/above the
background median), no-ops on reflection/galaxy (shadows are signal) and on already-dark
frames, and never runs the pixels out of [0, 1]. Calibrated in calibration/survey.py."""
import numpy as np

from lazystretch.processes.shadowanchor import shadow_anchor, shadow_anchor_applies_to


def _milky(H=120, W=160, seed=0):
    """A milky field: a TIGHT sky cluster ~0.12 (1%ile ~0.10, well above tau*f) + bright core."""
    rng = np.random.default_rng(seed)
    sky = np.clip(rng.normal(0.12, 0.008, (H, W)), 0.10, None)  # 1%ile ~0.10 (milky, no dark tail)
    yy, xx = np.mgrid[0:H, 0:W]
    core = 0.45 * np.exp(-(((xx - W / 2) / 12) ** 2 + ((yy - H / 2) / 10) ** 2))
    L = np.clip(sky + core, 0, 1)
    return np.stack([L, L * 0.9, L * 0.8], axis=-1)


def _clean(H=120, W=160, seed=1):
    """A field with a real dark tail (1%ile well below tau*f) -> the step must no-op."""
    rng = np.random.default_rng(seed)
    sky = np.clip(rng.normal(0.11, 0.04, (H, W)), 0.0, None)    # 1%ile ~0.02 (deep floor already)
    yy, xx = np.mgrid[0:H, 0:W]
    core = 0.45 * np.exp(-(((xx - W / 2) / 12) ** 2 + ((yy - H / 2) / 10) ** 2))
    L = np.clip(sky + core, 0, 1)
    return np.stack([L, L * 0.9, L * 0.8], axis=-1)


def test_class_gate():
    assert shadow_anchor_applies_to("emission")
    assert shadow_anchor_applies_to("open")
    assert shadow_anchor_applies_to("generic")
    assert not shadow_anchor_applies_to("reflection")
    assert not shadow_anchor_applies_to("galaxy")


def test_gated_classes_are_noops():
    img = _milky()
    for cls in ("reflection", "galaxy"):
        assert np.array_equal(shadow_anchor(img, cls), img)


def test_deepens_milky_floor():
    img = _milky()
    out = shadow_anchor(img, "emission")
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    # the 1%ile luminance drops (floor pulled toward black)
    li, lo = img.mean(2), out.mean(2)
    assert np.percentile(lo, 1) < np.percentile(li, 1) - 0.005


def test_cores_untouched_above_background():
    """Identity at/above the background median -> object cores are bit-unchanged."""
    img = _milky()
    out = shadow_anchor(img, "emission")
    f = float(np.mean([np.median(img[..., c]) for c in range(3)]))
    bright = img >= f
    assert np.allclose(out[bright], img[bright], atol=1e-9)


def test_already_dark_is_noop():
    """A frame whose floor already sits below the target is left put."""
    img = _clean()
    out = shadow_anchor(img, "emission")
    assert np.array_equal(out, img)


def test_mono_supported():
    mono = _milky()[..., 0]
    out = shadow_anchor(mono, "emission")
    assert out.shape == mono.shape and out.min() >= 0.0 and out.max() <= 1.0


def test_roughly_idempotent():
    """After one pull the floor sits near target, so a second pass barely moves it."""
    img = _milky()
    once = shadow_anchor(img, "emission")
    twice = shadow_anchor(once, "emission")
    assert np.max(np.abs(twice - once)) < 0.02
