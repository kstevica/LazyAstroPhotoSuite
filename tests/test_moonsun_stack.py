"""LazyMoonSun stacking engines — global burst stack + multi-point."""
import numpy as np
import pytest

from lazystretch.moonsun import register as reg, stack
from lazystretch.moonsun.model import MoonSunParams
from tests.test_moonsun_register import _disc


def _load(a):
    return a


def _burst(n=8, shift=0.0, noise=0.02, seed=0):
    """n frames of a disc, each randomly sub-pixel shifted and noised."""
    rng = np.random.default_rng(seed)
    base = _disc(256, radius=70)
    frames = []
    for _ in range(n):
        sx = rng.uniform(-shift, shift)
        sy = rng.uniform(-shift, shift)
        f = reg.apply_shift(base, sx, sy) if shift else base.copy()
        f = np.clip(f + rng.normal(0, noise, f.shape), 0, 1)
        frames.append(f)
    return base, frames


def test_stack_reduces_noise():
    base, frames = _burst(n=9, shift=0.0, noise=0.03)
    res = stack.stack_burst(frames, _load, keep=1.0)
    assert res is not None and res["n_stacked"] == 9
    core = slice(70, 186)
    single = np.abs(frames[0][core, core] - base[core, core]).std()
    stacked = np.abs(res["master"][core, core] - base[core, core]).std()
    assert stacked < 0.55 * single, (single, stacked)      # ~1/sqrt(9) ≈ 0.33


def test_stack_stays_sharp_vs_naive_mean():
    _base, frames = _burst(n=8, shift=3.0, noise=0.01, seed=3)
    res = stack.stack_burst(frames, _load, keep=1.0)
    assert res is not None
    naive = np.mean(frames, axis=0)                        # unregistered = smeared
    # raw gradient energy (grad_prep 'q' = std of the pre-rescale magnitude) is the
    # sharpness measure; registration preserves it, a naive mean smears it away.
    reg_sharp = reg.grad_prep(reg.intensity(res["master"]))["q"]
    naive_sharp = reg.grad_prep(reg.intensity(naive))["q"]
    assert reg_sharp > 1.1 * naive_sharp, (reg_sharp, naive_sharp)


def test_junk_gate_drops_black_frame():
    _base, frames = _burst(n=6, shift=0.0, noise=0.02)
    frames.append(np.zeros((256, 256)))                    # cap-on / aim-away frame
    res = stack.stack_burst(frames, _load, keep=1.0)
    assert res is not None
    assert res["n_graded"] == 6                            # the black frame was gated out


def test_stack_needs_two_frames():
    base, _ = _burst(n=1)
    assert stack.stack_burst([base], _load) is None


def test_multipoint_blends_onto_master():
    base, frames = _burst(n=8, shift=2.0, noise=0.03, seed=5)
    params = MoonSunParams(ap_size=48, ap_keep=0.6, ap_search=8, ap_max=200)
    res = stack.stack_multipoint(frames, _load, base, params)
    assert res is not None
    assert res["master"].shape == base.shape
    assert np.isfinite(res["master"]).all()
    assert res["n_aps"] >= 4 and res["mean_kept"] >= 3
    assert "anisoplanatism" in res["report"]
    assert res["aniso_max"] >= 0.0


def test_multipoint_reduces_noise_in_covered_region():
    base, frames = _burst(n=10, shift=0.0, noise=0.04, seed=8)
    params = MoonSunParams(ap_size=48, ap_keep=1.0, ap_search=8, ap_max=300)
    res = stack.stack_multipoint(frames, _load, base, params)
    assert res is not None
    core = slice(96, 160)                                  # disc centre, well covered
    single = np.abs(frames[0][core, core] - base[core, core]).std()
    stacked = np.abs(res["master"][core, core] - base[core, core]).std()
    assert stacked < single, (single, stacked)
