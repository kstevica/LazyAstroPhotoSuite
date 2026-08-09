"""LazyMoonSun registration crux — synthetic sub-pixel shift recovery.

Mirrors RegistrationProbe.js's acceptance test: shift a known frame by a known amount,
measure it back, and require the recovered offset to match to well under a pixel.
"""
import numpy as np
import pytest

from lazystretch.moonsun import register as reg


def _disc(size=256, cx=None, cy=None, radius=70):
    """A sun-like disc: limb-darkened, with a few surface spots for gradient content."""
    cx = size / 2 if cx is None else cx
    cy = size / 2 if cy is None else cy
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mu = np.clip(1.0 - (r / radius) ** 2, 0.0, 1.0)
    disc = np.where(r <= radius, 0.35 + 0.55 * np.sqrt(mu), 0.0)   # limb darkening
    rng = np.random.default_rng(7)
    for _ in range(12):                                            # sunspot-like features
        a = rng.uniform(-0.18, -0.06)
        sx, sy = rng.uniform(cx - 45, cx + 45), rng.uniform(cy - 45, cy + 45)
        disc += a * np.exp(-(((xx - sx) / 5) ** 2 + ((yy - sy) / 5) ** 2))
    return np.clip(disc, 0.0, 1.0)


def _measure_shift(ref_img, shifted_img):
    wk_r, _ = reg.working(ref_img)
    gr = reg.gradient(wk_r)
    ref = reg.make_ref(gr, gr.shape[1], gr.shape[0])
    wk_t, _ = reg.working(shifted_img)
    gt = reg.gradient(wk_t)
    return reg.measure_against(ref, gt, gt.shape[1], gt.shape[0])


@pytest.mark.parametrize("sx,sy", [(0, 0), (3, 0), (0, -2), (5, 4), (-6, 3)])
def test_integer_shift_recovered(sx, sy):
    base = _disc()
    moved = reg.apply_shift(base, sx, sy)
    dx, dy = _measure_shift(base, moved)
    # measure returns the shift of the target vs ref == the applied shift
    assert abs(dx - sx) < 0.2 and abs(dy - sy) < 0.2, (dx, dy)


@pytest.mark.parametrize("sx,sy", [(2.4, -1.7), (5.5, 3.3), (-3.2, 4.6), (0.5, -0.5)])
def test_subpixel_shift_recovered(sx, sy):
    base = _disc()
    moved = reg.apply_shift(base, sx, sy)
    dx, dy = _measure_shift(base, moved)
    err = np.hypot(dx - sx, dy - sy)
    assert err < 0.3, (sx, sy, dx, dy, err)          # RegistrationProbe's <0.3px bar


def test_apply_shift_roundtrip_aligns():
    base = _disc()
    moved = reg.apply_shift(base, 4.0, -3.0)
    dx, dy = _measure_shift(base, moved)
    realigned = reg.apply_shift(moved, -dx, -dy)      # how the stacker undoes the drift
    core = slice(40, 216)
    err = np.abs(realigned[core, core] - base[core, core]).mean()
    assert err < 0.02, err


def test_grad_prep_axis_energy_flags_1d_edge():
    # a purely horizontal step has gradient along one axis only -> tiny min/max axis ratio
    edge = np.zeros((96, 96))
    edge[48:, :] = 1.0
    gp = reg.grad_prep(edge, taper=12)
    ratio_edge = min(gp["ex"], gp["ey"]) / max(gp["ex"], gp["ey"])
    assert ratio_edge < 0.15                          # the 1D-edge gate would reject this
    # a 2D-structured disc patch has comparable per-axis energy
    gp2 = reg.grad_prep(_disc(96, radius=30), taper=12)
    ratio = min(gp2["ex"], gp2["ey"]) / max(gp2["ex"], gp2["ey"])
    assert ratio > 0.3


def test_disc_center_finds_offset_disc():
    img = _disc(256, cx=150, cy=110, radius=60)
    wk, _ = reg.working(img)
    c = reg.disc_center(wk)
    assert c is not None
    assert abs(c[0] - 150) < 4 and abs(c[1] - 110) < 4
