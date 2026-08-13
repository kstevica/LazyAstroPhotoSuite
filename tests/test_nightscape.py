"""Nightscape sky/foreground segmentation (lazystack/nightscape.py)."""
import numpy as np

from lazystretch.lazystack import nightscape as ns


def _synth(H=200, W=260, fg_side="right", seed=0):
    """A synthetic nightscape: smooth dark sky + a few stars, and a structured foreground block."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    img = 0.05 + 0.02 * (xx / W) + rng.normal(0, 0.002, (H, W))     # dark sky + mild gradient
    fg = np.zeros((H, W), bool)
    if fg_side == "right":
        fg[:, int(W * 0.72):] = True
    else:                                                           # bottom
        fg[int(H * 0.72):, :] = True
    for _ in range(40):                                             # stars in the sky only
        y, x = rng.integers(0, H), rng.integers(0, W)
        if not fg[y, x]:
            img[y, x] += 0.6
    tex = 0.30 + 0.12 * np.sin(xx / 2.5) * np.sin(yy / 2.5)         # structured land
    img[fg] = tex[fg]
    return np.clip(img, 0, 1), fg


def test_segment_sky_vertical_right_foreground():
    img, fg = _synth(fg_side="right")
    mask, info = ns.segment_sky(img)
    assert info["axis"] == "vertical" and info["sky_side"] == "left"
    assert mask[:, :40].mean() > 0.8            # far sky side ≈ sky
    assert mask[:, -20:].mean() < 0.2           # far foreground side ≈ foreground


def test_segment_sky_horizontal_bottom_foreground():
    img, fg = _synth(fg_side="bottom")
    mask, info = ns.segment_sky(img)
    assert info["axis"] == "horizontal" and info["sky_side"] == "top"
    assert mask[:30, :].mean() > 0.8            # top ≈ sky
    assert mask[-20:, :].mean() < 0.2           # bottom ≈ foreground


def test_segment_sky_bias_shifts_boundary():
    img, _ = _synth(fg_side="right")
    less = ns.segment_sky(img, bias=-0.08)[1]["foreground_fraction"]
    base = ns.segment_sky(img, bias=0.0)[1]["foreground_fraction"]
    more = ns.segment_sky(img, bias=0.08)[1]["foreground_fraction"]
    assert less > base > more                   # +bias → more sky, −bias → more foreground


def test_segment_sky_mask_is_feathered_and_ranged():
    img, _ = _synth(fg_side="right")
    mask, _ = ns.segment_sky(img)
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    assert np.any((mask > 0.2) & (mask < 0.8))  # a soft seam exists (not a hard 0/1 cut)


def test_foreground_sharpness_prefers_sharper():
    img, fg = _synth(fg_side="right")
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(img, 2.0)
    assert ns.foreground_sharpness(img, fg) > ns.foreground_sharpness(blurred, fg)
