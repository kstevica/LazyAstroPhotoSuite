"""Nightscape sky/foreground segmentation (lazystack/nightscape.py)."""
import numpy as np
import pytest

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


def _star_sky(N=8, H=170, W=210, seed=0):
    """A star-rich sky (left) + structured foreground (right), N frames, independent noise."""
    yy, xx = np.mgrid[0:H, 0:W]
    rng0 = np.random.default_rng(seed)
    pos = [(rng0.integers(10, H - 10), rng0.integers(10, int(W * 0.6))) for _ in range(45)]
    frames = []
    for i in range(N):
        img = np.full((H, W), 0.04)
        for (y, x) in pos:
            img += 0.9 * np.exp(-(((xx - x) / 1.4) ** 2 + ((yy - y) / 1.4) ** 2))
        img[:, int(W * 0.72):] = 0.3 + 0.05 * np.sin(yy[:, int(W * 0.72):] / 3.0)   # foreground
        img = img + np.random.default_rng(seed + 1 + i).normal(0, 0.03, (H, W))
        frames.append(np.clip(np.stack([img] * 3, axis=-1), 0, 1))
    return frames


def test_pick_foreground_returns_sharpest_index():
    frames = _star_sky(N=5)
    from scipy.ndimage import gaussian_filter
    frames[2] = np.clip(gaussian_filter(frames[2], (2, 2, 0)), 0, 1)   # blur frame 2's foreground
    mask, _ = ns.segment_sky(frames[0])
    best, scores = ns.pick_foreground(frames, 1 - mask)
    assert best != 2 and len(scores) == 5                              # not the blurred one


def test_build_sky_master_denoises_and_keeps_shape():
    pytest.importorskip("astroalign")
    frames = _star_sky(N=8)
    mask, info = ns.segment_sky(frames[0])
    sky, kept = ns.build_sky_master(frames, mask, reference=0)
    assert sky.shape == frames[0].shape and len(kept) >= 6 and np.isfinite(sky).all()
    from scipy.ndimage import gaussian_filter
    reg = (slice(55, 115), slice(20, 75))
    def hf(im):                                                        # robust high-freq noise (star-safe)
        L = im.mean(2)[reg]
        return float(np.median(np.abs(L - gaussian_filter(L, 3))))
    assert hf(sky) < 0.8 * hf(frames[0])                              # stacking reduced the noise


def test_composite_blends_foreground_over_sky():
    from lazystretch.processes import nightscape as nsc
    H, W = 60, 80
    sky = np.full((H, W, 3), 0.4)
    fg = np.full((H, W, 3), 0.03)                                     # dark linear foreground
    mask = np.zeros((H, W))
    mask[:, :W // 2] = 1.0                                            # left = sky, right = foreground
    out = nsc.composite(sky, fg, mask, brightness=0.5)
    assert np.allclose(out[30, 5], 0.4, atol=0.02)                   # sky side unchanged
    assert out[30, W - 5].mean() > fg[0, 0].mean()                   # foreground developed (lifted)
    assert out.shape == sky.shape


def test_composite_no_develop_uses_foreground_asis():
    from lazystretch.processes import nightscape as nsc
    sky = np.full((40, 50, 3), 0.4)
    fg = np.full((40, 50, 3), 0.25)
    mask = np.zeros((40, 50))                          # all foreground
    asis = nsc.composite(sky, fg, mask, develop=False, feather_frac=0.0)
    assert np.allclose(asis, 0.25, atol=1e-6)          # already-developed foreground used verbatim
    dev = nsc.composite(sky, fg, mask, develop=True, feather_frac=0.0)
    assert not np.allclose(dev, 0.25)                  # develop=True changes it


def test_composite_feather_widens_the_seam():
    from lazystretch.processes import nightscape as nsc
    sky = np.full((80, 80, 3), 0.9)
    fg = np.zeros((80, 80, 3))
    mask = np.zeros((80, 80))
    mask[:, :40] = 1.0                                 # sharp sky/foreground split
    hard = nsc.composite(sky, fg, mask, develop=False, feather_frac=0.0)
    soft = nsc.composite(sky, fg, mask, develop=False, feather_frac=0.06)
    def transition(o):
        v = o[40, :, 0]
        return int(((v > 0.1) & (v < 0.8)).sum())      # partial-blend pixels along the seam
    assert transition(soft) > transition(hard)


def test_develop_foreground_default_true_and_recipe():
    from lazystretch.objects.model import Parameters
    assert Parameters().developForeground is True


def test_composite_shape_mismatch_is_safe_noop():
    from lazystretch.processes import nightscape as nsc
    sky = np.full((40, 50, 3), 0.4)
    fg = np.full((30, 30, 3), 0.1)
    mask = np.ones((40, 50))
    assert np.array_equal(nsc.composite(sky, fg, mask), sky)          # mismatched fg → leave sky


def test_develop_foreground_brightness_monotonic():
    from lazystretch.processes import nightscape as nsc
    fg = np.full((40, 50, 3), 0.05)
    dim = nsc.develop_foreground(fg, 0.2).mean()
    bright = nsc.develop_foreground(fg, 0.9).mean()
    assert bright > dim > fg[0, 0].mean()                            # brighter dial → brighter, both lifted


def test_aligner_sky_mask_ignores_foreground():
    pytest.importorskip("astroalign")
    from lazystretch.lazystack import register as reg
    frames = _star_sky(N=2)
    mask, _ = ns.segment_sky(frames[0])
    al = reg.Aligner(frames[0], sky_mask=mask)
    out = al.align(frames[1])                                         # sky-masked registration runs
    assert out is not None and out.shape == frames[1].shape


def test_refine_mask_scribbles_snap_to_edge():
    pytest.importorskip("skimage")
    H, W = 120, 160
    med = np.zeros((H, W))
    med[:60] = 0.05                                    # top = sky (dark)
    med[60:] = 0.40                                    # bottom = foreground (bright) → strong horizon edge
    scr = np.zeros((H, W), np.int8)
    scr[10:20, 20:40] = 1                              # rough SKY stroke (top)
    scr[100:110, 20:40] = -1                           # rough EARTH stroke (bottom)
    m = ns.refine_mask(med, scr, auto_mask=None)
    assert m[5, 80] > 0.7 and m[115, 80] < 0.3         # deep sky vs deep foreground
    assert m[55, 80] > m[65, 80]                       # boundary snapped to the edge (~row 60)


def test_refine_mask_correction_stroke_carves_patch():
    pytest.importorskip("skimage")
    H, W = 120, 160
    med = np.zeros((H, W))
    med[:60] = 0.05
    med[60:] = 0.40
    scr = np.zeros((H, W), np.int8)
    scr[10:20, 20:40] = 1
    scr[100:110, 20:40] = -1
    base = ns.refine_mask(med, scr)
    scr2 = scr.copy()
    scr2[15:35, 100:130] = -1                          # force a top (sky) patch to EARTH
    corr = ns.refine_mask(med, scr2)
    assert corr[25, 115] < base[25, 115]               # that patch became more foreground


def test_plan_tracks_moving_foreground_frame():
    # guided/tracked nightscape: static sky, a horizon that DRIFTS (foreground moves). The mask must
    # follow the CHOSEN foreground frame (so it aligns with the composited foreground layer), not the
    # smeared average position.
    H, W, N = 140, 200, 8
    yy, xx = np.mgrid[0:H, 0:W]
    rng = np.random.default_rng(0)
    stars = [(rng.integers(5, H - 5), rng.integers(5, W - 5)) for _ in range(40)]
    frames = []
    for i in range(N):
        img = np.full((H, W), 0.03)
        for (y, x) in stars:
            img[y, x] = 0.9                                  # same stars each frame (sky tracked)
        bnd = 90 + i * 10                                    # horizon drifts right (foreground moves)
        img[xx >= bnd] = (0.35 + 0.15 * np.sin(yy / 3.0))[xx >= bnd]
        frames.append(np.clip(np.stack([img] * 3, axis=-1), 0, 1))
    cube = np.stack(frames)
    best, mask, info = ns.plan(cube, cube.shape[1:3])
    bnd = 90 + best * 10
    assert (mask[:, bnd - 20:bnd - 5] > 0.5).mean() > 0.8    # sky left of the chosen frame's horizon
    assert (mask[:, bnd + 5:bnd + 20] < 0.5).mean() > 0.8    # foreground right of it


def test_plan_override_uses_painted_mask():
    small = np.zeros((4, 40, 50, 3), np.float32) + 0.05
    override = np.zeros((40, 50))
    override[:, :25] = 1.0                              # left = sky, right = foreground
    best, mask, info = ns.plan(small, (80, 100), override_small=override)
    assert info["axis"] == "manual"
    assert mask.shape == (80, 100)                      # upsampled to full res
    assert mask[40, 10] > 0.7 and mask[40, 90] < 0.3   # painted split preserved


def test_decode_workers_bounds():
    from lazystretch.lazystack.run import _decode_workers
    from lazystretch.lazystack.model import LazyStackParams
    assert _decode_workers(LazyStackParams(decode_workers=3), 10) == 3    # explicit wins
    assert _decode_workers(LazyStackParams(decode_workers=1), 10) == 1    # serial
    assert _decode_workers(LazyStackParams(decode_workers=99), 4) == 4    # never exceeds frame count
    w = _decode_workers(LazyStackParams(decode_workers=0), 20)            # auto
    assert 1 <= w <= 6


def test_load_preview_downsamples(tmp_path):
    from lazystretch.io.image_io import load_preview, save_image
    p = tmp_path / "big.tif"
    save_image(str(p), np.linspace(0, 1, 2000 * 2600).reshape(2000, 2600).astype(np.float32), bit_depth=16)
    out = load_preview(str(p), max_dim=800)
    assert max(out.shape[:2]) <= 800 and out.min() >= 0 and out.max() <= 1   # downsampled, normalized


def test_nightscape_brightness_recipe_roundtrip():
    from lazystretch.objects.model import Parameters
    from lazystretch.io import recipes as R
    p = Parameters(); p.nightscapeBrightness = 0.35
    p2 = Parameters(); R.apply_recipe(p2, R.recipe_from_params(p))
    assert p2.nightscapeBrightness == 0.35
