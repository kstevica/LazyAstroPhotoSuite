"""Transient (meteor) preservation: integration transient emit + meteor detection."""
from pathlib import Path

import numpy as np
import pytest

from lazystretch.io.image_io import save_image
from lazystretch.lazystack import calibrate as cal, integrate as integ, meteors, run as lsrun
from lazystretch.lazystack.model import LazyStackParams


def _sky_stack(H=300, W=420, N=8, seed=0):
    rng = np.random.default_rng(seed)
    return [np.clip(0.05 + rng.normal(0, 0.003, (H, W, 3)), 0, 1) for _ in range(N)]


def _add_streak(frame, y0, x0, n, dy, dx, amp, green=0.0):
    rr = np.arange(n)
    ys = np.clip(y0 + rr * dy, 0, frame.shape[0] - 1)
    xs = np.clip(x0 + rr * dx, 0, frame.shape[1] - 1)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            yy = np.clip(ys + oy, 0, frame.shape[0] - 1)
            xx = np.clip(xs + ox, 0, frame.shape[1] - 1)
            frame[yy, xx] += amp
    frame[ys, xs, 1] += green            # a greenish core (meteor colour to preserve)
    return ys, xs


def test_transient_emit_and_meteor_detection():
    frames = _sky_stack()
    ys, xs = _add_streak(frames[3], 40, 60, 55, 1, 3, 0.5, green=0.2)   # meteor in ONE frame
    frames[5][100:104, 200:204] += 0.6                                  # compact CR (must be rejected)
    cube = np.stack([f.astype(np.float64) for f in frames])
    out, transient, tframe = integ.sigma_clip_mean(cube, return_transient=True)

    assert transient.shape == frames[0].shape and tframe.shape == (300, 420)
    assert float(transient[ys[25], xs[25]].max()) > 0.2          # the meteor light is preserved
    assert transient[..., 1].max() > 0                           # colour kept (green channel)

    met, soft, labels = meteors.detect_meteors(transient, tframe)
    assert len(met) == 1                                         # the streak, not the CR blob
    assert met[0]["frame"] == 3 and met[0]["id"] == 1 and met[0]["aspect"] > 3 and met[0]["length"] > 50
    assert soft[ys[25], xs[25]] > 0.3                            # soft mask covers the trail
    assert soft[102, 202] < 0.1                                  # CR blob not preserved
    assert labels[ys[25], xs[25]] == 1 and labels[102, 202] == 0  # id map marks the trail only


def test_combine_files_returns_transient(tmp_path):
    frames = _sky_stack(H=300, W=420, N=6)
    _add_streak(frames[2], 20, 20, 40, 1, 3, 0.5)
    paths = []
    for i, f in enumerate(frames):
        p = tmp_path / f"r_{i}.npy"
        np.save(str(p), f.astype(np.float32))
        paths.append(str(p))
    master, transient, tframe = integ.combine_files(paths, return_transient=True)
    assert master.shape == frames[0].shape
    assert transient.shape == frames[0].shape and tframe.shape == (300, 420)
    met, _, _ = meteors.detect_meteors(transient, tframe)
    assert len(met) == 1 and met[0]["frame"] == 2


def test_stack_writes_meteor_layer(tmp_path):
    pytest.importorskip("photutils")
    rng = np.random.default_rng(3)
    H, W = 320, 440
    yy, xx = np.mgrid[0:H, 0:W]
    stars = np.zeros((H, W))
    for _ in range(45):
        cy, cx = rng.integers(12, H - 12), rng.integers(12, W - 12)
        stars += rng.uniform(0.4, 0.9) * np.exp(-(((xx - cx) / 1.6) ** 2 + ((yy - cy) / 1.6) ** 2))
    lights = tmp_path / "lights"
    lights.mkdir()
    for i in range(6):
        f = np.clip(0.03 + stars + rng.normal(0, 0.003, (H, W)), 0, 1)
        rgb = np.stack([f, f, f], axis=-1)
        if i == 2:                                            # meteor trail in ONE frame, greenish
            rr = np.arange(70)
            ys = np.clip(30 + rr, 0, H - 1)
            xs = np.clip(20 + rr * 2, 0, W - 1)
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    rgb[np.clip(ys + oy, 0, H - 1), np.clip(xs + ox, 0, W - 1)] += 0.5
            rgb[ys, xs, 1] += 0.25
        save_image(str(lights / f"l_{i:03d}.tif"), np.clip(rgb, 0, 1), bit_depth=16)
    p = LazyStackParams(do_calibrate=False, do_cosmetic=False, fix_walking_noise=False,
                        preserve_meteors=True, stage_to_disk=True, reuse_cache=False)
    res = lsrun.stack(str(tmp_path), p)
    assert res is not None and res["meteor_layer_path"] and len(res["meteors"]) >= 1
    layer = np.load(res["meteor_layer_path"])
    assert layer.ndim == 3 and layer.shape[:2] == res["master"].shape[:2]
    assert layer[..., 1].max() > 0.05                        # green meteor colour preserved
    m0 = res["meteors"][0]
    assert m0.get("source") and m0["source"].endswith(".tif") and m0["id"] == 1   # source filename recorded
    assert Path(res["master_path"]).with_name("lazystack_master_meteorlabels.npy").exists()


def test_capture_time_from_fits_and_mtime(tmp_path):
    from lazystretch.io.image_io import capture_time, save_image
    # FITS DATE-OBS header is used when present
    fp = tmp_path / "light.fits"
    save_image(str(fp), np.zeros((8, 8), np.float32), bit_depth=16,
               header={"DATE-OBS": "2026-08-09T01:25:33"})
    assert capture_time(str(fp)) == "2026-08-09T01:25:33"
    # unknown format falls back to a (non-None) mtime string
    tp = tmp_path / "x.tif"
    save_image(str(tp), np.zeros((8, 8), np.float32), bit_depth=16)
    assert capture_time(str(tp))


def test_taper_ends_fades_trail_ends_not_middle():
    # a long horizontal trail, uniform developed amplitude
    H, W = 60, 800
    labels = np.zeros((H, W), dtype=np.int16)
    labels[29:32, 40:760] = 1
    dev = np.zeros((H, W, 3), np.float64)
    dev[labels > 0] = [0.2, 0.5, 0.2]                       # green-dominant, constant
    dev_before = dev.copy()
    out = meteors.taper_ends(dev, labels)
    mid = out[30, 400]
    near_end = out[30, 45]                                  # a few px inside the start
    assert mid[1] > 0.49                                    # middle unchanged (~full)
    assert near_end[1] < 0.15                               # end faded down
    assert out[30, 41, 1] < near_end[1]                    # monotonic toward the tip
    # hue preserved everywhere the trail survives (same green:red ratio as input 0.5:0.2)
    m = out[..., 1] > 1e-3
    assert np.allclose(out[..., 0][m] / out[..., 1][m], 0.2 / 0.5, atol=1e-6)
    assert not np.shares_memory(out, dev)                   # returns a copy...
    assert np.array_equal(dev, dev_before)                  # ...and never mutates the input


def test_taper_ends_multi_trail_independent():
    H, W = 120, 600
    labels = np.zeros((H, W), dtype=np.int16)
    labels[20:23, 30:560] = 1                               # long trail
    labels[90:93, 250:330] = 2                              # short trail
    dev = np.zeros((H, W, 3), np.float64)
    dev[labels == 1] = 0.4
    dev[labels == 2] = 0.4
    out = meteors.taper_ends(dev, labels)
    assert out[21, 295, 0] > 0.39                           # trail 1 middle untouched by trail 2's fade
    assert out[21, 32, 0] < 0.2 and out[21, 557, 0] < 0.2   # trail 1 both ends fade
    assert out[91, 290, 0] > 0.0                            # trail 2 has a bright-ish centre
    assert out[91, 251, 0] < out[91, 290, 0]                # trail 2 fades at its own end


def test_taper_ends_skips_field_of_view_cut():
    # a trail whose LEFT tip sits on the image border (a FOV cut, not a burn-out)
    H, W = 80, 400
    labels = np.zeros((H, W), dtype=np.int16)
    labels[39:42, 0:300] = 1                                # x starts at 0 = left edge
    dev = np.zeros((H, W, 3), np.float64)
    dev[labels > 0] = 0.4
    out = meteors.taper_ends(dev, labels)
    assert out[40, 2, 0] > 0.39                             # border end NOT tapered (kept full)
    assert out[40, 297, 0] < 0.2                            # interior end still fades


def test_taper_ends_dim_end_fades_longer():
    # bright left half, dim right half -> the dim (right) end fades over a LONGER length
    H, W = 120, 600
    labels = np.zeros((H, W), dtype=np.int16)
    labels[59:62, 30:560] = 1
    dev = np.zeros((H, W, 3), np.float64)
    dev[59:62, 30:295] = 0.6                                # bright half
    dev[59:62, 295:560] = 0.15                              # dim half
    out = meteors.taper_ends(dev, labels)
    env_bright = out[60, 30 + 40, 0] / 0.6                  # fade fraction 40 px in from the bright tip
    env_dim = out[60, 559 - 40, 0] / 0.15                   # 40 px in from the dim tip
    assert env_dim < env_bright - 0.2                       # dim end is less risen at the same offset
    assert out[60, 100, 0] / 0.6 > 0.95                     # bright-side middle env preserved
    assert out[60, 295, 0] / 0.15 > 0.95                    # dim-side middle env preserved


def test_selection_layer_masks_by_id():
    labels = np.zeros((50, 60), dtype=np.int16)
    labels[10:12, 5:40] = 1
    labels[30:32, 5:40] = 2
    layer = np.zeros((50, 60, 3), np.float32)
    layer[labels > 0] = 0.4
    both = meteors.selection_layer(layer, labels, None)
    only1 = meteors.selection_layer(layer, labels, [1])
    assert both[10, 20, 0] > 0 and both[30, 20, 0] > 0       # None -> all trails
    assert only1[10, 20, 0] > 0 and only1[30, 20, 0] == 0    # [1] -> only trail 1


def test_protect_extended_keeps_small_reverts_large():
    orig = np.full((60, 80), 0.1)
    orig[5, 5] = 0.9                                          # a hot pixel (small change)
    sy = np.clip(10 + np.arange(40), 0, 59); sx = np.clip(20 + np.arange(40), 0, 79)
    orig[sy, sx] = 0.8                                        # a trail (large change)
    cleaned = orig.copy()
    cleaned[5, 5] = 0.1                                       # both "repaired" by a CR detector
    cleaned[sy, sx] = 0.1
    out = cal._protect_extended(orig, cleaned, max_component=12)
    assert out[5, 5] < 0.2                                    # small repair accepted
    assert out[sy[20], sx[20]] > 0.6                         # large trail reverted (preserved)


def test_cosmetic_correct_preserves_meteor_trail():
    rng = np.random.default_rng(0)
    H, W = 100, 140
    frame = np.clip(0.10 + rng.normal(0, 0.003, (H, W)), 0, 1)
    rr = np.arange(60)
    ys = np.clip(20 + rr, 0, H - 1); xs = np.clip(10 + rr * 2, 0, W - 1)
    frame[ys, xs] = 0.8
    assert cal.cosmetic_correct(frame)[ys[30], xs[30]] > 0.6   # trail survives to integration


def test_stretch_composites_meteor_layer():
    from lazystretch.objects.model import Parameters
    from lazystretch.pipeline.runcore import run_pipeline
    rng = np.random.default_rng(1)
    H, W = 180, 240
    img = np.stack([np.clip(0.05 + rng.normal(0, 0.005, (H, W)), 0, 1) for _ in range(3)], axis=-1)
    layer = np.zeros((H, W, 3), np.float32)                  # a bright green meteor streak
    rr = np.arange(60)
    ys = np.clip(30 + rr, 0, H - 1); xs = np.clip(20 + rr * 3, 0, W - 1)
    layer[ys, xs, 1] = 0.4
    p = Parameters.for_object("milkyway", meteorStrength=1.0, autoAssess=False, cropPercent=0.0)
    p.meteor_layer = layer
    on = run_pipeline(img, p, preview=True)
    off = run_pipeline(img, Parameters.for_object("milkyway", meteorStrength=0.0,
                                                  autoAssess=False, cropPercent=0.0), preview=True)
    assert any("meteor" in s.lower() for s in on.steps_run)
    # the trail is brighter with the composite on, and its green survives
    ty, tx = int(ys[30]), int(xs[30])
    assert on.image[ty, tx, 1] > off.image[ty, tx, 1] + 0.1
    assert on.image[ty, tx, 1] > on.image[ty, tx, 0]         # green-dominant (colour preserved)
