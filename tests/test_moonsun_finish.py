"""LazyMoonSun finish — deterministic invariants (exact look needs real solar data)."""
import numpy as np

from lazystretch.moonsun import finish as fin
from lazystretch.moonsun.model import MoonSunParams
from tests.test_moonsun_register import _disc


def _color_sun(size=256, radius=80, tint=(1.0, 0.85, 0.6)):
    """A limb-darkened disc tinted like a filtered solar frame (kept dim/linear)."""
    lum = _disc(size, radius=radius) * 0.5          # linear-ish, unstretched
    return np.clip(np.stack([lum * t for t in tint], axis=-1), 0, 1)


def test_mtf_midpoint_and_endpoints():
    assert abs(float(fin.mtf(0.5, np.array(0.5)))) - 0.5 < 1e-9
    assert abs(float(fin.mtf(0.3, np.array(0.0)))) < 1e-9
    assert abs(float(fin.mtf(0.3, np.array(1.0))) - 1.0) < 1e-9
    # m < 0.5 brightens the midpoint
    assert float(fin.mtf(0.3, np.array(0.5))) > 0.5


def test_ht_triple_identity():
    img = np.linspace(0, 1, 64 * 64 * 3).reshape(64, 64, 3)
    out = fin._ht(img, 0.0, 0.5, 1.0)
    assert np.allclose(out, np.clip(img, 0, 1), atol=1e-9)


def test_finish_runs_and_classifies_sun():
    res = fin.finish(_color_sun(), MoonSunParams.preset("sun"))
    assert res is not None
    assert res["image"].shape == (256, 256, 3)
    assert np.isfinite(res["image"]).all()
    assert res["image"].min() >= 0 and res["image"].max() <= 1
    assert res["body"] == "sun"                     # limb darkens outward


def test_finish_none_on_black_frame():
    assert fin.finish(np.zeros((128, 128, 3)), MoonSunParams.preset("sun")) is None


def test_finish_full_disc_runs_ring_eraser_without_crashing():
    # a large, fill~1 disc triggers the ring eraser; its bin index must stay in range
    size = 900
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
    disc = np.where(r < 360, 0.45 + 0.1 * np.sqrt(np.clip(1 - (r / 360) ** 2, 0, 1)), 0.0)
    rgb = np.clip(np.stack([disc, disc * 0.85, disc * 0.6], axis=-1), 0, 1)
    res = fin.finish(rgb, MoonSunParams.preset("sun"))
    assert res is not None and res["image"].shape == (size, size, 3)
    assert np.isfinite(res["image"]).all()


def test_wb_neutralizes_tint_neutral_tone():
    p = MoonSunParams.preset("neutral")             # neutral tone, no surface/tone recolor
    res = fin.finish(_color_sun(tint=(1.0, 0.8, 0.55)), p)
    assert res is not None
    img = res["image"]
    disc = img.reshape(-1, 3)
    bright = disc[disc.mean(axis=1) > 0.3]
    r, g, b = bright.mean(axis=0)
    # after white balance the disc should be far closer to neutral than the 1:0.8:0.55 input
    assert abs(r - g) < 0.08 and abs(b - g) < 0.08, (r, g, b)


def test_mtf_lift_raises_disc_median():
    p = MoonSunParams.preset("neutral")             # disc_target 0.60, no tone recolor
    src = _color_sun()
    res = fin.finish(src, p)
    assert res is not None
    lum = res["image"][..., :3].mean(axis=2)
    disc = lum[lum > 0.05]
    assert 0.45 < float(np.median(disc)) < 0.75      # lifted toward the 0.60 target


def test_golden_tone_orders_channels():
    p = MoonSunParams.preset("sun")                  # golden tone on
    res = fin.finish(_color_sun(), p)
    assert res is not None
    img = res["image"]
    lit = img[img[..., :3].mean(axis=2) > 0.2]
    assert (lit[:, 0] >= lit[:, 1] - 1e-6).mean() > 0.95     # R >= G on the disc
    assert (lit[:, 1] >= lit[:, 2] - 1e-6).mean() > 0.95     # G >= B


def test_crop_to_disc_trims_whitespace_with_margin():
    # a small disc in a large frame -> crop removes the surrounding black, keeps a margin
    size = 400
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    r = np.sqrt((xx - 200) ** 2 + (yy - 200) ** 2)
    disc = np.where(r < 60, 0.7, 0.0)
    out = fin.crop_to_disc(disc, pad_frac=0.15)
    assert out.shape[0] < size and out.shape[1] < size    # trimmed
    assert out.shape[0] > 2 * 60 and out.shape[1] > 2 * 60  # disc + margin kept
    assert float(out.max()) > 0.5                          # the disc is still there


def test_finish_crop_shrinks_frame():
    from dataclasses import replace
    src = _color_sun(size=400, radius=70)
    p = replace(MoonSunParams.preset("neutral"), crop=True)
    res = fin.finish(src, p)
    assert res is not None
    assert res["image"].shape[0] < 400 and res["image"].shape[1] < 400
    # without crop the frame keeps its size
    res2 = fin.finish(src, MoonSunParams.preset("neutral"))
    assert res2["image"].shape[:2] == (400, 400)


def test_atrous_sharpen_adds_high_frequency():
    src = _disc(256, radius=80)
    sharp = fin.atrous_sharpen(src, 0.6)
    # high-frequency energy (detail) rises after sharpening
    from lazystretch.moonsun import register as reg
    assert reg.grad_prep(sharp)["q"] > reg.grad_prep(src)["q"]
    assert sharp.shape == src.shape and sharp.min() >= 0 and sharp.max() <= 1
