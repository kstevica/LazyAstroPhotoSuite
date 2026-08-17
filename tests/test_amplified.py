"""Experimental amplified-signal stacking — unit + end-to-end tests (hermetic, synthetic)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from lazystretch.io.image_io import load_image, save_image
from lazystretch.lazystack import amplified as amp
from lazystretch.lazystack import integrate as integ
from lazystretch.lazystack import measure as meas
from lazystretch.lazystack import run as lsrun
from lazystretch.lazystack.model import LazyStackParams
from lazystretch.moonsun import register as fftreg


def _starfield(h=128, w=128, n=40, seed=0, star_sigma=0.9, noise=0.004):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 0.06)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n):
        cy, cx = rng.uniform(12, h - 12), rng.uniform(12, w - 12)
        a = rng.uniform(0.2, 0.8)
        img += a * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * star_sigma ** 2)))
    img += rng.normal(0, noise, (h, w))
    return np.clip(img, 0, 1)


# ------------------------------------------------------------------ integration core

def test_second_weighted_mean_returned_after_primary():
    a = np.full((8, 8), 0.2)
    b = np.full((8, 8), 0.4)
    a[0, 0] = np.nan                                        # no-data in frame A
    out, out_hi = integ.sigma_clip_mean(np.stack([a, b]), weights=[1, 1], weights_hi=[1, 0])
    assert np.allclose(out[1, 1], 0.3, atol=1e-6)
    assert np.allclose(out_hi[1, 1], 0.2, atol=1e-6)        # hi mean = frame A only
    assert np.allclose(out[0, 0], 0.4, atol=1e-6)           # A is no-data there
    assert np.allclose(out_hi[0, 0], 0.4, atol=1e-6)        # hi falls back, not NaN


def test_model_floor_stops_overclipping_flat_sky():
    # 10 frames at exactly 0.25 → MAD 0; two honest 0.26 samples get clipped without the
    # photon-transfer floor, and survive with it (σ_model ≈ 0.05 ≫ the 0.01 deviation).
    cube = np.stack([np.full((4, 4), 0.25)] * 10 + [np.full((4, 4), 0.26)] * 2)
    plain = integ.sigma_clip_mean(cube)
    floored = integ.sigma_clip_mean(cube, model_ab=(0.0, 0.05 ** 2))
    assert np.allclose(plain, 0.25, atol=1e-4)
    assert float(floored[0, 0]) > float(plain[0, 0]) + 5e-4


def test_model_floor_capped_above_bright_structure():
    # Above m_cap the fit regime doesn't apply: the floor must switch off so a satellite
    # trail over bright nebulosity still gets clipped (and reaches the transient map).
    cube = np.stack([np.full((4, 4), 0.25)] * 10 + [np.full((4, 4), 0.26)] * 2)
    capped = integ.sigma_clip_mean(cube, model_ab=(0.0, 0.05 ** 2, 0.20))   # cap < signal
    plain = integ.sigma_clip_mean(cube)
    assert np.allclose(capped, plain, atol=1e-6)            # floor inert above the cap


def test_photon_transfer_recovers_slope():
    a_true, b_true = 4e-5, 4e-8
    rng = np.random.default_rng(3)
    base = np.tile(np.linspace(0.02, 0.5, 256), (64, 1))
    handles = [np.clip(base + rng.normal(0, np.sqrt(a_true * base + b_true)), 0, 1)
               .astype(np.float32) for _ in range(16)]
    model = amp.photon_transfer(handles)
    assert model is not None
    assert 0.5 * a_true < model["a"] < 2.0 * a_true
    assert model["gain_full_scale"] == pytest.approx(1.0 / model["a"])
    assert 0.0 < model["m_cap"] <= 0.5                      # bright-structure cap present


def test_photon_transfer_envelope_resists_gradient_contamination():
    # Star-halo pixels carry registration-jitter variance ≫ photon noise; the lower-
    # envelope fit (bins ≤ p80, 25th-pct variance) must not let them inflate the slope.
    a_true, b_true = 4e-5, 4e-8
    rng = np.random.default_rng(11)
    base = np.tile(np.linspace(0.02, 0.4, 256), (64, 1))
    halo = rng.random(base.shape) < 0.3                     # sparse halos, as in real fields
    handles = []
    for _ in range(16):
        f = base + rng.normal(0, np.sqrt(a_true * base + b_true))
        hot = (base > 0.25) & halo                          # 20× excess scatter there
        f[hot] += rng.normal(0, 0.02, size=int(hot.sum()))
        handles.append(np.clip(f, 0, 1).astype(np.float32))
    model = amp.photon_transfer(handles)
    assert model is not None
    assert model["a"] < 5.0 * a_true                        # inflation bounded, not ×100


def test_photon_transfer_needs_frames_and_range():
    flat = [np.full((64, 256), 0.2, dtype=np.float32) for _ in range(16)]
    assert amp.photon_transfer(flat) is None                # no signal range → no fit
    assert amp.photon_transfer(flat[:3]) is None            # too few frames


# ---------------------------------------------------------------------- band weights

def test_band_weights_downweight_soft_frames_high_band_only():
    ms = [{"noise": 0.01, "fwhm": 2.0}, {"noise": 0.01, "fwhm": 2.1},
          {"noise": 0.01, "fwhm": 4.0}]
    w = amp.band_weights(ms, [0, 1, 2])
    lo, hi = np.asarray(w["w_lo"]), np.asarray(w["w_hi"])
    assert lo[2] == pytest.approx(lo[0])                    # low band: full membership
    assert hi[2] / hi[0] < 0.35                             # high band: MTF-suppressed
    n = [{"noise": 0.02, "fwhm": 2.0}, {"noise": 0.01, "fwhm": 2.0}]
    w2 = amp.band_weights(n, [0, 1])
    assert w2["w_lo"][1] / w2["w_lo"][0] == pytest.approx(4.0, rel=1e-3)   # 1/σ²


def test_band_merge_detail_from_sharp_depth_from_all():
    lo = np.full((64, 64), 0.3)
    yy, xx = np.mgrid[0:64, 0:64]
    lo = lo + 0.1 * np.exp(-(((yy - 20.0) ** 2 + (xx - 20.0) ** 2) / (2 * 8.0 ** 2)))
    hi = np.full((64, 64), 0.3)
    hi[40, 40] = 0.8                                        # sharp star only in the hi mean
    merged = amp.band_merge(lo, hi, sigma_px=2.0)
    assert merged[40, 40] > 0.6                             # detail came from the sharp mean
    assert merged[20, 20] > 0.36                            # broad faint blob survived (low band)
    assert abs(merged[5, 60] - 0.3) < 0.02                  # background untouched


# --------------------------------------------------------------------- static pattern

def test_static_pattern_requires_dither():
    frames = [np.full((32, 32), 0.2) for _ in range(8)]
    assert amp.static_pattern(None, frames, list(range(8)),
                              fwhm_px=2.0, dither_px=0.5) is None


def test_static_pattern_recovers_fixed_structure():
    rng = np.random.default_rng(5)
    h = w = 96
    sky = _starfield(h, w, n=12, seed=9, noise=0.0)
    pattern = np.zeros((h, w))
    pattern[:, ::7] = 0.03                                  # fixed high-freq stripes
    frames = []
    for i in range(9):
        shifted = np.roll(sky, (i * 5) % 30, axis=1)        # the sky moves with the dither
        frames.append(np.clip(shifted + pattern + rng.normal(0, 0.004, (h, w)), 0, 1))
    p = amp.static_pattern(None, frames, list(range(9)), fwhm_px=2.0, dither_px=6.0)
    assert p is not None
    corr = np.corrcoef(p.ravel(), pattern.ravel())[0, 1]
    assert corr > 0.5                                       # the stripes were found
    fixed = amp.subtract_pattern(frames[0], p)              # frame 0 has zero roll → sky is truth
    assert np.std(fixed - sky) < np.std(frames[0] - sky)    # residual shrank after subtraction


def test_static_pattern_spares_large_scale_structure():
    # Structure LARGER than the dither (broad nebulosity, sensor-static to the median)
    # must not enter the pattern — the highpass is band-limited to provable scales.
    rng = np.random.default_rng(6)
    h = w = 96
    yy, xx = np.mgrid[0:h, 0:w]
    blob = 0.08 * np.exp(-(((yy - 48.0) ** 2 + (xx - 48.0) ** 2) / (2 * 12.0 ** 2)))
    stars = _starfield(h, w, n=10, seed=3, noise=0.0) - 0.06   # star signal only
    frames = []
    for i in range(9):
        moving = np.roll(stars, (i * 5) % 30, axis=1)          # stars dither; blob is static
        frames.append(np.clip(0.06 + moving + blob + rng.normal(0, 0.004, (h, w)), 0, 1))
    p = amp.static_pattern(None, frames, list(range(9)), fwhm_px=2.0, dither_px=6.0)
    if p is not None:                                          # None is also a pass
        assert abs(float(p[48, 48])) < 0.008                   # blob crest ~0.08: >90% spared


# ------------------------------------------------------------------------------ cull

def _m(fwhm=2.0, ecc=0.3, bg=0.10, stars=50, noise=0.01, snr=10.0):
    return {"stars": stars, "fwhm": fwhm, "ecc": ecc, "bg": bg, "noise": noise, "snr": snr}


def test_cull_amplified_keeps_soft_rejects_trailed():
    ms = [_m() for _ in range(6)] + [_m(fwhm=5.0), _m(ecc=0.9)]
    classic = meas.cull(ms, LazyStackParams())
    assert 6 not in classic["keep"] and 7 not in classic["keep"]
    ampd = meas.cull(ms, LazyStackParams(amplified=True))
    assert 6 in ampd["keep"]                                # soft frame kept for the low band
    assert 7 not in ampd["keep"]                            # trailed still hard-rejected
    assert 6 in ampd["soft_kept"] and "kept" in ampd["soft_kept"][6]


# ------------------------------------------------------------------------ end to end

def test_stack_amplified_end_to_end(tmp_path):
    pytest.importorskip("photutils")
    lights = tmp_path / "lights"
    lights.mkdir()
    base = _starfield(n=40, seed=7)
    for i in range(8):
        f = fftreg.apply_shift(base, i * 1.5 - 5, -i * 1.2 + 4)
        if i == 7:
            f = gaussian_filter(f, 2.2)                     # one soft frame
        save_image(str(lights / f"light_{i:03d}.fits"), np.clip(f, 0, 1), bit_depth=16)
    logs: list = []
    p = LazyStackParams(do_calibrate=False, do_cosmetic=False, stage_to_disk=False,
                        amplified=True)
    res = lsrun.stack(str(tmp_path), p, log=lambda s: logs.append(s))
    assert res is not None
    assert res["amplified"] is not None
    assert res["n_stacked"] >= 7                            # the soft frame was not culled
    loaded = load_image(str(tmp_path / "lazystack" / lsrun.MASTER_NAME))
    assert int(loaded.keyword("LZSAMP")) == 1
    assert int(loaded.keyword("LZSGRID")) in (1, 2)
    meta_path = tmp_path / "lazystack" / "amplified_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert "grid" in meta and "dither_px" in meta
    assert any("AMPLIFIED" in s for s in logs)
