"""SNR-protect mask: LazyStack noise-map emit + the mask builder that feeds the stretch."""
import numpy as np
import pytest

from lazystretch.lazystack import integrate as integ
from lazystretch.processes import snrmask


def test_sigma_clip_returns_noise_map():
    rng = np.random.default_rng(0)
    # 10 frames: quiet pixel (low σ) vs noisy pixel (high σ) -> noise map must separate them.
    cube = np.stack([np.full((8, 8), 0.3) + rng.normal(0, 0.002, (8, 8)) for _ in range(10)])
    cube[:, 4, 4] += rng.normal(0, 0.05, 10)                  # one high-variance pixel
    out, cov, noise = integ.sigma_clip_mean(cube, return_coverage=True, return_noise=True)
    assert noise.shape == (8, 8) and np.all(cov == 10)
    assert noise[4, 4] > 5 * np.median(noise)                 # the noisy pixel stands out


def test_snr_protect_mask_high_in_noise_low_in_signal():
    H, W = 120, 150
    yy, xx = np.mgrid[0:H, 0:W]
    blob = 0.4 * np.exp(-(((xx - W / 2) / 25) ** 2 + ((yy - H / 2) / 20) ** 2))
    master = np.clip(0.02 + blob, 0, 1)
    noise = np.full((H, W), 0.01, np.float32)                 # uniform σ -> SNR follows the signal
    m = snrmask.snr_protect_mask(master, noise, strength=1.0, smooth=4.0)
    assert m[H // 2, W // 2] < 0.2                            # bright/high-SNR -> not protected
    assert m[5, 5] > 0.6                                      # dark/low-SNR corner -> protected
    assert m.min() >= 0.0 and m.max() <= 1.0


def test_snr_protect_mask_strength_scales():
    rng = np.random.default_rng(1)
    master = rng.uniform(0, 0.3, (64, 80))
    noise = np.full((64, 80), 0.02, np.float32)
    assert snrmask.snr_protect_mask(master, noise, strength=0.5).max() <= 0.5 + 1e-9
    assert np.allclose(snrmask.snr_protect_mask(master, noise, strength=0.0), 0.0)


def test_load_noise_map_roundtrip(tmp_path):
    mp = tmp_path / "lazystack_master.fits"
    mp.write_bytes(b"")                                       # path only; never opened as an image
    assert snrmask.load_noise_map(str(mp)) is None            # absent -> None (graceful fallback)
    np.save(str(tmp_path / "lazystack_master_noise.npy"), np.ones((10, 12), np.float32))
    got = snrmask.load_noise_map(str(mp))
    assert got is not None and got.shape == (10, 12)


def test_snr_protect_mask_folds_coverage():
    master = np.full((60, 80, 3), 0.2)                       # uniform SNR -> SNR protect ~0
    noise = np.full((60, 80), 0.02, np.float32)
    cov_full = np.full((60, 80), 20, np.int32)
    cov_partial = cov_full.copy()
    cov_partial[:, :10] = 3                                   # left columns poorly covered
    m_full = snrmask.snr_protect_mask(master, noise, strength=1.0, coverage=cov_full, smooth=1)
    m_part = snrmask.snr_protect_mask(master, noise, strength=1.0, coverage=cov_partial, smooth=1)
    assert m_part[:, :10].mean() > m_full[:, :10].mean() + 0.2   # low frame-support -> more protection


def test_load_coverage_map_roundtrip(tmp_path):
    mp = tmp_path / "lazystack_master.fits"
    mp.write_bytes(b"")
    assert snrmask.load_coverage_map(str(mp)) is None
    np.save(str(tmp_path / "lazystack_master_coverage.npy"), np.full((8, 9), 20, np.int32))
    got = snrmask.load_coverage_map(str(mp))
    assert got is not None and got.shape == (8, 9)


def _osc(H=200, W=260, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    gas = 0.30 * np.exp(-(((xx - W / 2) / 60) ** 2 + ((yy - H / 2) / 45) ** 2))
    L = np.clip(0.06 + gas + rng.normal(0, 0.006, (H, W)), 0, 1)
    return np.stack([L, L * 0.82, L * 0.7], axis=-1)


def test_runcore_snr_protect_active_with_matching_map():
    from lazystretch.objects.model import Parameters
    from lazystretch.pipeline.runcore import run_pipeline
    img = _osc()
    p = Parameters.for_object("milkyway", snrProtect=0.8)
    p.snr_noise_map = np.full(img.shape[:2], 0.01, np.float32)
    res = run_pipeline(img, p, preview=False)
    assert any("SNR-protect mask" in s for s in res.steps_run)


def test_runcore_snr_protect_disabled_on_shape_mismatch():
    from lazystretch.objects.model import Parameters
    from lazystretch.pipeline.runcore import run_pipeline
    img = _osc()
    p = Parameters.for_object("milkyway", snrProtect=0.8)
    p.snr_noise_map = np.full((5, 5), 0.01, np.float32)       # wrong shape -> gracefully skipped
    res = run_pipeline(img, p, preview=False)
    assert not any("SNR-protect mask" in s for s in res.steps_run)


def test_runcore_snr_protect_off_by_default():
    from lazystretch.objects.model import Parameters
    from lazystretch.pipeline.runcore import run_pipeline
    img = _osc()
    p = Parameters.for_object("milkyway")                     # snrProtect defaults 0
    p.snr_noise_map = np.full(img.shape[:2], 0.01, np.float32)
    res = run_pipeline(img, p, preview=False)
    assert not any("SNR-protect mask" in s for s in res.steps_run)
