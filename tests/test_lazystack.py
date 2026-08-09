"""LazyStack — calibration, integration, measure/cull, register, contract, run."""
import numpy as np
import pytest

from lazystretch.io.image_io import load_image, save_image
from lazystretch.lazystack import calibrate as cal, contract, integrate as integ
from lazystretch.lazystack import measure as meas, register as reg, run as lsrun
from lazystretch.lazystack.model import LazyStackParams
from lazystretch.moonsun import register as fftreg
from tests.test_moonsun_register import _disc


def _starfield(H=200, W=240, n=30, seed=0, trail=False):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(0.03, 0.004, (H, W)), 0, 1)
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(n):
        cy, cx = rng.integers(15, H - 15), rng.integers(15, W - 15)
        sx, sy = (5.0, 1.6) if trail else (1.6, 1.6)     # trail = elongated PSF
        img += rng.uniform(0.4, 0.9) * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    return np.clip(img, 0, 1)


# --- calibration + integration ---

def test_sigma_clip_rejects_outliers():
    rng = np.random.default_rng(0)
    inliers = [np.clip(0.4 + rng.normal(0, 0.003, (16, 16)), 0, 1) for _ in range(12)]
    cube = np.stack(inliers + [np.full((16, 16), 0.95)])      # one hot outlier frame
    out = integ.sigma_clip_mean(cube, sigma_low=3, sigma_high=3)
    assert np.allclose(out, 0.4, atol=0.01)                   # outlier clipped away


def test_combine_files_matches_in_memory(tmp_path):
    rng = np.random.default_rng(0)
    frames = [np.clip(rng.random((40, 50, 3)), 0, 1).astype(np.float32) for _ in range(6)]
    paths = []
    for i, f in enumerate(frames):
        p = tmp_path / f"f_{i}.npy"
        np.save(str(p), f)
        paths.append(p)
    streamed = integ.combine_files(paths, sigma_low=4, sigma_high=3)
    in_mem = integ.integrate(frames, sigma_low=4, sigma_high=3)
    assert streamed.shape == (40, 50, 3)
    assert np.allclose(streamed, in_mem, atol=1e-5)       # bounded path == in-memory result


def test_combine_files_row_banding(tmp_path):
    # force multiple bands (tiny target) and confirm the result is unchanged
    rng = np.random.default_rng(1)
    frames = [np.clip(rng.random((64, 20, 3)), 0, 1).astype(np.float32) for _ in range(5)]
    paths = [tmp_path / f"g_{i}.npy" for i in range(5)]
    for p, f in zip(paths, frames):
        np.save(str(p), f)
    banded = integ.combine_files(paths, target_bytes=4000)   # ~1-2 rows per band
    full = integ.integrate(frames)
    assert np.allclose(banded, full, atol=1e-5)


def test_calibrate_light_subtracts_dark_and_divides_flat():
    signal = np.full((10, 10), 0.30)
    dark = np.full((10, 10), 0.05)
    flat = np.full((10, 10), 0.5)                              # uniform flat -> no-op after norm
    out = cal.calibrate_light(signal + dark, dark=dark, flat=flat)
    assert np.allclose(out, 0.30, atol=1e-6)


def test_integrate_reduces_noise():
    rng = np.random.default_rng(1)
    truth = _disc(120, radius=40)
    frames = [np.clip(truth + rng.normal(0, 0.05, truth.shape), 0, 1) for _ in range(12)]
    out = integ.integrate(frames)
    single = np.abs(frames[0] - truth).std()
    stacked = np.abs(out - truth).std()
    assert stacked < 0.5 * single


# --- measure + cull ---

def test_measure_and_cull_rejects_trailed():
    pytest.importorskip("photutils")
    frames = [_starfield(seed=i) for i in range(5)]
    frames.append(_starfield(seed=99, trail=True))            # a trailed frame
    measures = [meas.measure_frame(f) for f in frames]
    assert all(m is not None for m in measures)
    culled = meas.cull(measures, LazyStackParams(ecc_hard=0.5))
    assert 5 in culled["rejected"]                            # the trailed frame rejected
    assert 5 not in culled["keep"]
    assert culled["reference"] in culled["keep"]


# --- registration (FFT fallback) ---

def test_register_fft_fallback_aligns_shifted_frames(monkeypatch):
    # force the translation-only FFT fallback (deterministic, no astroalign needed)
    monkeypatch.setattr(reg, "astroalign_available", lambda: False)
    base = _disc(200, radius=60)
    frames = [base,
              fftreg.apply_shift(base, 4.0, -3.0),
              fftreg.apply_shift(base, -5.0, 2.0)]
    aligned, kept = reg.register(frames, reference=0)
    assert len(kept) == 3
    core = slice(60, 140)
    for a in aligned:
        assert np.abs(a[core, core] - base[core, core]).mean() < 0.03


def test_register_astroalign_aligns_star_field():
    pytest.importorskip("astroalign")
    if not reg.astroalign_available():
        pytest.skip("astroalign not importable")
    base = _starfield(H=300, W=300, n=60, seed=2)
    frames = [base, fftreg.apply_shift(base, 6.0, -4.0)]
    aligned, kept = reg.register(frames, reference=0)
    assert len(kept) == 2
    core = slice(60, 240)
    assert np.abs(aligned[1][core, core] - base[core, core]).mean() < 0.02


# --- contract ---

def test_measure_edges_finds_dark_border_and_floor():
    master = np.full((200, 200), 0.4)
    master[:12, :] = 0.0                                       # 12px junk top border
    edges = contract.measure_edges(master)
    assert edges["T"] >= 12
    assert edges["L"] >= int(0.02 * 200)                      # 2% safety floor everywhere
    hdr = contract.contract_header(20, edges, 600.0)
    assert hdr["LZSNSUB"] == 20 and hdr["LZSVER"] == "0.3.0"


# --- end-to-end run ---

def test_stack_folder_writes_master_with_contract(tmp_path):
    pytest.importorskip("photutils")
    lights = tmp_path / "lights"
    lights.mkdir()
    base = _starfield(n=40, seed=7)
    for i in range(6):
        f = fftreg.apply_shift(base, i * 1.5 - 3, -i * 1.0 + 2)
        save_image(str(lights / f"light_{i:03d}.fits"), np.clip(f, 0, 1), bit_depth=16)
    res = lsrun.stack(str(tmp_path), LazyStackParams(do_calibrate=False, do_cosmetic=False))
    assert res is not None
    master_path = tmp_path / "lazystack" / lsrun.MASTER_NAME
    assert master_path.exists()
    loaded = load_image(str(master_path))
    assert int(loaded.keyword("LZSNSUB")) == res["n_stacked"] >= 2
    assert loaded.keyword("LZSCROPL") is not None


def test_stack_in_memory_mode_creates_no_work_files(tmp_path):
    pytest.importorskip("photutils")
    lights = tmp_path / "lights"
    lights.mkdir()
    base = _starfield(n=40, seed=11)
    for i in range(6):
        f = fftreg.apply_shift(base, i * 1.2 - 2, -i * 0.8 + 1)
        save_image(str(lights / f"light_{i:03d}.fits"), np.clip(f, 0, 1), bit_depth=16)
    params = LazyStackParams(do_calibrate=False, do_cosmetic=False, stage_to_disk=False)
    res = lsrun.stack(str(tmp_path), params)
    assert res is not None and res["n_stacked"] >= 2
    assert (tmp_path / "lazystack" / lsrun.MASTER_NAME).exists()
    assert not (tmp_path / "lazystack" / "work").exists()   # in-memory -> no work files


def test_stack_reuses_existing_work_files(tmp_path):
    pytest.importorskip("photutils")
    lights = tmp_path / "lights"
    lights.mkdir()
    base = _starfield(n=40, seed=21)
    for i in range(6):
        f = fftreg.apply_shift(base, i - 2, -i + 1)
        save_image(str(lights / f"l_{i:03d}.fits"), np.clip(f, 0, 1), bit_depth=16)
    p = LazyStackParams(do_calibrate=False, do_cosmetic=False,
                        stage_to_disk=True, reuse_cache=True)
    assert lsrun.stack(str(tmp_path), p) is not None
    work = tmp_path / "lazystack" / "work"
    assert work.exists()                                  # kept for reuse (not cleaned up)
    reg_files = list(work.glob("reg_*.npy"))
    assert reg_files
    mtimes = {f.name: f.stat().st_mtime_ns for f in reg_files}

    logs = []
    assert lsrun.stack(str(tmp_path), p, log=lambda s: logs.append(s)) is not None
    assert any("reusing cached registration" in s for s in logs)   # 2nd run reused
    for f in work.glob("reg_*.npy"):
        assert f.stat().st_mtime_ns == mtimes.get(f.name)          # not rewritten


def test_measure_only_advises_without_stacking(tmp_path):
    pytest.importorskip("photutils")
    lights = tmp_path / "lights"
    lights.mkdir()
    for i in range(4):
        save_image(str(lights / f"l_{i}.fits"), _starfield(seed=i), bit_depth=16)
    res = lsrun.measure_only(str(tmp_path), LazyStackParams())
    assert res is not None and len(res["measures"]) == 4
    assert "keep" in res["cull"]
    assert not (tmp_path / "lazystack").exists()               # nothing stacked
