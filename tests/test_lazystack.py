"""LazyStack — calibration, integration, measure/cull, register, contract, run."""
from pathlib import Path

import numpy as np
import pytest

from lazystretch.io.image_io import load_image, save_image
from lazystretch.lazystack import calibrate as cal, contract, integrate as integ
from lazystretch.lazystack import measure as meas, normalize as nrm, register as reg, run as lsrun
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


def test_normalize_matches_background_and_scale():
    ref = _disc(120, radius=40) * 0.5 + 0.10          # bg ~0.10
    ref_med, ref_sig = nrm.frame_stats(ref)
    bright_bg = ref + 0.06                             # sky drifted up
    out = nrm.normalize_to_ref(bright_bg, ref_med, ref_sig)
    assert abs(float(np.median(out)) - ref_med) < 0.01    # background matched back
    dim = (ref - 0.10) * 0.5 + 0.10                    # transparency loss (half signal)
    out2 = nrm.normalize_to_ref(dim, ref_med, ref_sig)
    m2, s2 = nrm.frame_stats(out2)
    assert abs(s2 - ref_sig) < 0.05 * ref_sig + 1e-3      # signal scale restored


def test_normalize_reference_is_identity():
    ref = _disc(120, radius=40) * 0.6 + 0.05
    ref_med, ref_sig = nrm.frame_stats(ref)
    out = nrm.normalize_to_ref(ref, ref_med, ref_sig)
    assert np.allclose(out, ref, atol=1e-9)


def test_local_normalize_removes_spatial_gradient():
    rng = np.random.default_rng(3)
    ref = np.clip(rng.normal(0.1, 0.01, (256, 256)), 0, 1)     # flat-ish reference
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float64)
    gradient = 0.15 * (xx / 255.0) + 0.10 * (yy / 255.0)       # a ramp global norm can't fix
    frame = np.clip(ref + gradient, 0, 1)
    ref_med, ref_sig = nrm.frame_stats(ref)

    glob = nrm.normalize_to_ref(frame, ref_med, ref_sig)       # single offset — ramp remains
    loc = nrm.local_normalize_to_ref(frame, ref, ref_med, ref_sig)
    # residual low-frequency gradient vs the reference: LN should be far smaller than global
    glob_grad = float(np.abs(nrm._low_frequency(glob - ref)).mean())
    loc_grad = float(np.abs(nrm._low_frequency(loc - ref)).mean())
    assert loc_grad < 0.2 * glob_grad
    assert loc.shape == frame.shape


def test_low_frequency_rejects_bright_stars():
    # the LN gradient model must ignore stars — averaging them in punched dark/coloured holes
    flat = np.full((1024, 1024), 0.10)
    stars = flat.copy()
    rng = np.random.default_rng(0)
    ys, xs = rng.integers(10, 1014, 90), rng.integers(10, 1014, 90)
    stars[ys, xs] = 0.95
    lf_flat = nrm._low_frequency(flat)
    lf_stars = nrm._low_frequency(stars)
    assert float(np.abs(lf_stars - lf_flat).max()) < 0.02   # background model unmoved by stars


def test_local_normalize_does_not_hole_bright_stars():
    base = _starfield(H=1024, W=1024, n=120, seed=5)        # bright point sources
    ref = np.clip(base + 0.05, 0, 1)
    yy, xx = np.mgrid[0:1024, 0:1024].astype(np.float64)
    frame = np.clip(fftreg.apply_shift(base, 0.7, -0.5) + 0.05 + 0.12 * (xx / 1023), 0, 1)
    ref_med, ref_sig = nrm.frame_stats(ref)
    loc = nrm.local_normalize_to_ref(frame, ref, ref_med, ref_sig)
    # the old bilinear model punched deep dark holes at bright stars; block-median must not
    assert float(loc.min()) > -0.03
    assert float(loc.max()) > 0.4                           # bright stars still present


def test_local_normalize_reference_near_identity():
    ref = _disc(200, radius=60) * 0.6 + 0.05
    ref_med, ref_sig = nrm.frame_stats(ref)
    out = nrm.local_normalize_to_ref(ref, ref, ref_med, ref_sig)
    assert np.abs(out - ref).mean() < 5e-3            # ref maps to ~itself


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


def test_stack_prunes_orphaned_work_files(tmp_path):
    pytest.importorskip("photutils")
    lights = tmp_path / "lights"
    lights.mkdir()
    base = _starfield(n=40, seed=31)
    for i in range(5):
        save_image(str(lights / f"l_{i:03d}.fits"),
                   np.clip(fftreg.apply_shift(base, i - 2, -i + 1), 0, 1), bit_depth=16)
    p = LazyStackParams(do_calibrate=False, do_cosmetic=False,
                        stage_to_disk=True, reuse_cache=True)
    assert lsrun.stack(str(tmp_path), p) is not None
    work = tmp_path / "lazystack" / "work"
    # simulate an orphan from an earlier run (different reference / normalize setting)
    orphan = work / "reg_deadbeef_cafebabe.npy"
    orphan.write_bytes(b"stale")
    logs = []
    assert lsrun.stack(str(tmp_path), p, log=lambda s: logs.append(s)) is not None
    assert not orphan.exists()                                     # pruned
    assert any("Pruned" in s for s in logs)
    assert list(work.glob("reg_*_n_cw.npy"))                       # current files kept (coverage + walking-noise aware)


def test_sigma_clip_median_rejects_correlated_minority():
    # A hot value present in a MINORITY of frames at one pixel: mean/std leaves it in, median/MAD rejects.
    rng = np.random.default_rng(5)
    cube = np.stack([np.clip(0.4 + rng.normal(0, 0.003, (16, 16)), 0, 1) for _ in range(12)])
    cube[:5, 8, 8] = 0.95                              # 5 of 12 frames hot at the same pixel
    out = integ.sigma_clip_mean(cube, sigma_low=3, sigma_high=3)
    assert out[8, 8] < 0.5                             # correlated cluster rejected, not averaged to ~0.63


def test_static_hot_pixel_map_flags_fixed_not_drifting():
    base = _starfield(H=140, W=160, n=30, seed=3)
    hy, hx = 70, 80
    frames = []
    for i in range(8):
        f = np.clip(fftreg.apply_shift(base, 2 * (i - 4), 0), 0, 1).copy()   # fast horizontal drift
        f[hy, hx] = 1.0                               # a FIXED-sensor hot pixel every frame
        frames.append(f)
    bad = cal.static_hot_pixel_map(frames)
    assert bad[hy, hx]                                 # the persistent fixed pixel is flagged
    assert bad.sum() < 0.01 * base.size                # drifting stars are NOT mass-flagged


def test_repair_bad_pixels_replaces_with_local_median():
    a = np.full((20, 20), 0.2)
    a[10, 10] = 0.95
    mask = np.zeros((20, 20), bool)
    mask[10, 10] = True
    out = cal.repair_bad_pixels(a, mask)
    assert abs(out[10, 10] - 0.2) < 0.01              # repaired toward the local background
    assert out[0, 0] == 0.2                           # untouched elsewhere


def test_stack_runs_walking_noise_stage(tmp_path):
    pytest.importorskip("photutils")
    lights = tmp_path / "lights"
    lights.mkdir()
    base = _starfield(n=40, seed=12)
    for i in range(6):
        save_image(str(lights / f"l_{i:03d}.fits"),
                   np.clip(fftreg.apply_shift(base, i - 3, 1 - i), 0, 1), bit_depth=16)
    logs = []
    p = LazyStackParams(do_calibrate=False, do_cosmetic=False, fix_walking_noise=True,
                        stage_to_disk=True, reuse_cache=False)
    assert lsrun.stack(str(tmp_path), p, log=lambda s: logs.append(s)) is not None
    assert any("static hot" in s.lower() for s in logs)   # the walking-noise stage ran


def test_sigma_clip_ignores_nodata_nan():
    # A pixel covered by only 2 of 4 frames must be the mean of the 2 real values, not /4.
    cube = np.stack([
        np.full((4, 4), 0.5),
        np.full((4, 4), 0.5),
        np.full((4, 4), np.nan),      # no-data (registration border)
        np.full((4, 4), np.nan),
    ])
    out, cov = integ.sigma_clip_mean(cube, return_coverage=True)
    assert np.allclose(out, 0.5)                       # not 0.25 (divide-by-N against zeros)
    assert np.all(cov == 2)


def test_coverage_edges_crops_ragged_border_but_not_full_coverage():
    n = 20
    full = np.full((100, 120), n, dtype=np.int32)
    assert contract.coverage_edges(full, n) == {"L": 0, "R": 0, "T": 0, "B": 0}
    ragged = full.copy()
    ragged[:6, :] = 0        # top 6 rows never fully covered
    ragged[:, :4] = 0        # left 4 cols never fully covered
    e = contract.coverage_edges(ragged, n)
    assert e["T"] == 6 and e["L"] == 4 and e["B"] == 0 and e["R"] == 0


def test_stack_crops_dithered_master_smaller_than_source(tmp_path):
    pytest.importorskip("photutils")
    lights = tmp_path / "lights"
    lights.mkdir()
    base = _starfield(n=40, seed=7)
    H, W = base.shape
    for i in range(6):                                  # dithered shifts leave ragged borders
        save_image(str(lights / f"l_{i:03d}.fits"),
                   np.clip(fftreg.apply_shift(base, 2 * i - 5, -2 * i + 4), 0, 1), bit_depth=16)
    p = LazyStackParams(do_calibrate=False, do_cosmetic=False, edge_crop=True,
                        stage_to_disk=True, reuse_cache=False)
    res = lsrun.stack(str(tmp_path), p)
    assert res is not None
    mh, mw = res["master"].shape[:2]
    assert mh < H and mw < W                            # common-overlap crop trimmed the borders
    assert res["master_path"]


def test_suppress_banding_removes_fine_columns_keeps_broad_gradient():
    # Banding suppression targets the fine column-to-column FPN jumps, NOT broad column trends
    # (those are indistinguishable from a real horizontal gradient and must be preserved).
    rng = np.random.default_rng(0)
    H, W = 300, 400
    xx = np.mgrid[0:H, 0:W][1]
    base = np.full((H, W), 0.10) + rng.normal(0, 0.001, (H, W)) + 0.02 * (xx / W)   # + real gradient
    fine = 0.01 * np.sin(2 * np.pi * np.arange(W) / 3.0)            # fine column FPN (period 3 px)
    banded = np.clip(base + fine[None, :], 0, 1)
    fixed = cal.suppress_banding(banded, do_rows=False)
    resid = np.median(fixed, axis=0) - np.median(base, axis=0)
    assert np.std(resid) < 0.2 * np.std(fine)                       # fine banding removed
    cm = np.median(fixed, axis=0)                                   # broad gradient preserved
    assert abs((cm[-10:].mean() - cm[:10].mean()) - 0.02) < 0.005


def test_calibrate_light_shape_mismatch_raises_clear_error():
    light = np.zeros((10, 12, 3))
    dark = np.zeros((12, 10, 3))                       # transposed (portrait vs landscape)
    with pytest.raises(ValueError) as ei:
        cal.calibrate_light(light, dark=dark)
    assert "does not match" in str(ei.value) and "orientation" in str(ei.value).lower()


def test_stack_writes_noise_map_companion(tmp_path):
    pytest.importorskip("photutils")
    lights = tmp_path / "lights"
    lights.mkdir()
    base = _starfield(n=40, seed=15)
    for i in range(6):
        save_image(str(lights / f"l_{i:03d}.fits"),
                   np.clip(fftreg.apply_shift(base, i - 3, 2 - i), 0, 1), bit_depth=16)
    p = LazyStackParams(do_calibrate=False, do_cosmetic=False, emit_snr_map=True,
                        edge_crop=True, stage_to_disk=True, reuse_cache=False)
    res = lsrun.stack(str(tmp_path), p)
    assert res is not None and res["noise_map_path"]
    noise = np.load(res["noise_map_path"])
    assert noise.shape == res["master"].shape[:2]      # 2D, aligned to the cropped master
    assert np.all(np.isfinite(noise)) and float(noise.min()) >= 0.0
    cov_path = Path(res["master_path"]).with_name("lazystack_master_coverage.npy")
    assert cov_path.exists()                            # coverage companion written too
    cov = np.load(str(cov_path))
    assert cov.shape == res["master"].shape[:2] and int(cov.max()) <= res["n_stacked"]


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


def test_demaze_removes_grid_keeps_stars_and_noise():
    from scipy.ndimage import uniform_filter
    from lazystretch.lazystack.calibrate import demaze
    H, W = 120, 120
    yy, xx = np.mgrid[0:H, 0:W]
    grid = 0.02 * (((yy % 6) < 3).astype(float) - 0.5) * (((xx % 6) < 3).astype(float) - 0.5)
    img = np.clip(0.2 + grid + 0.005 * np.random.default_rng(0).standard_normal((H, W)), 0, 1)
    img[60, 60] = 0.9                                      # a star

    def grid_amp(x):
        y = x - uniform_filter(x, 7)
        return float(np.median(np.abs(np.median(y[:120, :120].reshape(20, 6, 20, 6), axis=(0, 2)))))

    out = demaze(img)
    assert grid_amp(out) < 0.3 * grid_amp(img)            # 6x6 grid largely removed
    assert out[60, 60] > 0.85                              # star preserved
    # what demaze removed must be ONLY the periodic 6x6 grid (its own tile-median reconstructs it) —
    # i.e. random noise is untouched.
    delta = np.clip(img, 0, 1) - out
    core = delta[6:114, 6:114]
    grid_of_delta = np.tile(np.median(core.reshape(18, 6, 18, 6), axis=(0, 2)), (18, 18))
    assert np.std(core - grid_of_delta) < 0.2 * (np.std(core) + 1e-12)
    assert np.allclose(demaze(img, strength=0.0), np.clip(img, 0, 1))     # strength 0 = no-op
    assert np.allclose(demaze(np.full((10, 10), 0.3)), np.full((10, 10), 0.3))   # too small = no-op
