"""LazyMoonSun orchestration — folder scan, master write, mp_report."""
import numpy as np

from lazystretch.io.image_io import load_image, save_image
from lazystretch.moonsun import run as msrun
from lazystretch.moonsun.model import MoonSunParams
from lazystretch.moonsun import register as reg
from tests.test_moonsun_register import _disc


def _write_burst(folder, n=6, shift=2.0, seed=0):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = _disc(256, radius=80)
    for i in range(n):
        f = reg.apply_shift(base, rng.uniform(-shift, shift), rng.uniform(-shift, shift))
        f = np.clip(f + rng.normal(0, 0.02, f.shape), 0, 1)
        save_image(str(folder / f"frame_{i:03d}.fits"), f, bit_depth=16)
    return base


def test_find_frames_lists_and_flags_raws(tmp_path):
    (tmp_path / "a.fits").write_bytes(b"x")
    (tmp_path / "b.tif").write_bytes(b"x")
    (tmp_path / "c.cr2").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    scan = msrun.find_frames(str(tmp_path))
    names = {p.rsplit("/", 1)[-1] for p in scan["frames"]}
    assert {"a.fits", "b.tif"} <= names
    if msrun._rawpy_available():                          # raws load when rawpy is present
        assert "c.cr2" in names and scan["raws"] == []
    else:
        assert any(p.endswith("c.cr2") for p in scan["raws"])


def test_stack_burst_folder_writes_master(tmp_path):
    burst = tmp_path / "burst"
    _write_burst(burst, n=6)
    res = msrun.stack_burst_folder(str(burst), MoonSunParams(keep=1.0))
    assert res is not None
    master_path = burst / "lazysun" / msrun.MASTER_NAME
    assert master_path.exists()
    loaded = load_image(str(master_path))
    assert loaded.data.shape[:2] == (256, 256)
    assert int(loaded.keyword("LSNFRAME")) == res["n_stacked"] == 6


def test_stack_burst_folder_needs_two_frames(tmp_path):
    burst = tmp_path / "one"
    burst.mkdir()
    save_image(str(burst / "only.fits"), _disc(128, radius=40), bit_depth=16)
    assert msrun.stack_burst_folder(str(burst), MoonSunParams()) is None


def test_multipoint_folder_writes_master_and_report(tmp_path):
    burst = tmp_path / "burst"
    _write_burst(burst, n=8)
    assert msrun.stack_burst_folder(str(burst), MoonSunParams(keep=1.0)) is not None
    params = MoonSunParams(ap_size=48, ap_keep=0.6, ap_search=8, ap_max=200)
    res = msrun.stack_multipoint_folder(str(burst), params)
    assert res is not None
    assert (burst / "lazysun" / msrun.MP_MASTER_NAME).exists()
    report = (burst / "lazysun" / "mp_report.txt")
    assert report.exists()
    text = report.read_text()
    assert "LazyMoonSun MP run" in text and "anisoplanatism" in text
    loaded = load_image(str(burst / "lazysun" / msrun.MP_MASTER_NAME))
    assert int(loaded.keyword("LSNMPAP")) == res["n_aps"]


def test_multipoint_needs_global_master_first(tmp_path):
    burst = tmp_path / "burst"
    _write_burst(burst, n=4)
    assert msrun.stack_multipoint_folder(str(burst), MoonSunParams()) is None
