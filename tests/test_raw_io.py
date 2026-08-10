"""Camera-raw loading — extension routing + folder scans (rawpy)."""
import numpy as np
import pytest

from lazystretch.io.image_io import RAW_DECODER_VERSION, RAW_EXT, _raw_demosaic, load_image
from lazystretch.lazystack import run as lsrun
from lazystretch.moonsun import run as msrun


class _FakeRaw:
    def __init__(self, n):
        self.raw_pattern = np.zeros((n, n), dtype=int)


def test_raw_extensions_registered():
    for e in (".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf"):
        assert e in RAW_EXT


def test_demosaic_picks_markesteijn_for_xtrans():
    pytest.importorskip("rawpy")
    _algo, name, is_xtrans = _raw_demosaic(_FakeRaw(6))     # 6x6 pattern == X-Trans
    assert is_xtrans and name == "AHD"                       # AHD (index 3) -> libraw 3-pass Markesteijn
    _algo, name, is_xtrans = _raw_demosaic(_FakeRaw(2))     # 2x2 pattern == Bayer
    assert not is_xtrans and name in ("DHT", "AAHD", "AHD")


def test_decoder_version_is_in_cache_key():
    # The raw decode cache must invalidate when the decoder changes (else stale VNG TIFFs served).
    import inspect

    from lazystretch.io import cache
    assert "RAW_DECODER_VERSION" in inspect.getsource(cache._cache_key)


def test_load_image_routes_raw_to_decoder(tmp_path):
    pytest.importorskip("rawpy")
    p = tmp_path / "bogus.cr2"
    p.write_bytes(b"not actually a raw file")
    with pytest.raises(Exception) as ei:
        load_image(str(p))
    # it tried to DECODE the raw (rawpy error), not reject the extension
    assert "unsupported image extension" not in str(ei.value)


def test_moonsun_scan_includes_raws(tmp_path):
    pytest.importorskip("rawpy")
    (tmp_path / "frame.cr2").write_bytes(b"x")
    scan = msrun.find_frames(str(tmp_path))
    assert any(p.endswith("frame.cr2") for p in scan["frames"])


def test_lazystack_scan_includes_raws(tmp_path):
    pytest.importorskip("rawpy")
    lights = tmp_path / "lights"
    lights.mkdir()
    (lights / "sub.nef").write_bytes(b"x")
    sets = lsrun.find_sets(str(tmp_path))
    assert any(p.endswith("sub.nef") for p in sets["lights"])
