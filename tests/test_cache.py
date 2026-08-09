"""Decode cache for camera raws (io/cache.py)."""
import numpy as np

from lazystretch.io import cache
from lazystretch.io.image_io import LoadedImage, load_image as real_load, save_image


def test_cached_load_decodes_once_then_serves_cache(tmp_path, monkeypatch):
    arr = np.clip(np.random.default_rng(0).random((20, 24, 3)), 0, 1)
    raw = tmp_path / "frame.cr2"
    raw.write_bytes(b"dummy-raw")
    cdir = tmp_path / "cache"
    calls = {"n": 0}

    def stub(path):
        if str(path).endswith(".cr2"):
            calls["n"] += 1                          # count expensive decodes
            return LoadedImage(data=arr.copy(), path=str(path))
        return real_load(path)                       # real load for the .tif cache

    monkeypatch.setattr(cache, "load_image", stub)
    a = cache.cached_load(str(raw), cdir)
    b = cache.cached_load(str(raw), cdir)
    assert calls["n"] == 1                            # decoded once; second read from cache
    assert cache.cache_path(str(raw), cdir).exists()
    assert np.allclose(a, b, atol=2e-5)               # 16-bit TIFF round-trip


def test_cached_load_passthrough_for_fast_formats(tmp_path):
    p = tmp_path / "x.fits"
    save_image(str(p), np.zeros((8, 8)), bit_depth=16)
    cdir = tmp_path / "cache"
    out = cache.cached_load(str(p), cdir)
    assert out is not None
    assert not cdir.exists()                          # fast formats are never cached


def test_cached_load_disabled_skips_cache(tmp_path, monkeypatch):
    raw = tmp_path / "f.cr2"
    raw.write_bytes(b"d")
    monkeypatch.setattr(cache, "load_image",
                        lambda p: LoadedImage(data=np.zeros((8, 8, 3)), path=str(p)))
    out = cache.cached_load(str(raw), tmp_path / "cache", enabled=False)
    assert out is not None
    assert not (tmp_path / "cache").exists()


def test_cache_key_changes_when_file_changes(tmp_path):
    raw = tmp_path / "f.cr2"
    raw.write_bytes(b"one")
    k1 = cache.cache_path(str(raw), tmp_path).name
    raw.write_bytes(b"different-size")               # mtime + size change
    k2 = cache.cache_path(str(raw), tmp_path).name
    assert k1 != k2                                   # stale cache auto-invalidates
