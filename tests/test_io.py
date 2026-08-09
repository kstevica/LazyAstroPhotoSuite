"""Tests for image I/O (io/image_io.py) — real masters + synthetic round-trips."""
import os

import numpy as np
import pytest

from lazystretch.io.image_io import _canonicalize, _norm_dtype, load_image, save_image

EXAMPLE_DIR = "/Users/kstevica/Dev/Astro/LazyStretch/example"
_HAVE_EXAMPLES = os.path.isdir(EXAMPLE_DIR)


# --- canonicalization + normalization (no files needed) ----------------------

def test_canonicalize_planes_first_rgb():
    a = np.zeros((3, 20, 30))          # FITS (C,H,W)
    out = _canonicalize(a, planes_first=True)
    assert out.shape == (20, 30, 3)


def test_canonicalize_strips_alpha():
    a = np.zeros((20, 30, 4))          # RGBA (H,W,4)
    out = _canonicalize(a, planes_first=False)
    assert out.shape == (20, 30, 3)


def test_canonicalize_squeezes_mono():
    assert _canonicalize(np.zeros((20, 30, 1)), planes_first=False).shape == (20, 30)


def test_norm_dtype_integer_scaling():
    a = np.array([[0, 255]], dtype=np.uint8)
    out = _norm_dtype(a)
    assert out.dtype == np.float64
    assert out[0, 0] == 0.0 and out[0, 1] == pytest.approx(1.0)


def test_norm_dtype_float_clips():
    a = np.array([[-0.2, 0.5, 1.4]], dtype=np.float32)
    out = _norm_dtype(a)
    assert out.min() == 0.0 and out.max() == 1.0


# --- synthetic round-trips ---------------------------------------------------

def test_fits_rgb_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.random((40, 60, 3))
    p = tmp_path / "rt.fits"
    save_image(p, img)
    back = load_image(p)
    assert back.is_color and back.data.shape == (40, 60, 3)
    assert np.allclose(back.data, img, atol=1e-6)   # float32 storage


def test_tiff_16bit_roundtrip(tmp_path):
    img = np.linspace(0, 1, 40 * 60 * 3).reshape(40, 60, 3)
    p = tmp_path / "rt.tif"
    save_image(p, img, bit_depth=16)
    back = load_image(p)
    assert back.data.shape == (40, 60, 3)
    assert np.max(np.abs(back.data - img)) < 1.0 / 65535 + 1e-9


def test_png_8bit_roundtrip(tmp_path):
    img = np.linspace(0, 1, 30 * 30).reshape(30, 30)
    p = tmp_path / "rt.png"
    save_image(p, img, bit_depth=8)
    back = load_image(p)
    assert back.data.shape == (30, 30)
    assert np.max(np.abs(back.data - img)) < 1.0 / 255 + 1e-9


def test_png_rgb_16bit_falls_back_to_8bit(tmp_path):
    # Pillow can't encode 16-bit RGB PNG; saving must not crash (writes 8-bit instead).
    rgb = np.clip(np.random.default_rng(0).random((24, 32, 3)), 0, 1)
    p = tmp_path / "rgb16.png"
    save_image(p, rgb, bit_depth=16)                    # would raise "Cannot handle... <u2" before
    back = load_image(p)
    assert back.data.shape == (24, 32, 3)
    assert np.max(np.abs(back.data - rgb)) < 1.0 / 255 + 1e-6


def test_png_mono_16bit_roundtrip(tmp_path):
    img = np.linspace(0, 1, 40 * 40).reshape(40, 40)
    p = tmp_path / "mono16.png"
    save_image(p, img, bit_depth=16)                    # 16-bit grayscale PNG is supported
    back = load_image(p)
    assert back.data.shape == (40, 40)
    assert np.max(np.abs(back.data - img)) < 1.0 / 65535 + 1e-6


# --- real masters ------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_EXAMPLES, reason="example masters not present")
def test_load_real_mono_fits():
    li = load_image(f"{EXAMPLE_DIR}/melotte_15_redcat_533-Hydrogen-alpha.fits")
    assert li.data.ndim == 2                       # mono
    assert li.data.dtype == np.float64
    assert 0.0 <= li.data.min() and li.data.max() <= 1.0
    assert li.source_format == "fits"


@pytest.mark.skipif(not _HAVE_EXAMPLES, reason="example masters not present")
def test_load_real_tiff():
    li = load_image(f"{EXAMPLE_DIR}/m78.tif")
    assert li.is_color and li.data.shape[2] == 3
    assert 0.0 <= li.data.min() and li.data.max() <= 1.0
