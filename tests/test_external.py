"""External-tool integration: fake CLI stubs prove the shell-out round-trip + graceful
degradation, without needing the real StarNet/GraXpert binaries."""
import os
import stat
import sys

import numpy as np
import pytest

from lazystretch.external import (SPCC, BlurX, DeepSNR, GraXpert, NoiseX, RCStarX, StarX,
                                   Tools, star_recombine)
from lazystretch.objects.model import Parameters
from lazystretch.pipeline.runcore import run_pipeline
from lazystretch.processes.deconv import richardson_lucy

# Fake StarNet2: `starnet2 -i in.tif -o out.tif -s stride` -> copy input to output (identity).
_FAKE_STARNET = """#!{py}
import shutil, sys
a = sys.argv
shutil.copyfile(a[a.index('-i') + 1], a[a.index('-o') + 1])
"""

# Fake GraXpert: `graxpert in.fits -cli -cmd X -output OUT ...` -> write OUT.fits = input.
_FAKE_GRAXPERT = """#!{py}
import shutil, sys
inp = sys.argv[1]
out = sys.argv[sys.argv.index("-output") + 1]
shutil.copyfile(inp, out + ".fits")
"""


def _make_exe(path, body):
    path.write_text(body.format(py=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return str(path)


def _rgb(H=64, W=80, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.3, 0.08, (H, W, 3)), 0, 1)


# --- graceful degradation (no tools) ----------------------------------------

def test_tools_absent_report_unavailable():
    t = Tools(GraXpert("/nope"), StarX("/nope"), DeepSNR("/nope"), SPCC(),
                  BlurX("/nope"), RCStarX("/nope"), NoiseX("/nope"))
    st = t.status()
    assert st["GraXpert (background)"] is False
    assert st["StarNet (star reduction)"] is False
    assert st["DeepSNR (noise reduction)"] is False


def test_starx_run_skips_when_absent():
    img = _rgb()
    out, ran = StarX("/nonexistent").run(img)
    assert ran is False
    assert np.array_equal(out, img)


# --- fake-tool round-trips ---------------------------------------------------

def test_starnet_roundtrip_via_fake(tmp_path):
    exe = _make_exe(tmp_path / "fake_starnet", _FAKE_STARNET)
    sx = StarX(exe)
    assert sx.is_available()
    img = _rgb()
    starless = sx.starless(img)              # identity copy through a 16-bit TIFF
    assert starless.shape == img.shape
    assert np.max(np.abs(starless - img)) < 1.0 / 65535 + 1e-6
    reduced = sx.reduce_stars(img, star_level=0.5, small_stars=0.1)
    assert reduced.shape == img.shape and reduced.min() >= 0 and reduced.max() <= 1
    mask = sx.star_mask(img)
    assert mask.shape == img.shape[:2]


def test_graxpert_roundtrip_via_fake(tmp_path):
    exe = _make_exe(tmp_path / "fake_graxpert", _FAKE_GRAXPERT)
    gx = GraXpert(exe)
    assert gx.is_available()
    img = _rgb()
    den = gx.denoise(img)
    bg = gx.background_extraction(img)
    for out in (den, bg):
        assert out.shape == img.shape
        assert np.max(np.abs(out - img)) < 1e-5   # FITS float32 identity


# --- pipeline uses the tools -------------------------------------------------

def test_pipeline_uses_fake_tools(tmp_path):
    starnet = _make_exe(tmp_path / "fake_starnet", _FAKE_STARNET)
    graxpert = _make_exe(tmp_path / "fake_graxpert", _FAKE_GRAXPERT)
    tools = Tools(GraXpert(graxpert), StarX(starnet), DeepSNR("/nope"), SPCC(),
                  BlurX("/nope"), RCStarX("/nope"), NoiseX("/nope"))
    img = _rgb(120, 160)
    # galaxy: doNR + doStarReduce on; execute (not preview) so the wall runs
    p = Parameters.for_object("galaxy")
    r = run_pipeline(img, p, preview=False, tools=tools)
    assert r.steps_skipped == []
    joined = " ".join(r.log)
    assert "via GraXpert" in joined            # NR ran through GraXpert
    assert "via StarNet" in joined             # star reduction ran through StarNet
    assert any("Star reduction" in s for s in r.steps_run)


def test_pipeline_degrades_without_tools():
    img = _rgb(120, 160)
    tools = Tools(GraXpert("/nope"), StarX("/nope"), DeepSNR("/nope"), SPCC(),
                  BlurX("/nope"), RCStarX("/nope"), NoiseX("/nope"))
    r = run_pipeline(img, Parameters.for_object("galaxy"), preview=False, tools=tools)
    assert r.steps_skipped == []
    joined = " ".join(r.log)
    assert "MMT fallback" in joined            # NR fell back to MMT
    assert "Star reduction skipped (no star tool installed)" in joined


# --- classical deconvolution + star math ------------------------------------

def test_richardson_lucy_runs():
    img = _rgb()
    out = richardson_lucy(img, sigma=1.2, iterations=5)
    assert out.shape == img.shape and out.min() >= 0 and out.max() <= 1


def test_star_recombine_reduces_stars():
    starless = np.full((10, 10), 0.2)
    orig = starless.copy()
    orig[5, 5] = 0.9                            # a "star"
    out = star_recombine(orig, starless, star_level=0.3)
    assert out[5, 5] < orig[5, 5]              # star dimmed
    assert abs(out[0, 0] - 0.2) < 1e-9        # background untouched
