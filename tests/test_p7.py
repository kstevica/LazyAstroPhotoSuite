"""P7 — v1.4.1 catch-up: Chroma NR, dark-lane, remove-stars, starsAdj, presets."""
import stat
import sys

import numpy as np
import pytest

from lazystretch.external import (BlurX, DeepSNR, GraXpert, NoiseX, RCStarX, SPCC, StarX,
                                   Tools)
from lazystretch.objects.model import Parameters
from lazystretch.objects.presets import apply_preset, curated_for
from lazystretch.pipeline.runcore import run_pipeline
from lazystretch.processes import chromanr, darklane, finishing


def _rgb(H=140, W=180, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    blob = 0.15 * np.exp(-(((xx - W * 0.6) / 30) ** 2 + ((yy - H / 2) / 24) ** 2))
    base = np.clip(rng.normal(0.06, 0.01, (H, W)) + blob, 0, 1)
    return np.stack([base, base * 0.85, base * 0.7], axis=-1)


# --- Chroma NR --------------------------------------------------------------

def test_chroma_nr_preserves_luminance_and_range():
    img = _rgb()
    out = chromanr.chroma_nr(img, 0.6)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    dlum = float(np.max(np.abs(img.mean(axis=2) - out.mean(axis=2))))
    assert dlum < 0.06, ("luminance should be ~preserved", dlum)


def test_chroma_nr_noops():
    img = _rgb()
    assert np.array_equal(chromanr.chroma_nr(img, 0.0), img)          # amount 0
    mono = img[..., 0]
    assert np.array_equal(chromanr.chroma_nr(mono, 0.6), mono)         # mono no-op


# --- dark-lane gradient -----------------------------------------------------

def test_dark_lane_runs_mono_and_rgb():
    img = _rgb(240, 320)
    for a in (img, img[..., 0]):
        out = darklane.dark_lane_gradient(a)
        assert out.shape == a.shape
        assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


def test_dark_lane_tiny_frame_unchanged():
    tiny = _rgb(20, 20)
    out = darklane.dark_lane_gradient(tiny)   # too few anchors -> unchanged
    assert np.array_equal(out, tiny)


# --- dehaze gradient arg -----------------------------------------------------

def test_dehaze_accepts_gradient_arg():
    img = _rgb()
    out = finishing.dehaze(img, 0.5, 0.2, "emission", 0.3)
    assert out.shape == img.shape and out.min() >= 0.0 and out.max() <= 1.0


# --- pipeline wiring --------------------------------------------------------

def test_pipeline_chroma_and_darklane_steps():
    img = _rgb()
    p = Parameters.for_object("emission", chromaNR=0.5, darkLaneGC=True)
    r = run_pipeline(img, p, preview=True)
    assert r.steps_skipped == []
    assert any("Chroma noise reduction" in s for s in r.steps_run)
    assert any("Dark-lane" in s for s in r.steps_run)


def test_pipeline_starsadj_changes_reduction_level(tmp_path):
    # fake StarNet (identity copy) so star reduction runs; assert the logged level = starLevel+starsAdj
    exe = tmp_path / "fake_starnet"
    exe.write_text(f"#!{sys.executable}\nimport shutil,sys\na=sys.argv;shutil.copyfile(a[a.index(chr(45)+chr(105))+1],a[a.index(chr(45)+chr(111))+1])\n")
    exe.chmod(exe.stat().st_mode | stat.S_IRWXU)
    tools = Tools(GraXpert("/nope"), StarX(str(exe)), DeepSNR("/nope"), SPCC(),
                  BlurX("/nope"), RCStarX("/nope"), NoiseX("/nope"))
    img = _rgb(120, 160)
    # galaxy starLevel = 0.60; starsAdj +0.20 -> effective 0.80
    p = Parameters.for_object("galaxy", starsAdj=0.20)
    r = run_pipeline(img, p, preview=False, tools=tools)
    assert r.steps_skipped == []
    assert any("Star reduction (level 0.80)" in s for s in r.steps_run)


def test_pipeline_remove_stars_outputs_layer(tmp_path):
    exe = tmp_path / "fake_starnet"
    exe.write_text(f"#!{sys.executable}\nimport shutil,sys\na=sys.argv;shutil.copyfile(a[a.index(chr(45)+chr(105))+1],a[a.index(chr(45)+chr(111))+1])\n")
    exe.chmod(exe.stat().st_mode | stat.S_IRWXU)
    tools = Tools(GraXpert("/nope"), StarX(str(exe)), DeepSNR("/nope"), SPCC(),
                  BlurX("/nope"), RCStarX("/nope"), NoiseX("/nope"))
    img = _rgb(120, 160)
    p = Parameters.for_object("galaxy", removeStars=True)
    r = run_pipeline(img, p, preview=False, tools=tools)
    assert r.stars_layer is not None
    assert r.stars_layer.shape == r.image.shape
    assert any("Remove stars" in s for s in r.steps_run)


# --- curated presets --------------------------------------------------------

def test_curated_for_leading_designation():
    assert curated_for("IC434 — Horsehead Nebula")["name"] == "Horsehead / Flame region"
    assert curated_for("IC 434")["name"] == "Horsehead / Flame region"     # spacing tolerated
    assert curated_for("M42")["name"] == "Orion Nebula"
    assert curated_for("M31") is None
    assert curated_for(None) is None


def test_apply_preset_sets_fields():
    p = Parameters.for_object("emission")
    apply_preset(p, curated_for("IC2118")["settings"])
    assert p.enhanceEmission is True and p.darkLaneGC is True
    assert abs(p.satAdj - 0.40) < 1e-9 and abs(p.starsAdj - 0.30) < 1e-9
