"""RC-Astro standalone CLI wrappers: detection, argv/roundtrip, and pipeline priority.

A fake ``rc-astro`` executable (shebang -> this venv's python, using the package's own
image I/O) stands in for the licensed tool, so the wrapper machinery — sub-command +
input + ``--output <file>`` argv, output read-back — is exercised for real. Host
auto-detection is neutralised by the hermetic conftest fixture, so the "absent" cases are
deterministic even though this machine really has the RC-Astro CLI installed.
"""
import stat
import sys
import textwrap

import numpy as np
import pytest

from lazystretch.external import (SPCC, BlurX, DeepSNR, GraXpert, NoiseX, RCStarX, StarX,
                                   Tools)
from lazystretch.external.rcastro import resolve_rcastro
from lazystretch.objects.model import Parameters
from lazystretch.pipeline.runcore import run_pipeline


def _make_fake_rcastro(dirpath) -> str:
    """Write an executable ``rc-astro`` that mimics the real argv + per-product output.

    Parses the real invocation (``[--no-banner] <sub> <in> ... --output <file> --overwrite``)
    and writes a transformed copy to the exact ``--output`` file.
    """
    script = dirpath / "rc-astro"
    script.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import os, sys
        from lazystretch.io.image_io import load_image, save_image
        a = sys.argv[1:]
        sub = a[1] if a and a[0] == "--no-banner" else a[0]
        inp = a[a.index(sub) + 1]
        out = a[a.index("--output") + 1]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        data = load_image(inp).data
        if sub == "sxt":            # simulate star removal: a dimmer starless
            data = data * 0.7
        elif sub == "bxt":          # simulate mild sharpening
            data = data ** 0.98
        save_image(out, data, bit_depth=16)
    """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


@pytest.fixture
def fake_rcastro(tmp_path):
    return _make_fake_rcastro(tmp_path)


def test_detection_explicit_and_env(fake_rcastro, monkeypatch):
    # explicit path resolves first
    assert resolve_rcastro(fake_rcastro) == fake_rcastro
    assert BlurX(fake_rcastro).is_available()
    # env var
    monkeypatch.setenv("LAZYSTRETCH_RCASTRO", fake_rcastro)
    assert NoiseX().is_available()


def test_absent_degrades_gracefully():
    # with host lookup neutralised, a bogus path resolves to nothing
    assert resolve_rcastro("/no/such/rc-astro") is None
    bx = BlurX("/no/such/rc-astro")
    assert not bx.is_available()
    img = np.linspace(0, 1, 300, dtype=np.float64).reshape(10, 10, 3)
    out, ok = bx.run(img)
    assert ok is False and np.array_equal(out, img)


def test_bxt_and_nxt_roundtrip(fake_rcastro):
    img = np.clip(np.random.default_rng(0).random((16, 24, 3)), 0.02, 0.98)
    bx = BlurX(fake_rcastro)
    dec = bx.deconvolve(img)
    assert dec.shape == img.shape and np.isfinite(dec).all()
    assert np.mean(np.abs(dec - img)) > 1e-4               # the tool actually ran

    nx = NoiseX(fake_rcastro)
    den = nx.denoise(img)
    assert den.shape == img.shape and np.isfinite(den).all()

    # optional flags are only passed when supplied (no guessed values otherwise)
    dec2 = bx.deconvolve(img, sharpen_stars=0.4, sharpen_nonstellar=0.6)
    assert dec2.shape == img.shape


def test_sxt_starless_stars_layer_and_reduce(fake_rcastro):
    img = np.clip(np.random.default_rng(1).random((20, 20, 3)), 0.05, 0.95)
    sx = RCStarX(fake_rcastro)
    assert sx.is_available() and sx.label == "StarXTerminator"

    starless, stars = sx.remove_stars(img)
    assert starless.shape == img.shape
    assert np.all(starless <= img + 1e-6)                  # fake sxt dims -> starless <= orig
    assert float(stars.mean()) > 0.0                       # a real, positive stars layer
    # star reduction blends the stars layer back at a fraction
    red = sx.reduce_stars(img, star_level=0.5)
    assert red.shape == img.shape and np.isfinite(red).all()
    # star mask peaks at 1.0
    m = sx.star_mask(img)
    assert m.ndim == 2 and pytest.approx(1.0, abs=1e-6) == float(m.max())


def test_tools_star_tool_prefers_rcastro(tmp_path, fake_rcastro):
    # a stand-in "StarNet" — StarX resolves any existing file given as its path
    fake_starnet = tmp_path / "starnet++"
    fake_starnet.write_text("#!/bin/sh\n")
    fake_starnet.chmod(0o755)

    # both present -> RC-Astro SXT wins
    t = Tools.resolve(starnet_path=str(fake_starnet), rcastro_path=fake_rcastro)
    st = t.star_tool()
    assert isinstance(st, RCStarX) and st.label == "StarXTerminator"

    # only StarNet -> falls back to it
    t2 = Tools.resolve(starnet_path=str(fake_starnet), rcastro_path="/no/such")
    assert t2.star_tool() is t2.starx and t2.starx.label == "StarNet"

    # neither -> None
    t3 = Tools.resolve(starnet_path="/no/such", rcastro_path="/no/such")
    assert t3.star_tool() is None


def _linear_scene(h=90, w=130):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    neb = 0.06 + 0.05 * np.exp(-(((xx - w / 2) / 40) ** 2 + ((yy - h / 2) / 26) ** 2))
    img = np.stack([neb, neb * 0.85, neb * 1.05], -1)
    rng = np.random.default_rng(4)
    return np.clip(img + rng.normal(0, 0.006, img.shape), 0, 1).astype(np.float64)


def test_noisex_runs_in_linear_slot_before_stretch(fake_rcastro):
    # only NoiseX present (others absent); NR must run in the LINEAR slot, before the stretch
    tools = Tools(GraXpert("/nope"), StarX("/nope"), DeepSNR("/nope"), SPCC(),
                  BlurX("/nope"), RCStarX("/nope"), NoiseX(fake_rcastro))
    p = Parameters.for_object("galaxy")
    p.doNR = True; p.inputStretched = False
    r = run_pipeline(_linear_scene(), p, preview=False, tools=tools)
    assert r.steps_skipped == []
    joined = " ".join(r.log)
    assert "via NoiseXTerminator (linear)" in joined
    assert "Noise reduction (post-stretch) skipped (NoiseX ran in the linear slot)" in joined
    # ordering: the linear NR step lands BEFORE the auto-stretch step
    nr = next(i for i, s in enumerate(r.steps_run) if "linear — NoiseXTerminator" in s)
    st = next(i for i, s in enumerate(r.steps_run) if s.startswith("Auto-stretch"))
    assert nr < st
    # and there is no separate post-stretch "Noise reduction" step
    assert not any(s == "Noise reduction" for s in r.steps_run)


def test_noisex_absent_keeps_poststretch_nr(fake_rcastro):
    # NoiseX absent -> the post-stretch chain still runs (here: MMT, since no DeepSNR/GraXpert)
    tools = Tools(GraXpert("/nope"), StarX("/nope"), DeepSNR("/nope"), SPCC(),
                  BlurX("/nope"), RCStarX("/nope"), NoiseX("/nope"))
    p = Parameters.for_object("galaxy")
    p.doNR = True; p.inputStretched = False
    r = run_pipeline(_linear_scene(), p, preview=False, tools=tools)
    joined = " ".join(r.log)
    assert "linear — NoiseXTerminator" not in joined
    assert any(s == "Noise reduction" for s in r.steps_run)   # post-stretch slot ran
    assert "MMT fallback" in joined


def test_tools_status_lists_rcastro(fake_rcastro):
    t = Tools.resolve(rcastro_path=fake_rcastro)
    s = t.status()
    assert s["BlurXTerminator (RC-Astro deconv)"] is True
    assert s["StarXTerminator (RC-Astro stars)"] is True
    assert s["NoiseXTerminator (RC-Astro NR)"] is True
    # absent RC-Astro leaves them False without disturbing the open-source rows
    t0 = Tools.resolve(rcastro_path="/no/such")
    s0 = t0.status()
    assert s0["BlurXTerminator (RC-Astro deconv)"] is False
    assert "GraXpert (background)" in s0
