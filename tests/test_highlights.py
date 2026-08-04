"""P1 highlight roll-off — no channel may hard-clip to pure white; below-knee is identity."""
import numpy as np
import pytest

from lazystretch.objects.model import Parameters
from lazystretch.pipeline.runcore import run_pipeline
from lazystretch.processes.highlights import highlight_rolloff


def test_rolloff_never_reaches_one():
    x = np.linspace(0.0, 1.0, 200)
    out = highlight_rolloff(x, knee=0.82)
    assert out.max() < 1.0                       # nothing slams to pure white
    assert out.min() >= 0.0


def test_rolloff_below_knee_is_identity():
    a = np.array([0.0, 0.3, 0.5, 0.81])
    assert np.allclose(highlight_rolloff(a, knee=0.82), a)


def test_rolloff_monotone_and_keeps_highlights_bright():
    x = np.linspace(0.82, 1.0, 60)
    out = highlight_rolloff(x, knee=0.82)
    assert np.all(np.diff(out) >= -1e-12)        # monotone (no inversion)
    assert out[-1] < 1.0 and out[-1] > 0.82      # 1.0 compressed, but still a bright highlight


def test_rolloff_preserves_channel_order():
    # a bright coloured pixel keeps R>G>B (hue) after the per-channel roll-off
    px = np.array([[[0.98, 0.9, 0.7]]])
    out = highlight_rolloff(px)[0, 0]
    assert out[0] > out[1] > out[2]
    assert out.max() < 1.0


def test_pipeline_has_no_pure_white():
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(0.06, 0.02, (140, 180, 3)), 0, 1)
    img[60:66, 90:96, :] = 0.99                  # a blown-ish star
    r = run_pipeline(img, Parameters.for_object("emission"), preview=True)
    assert any("Highlight roll-off" in s for s in r.steps_run)
    # after the roll-off, essentially nothing is at pure white
    assert float((r.image >= 0.999).mean()) < 1e-4
