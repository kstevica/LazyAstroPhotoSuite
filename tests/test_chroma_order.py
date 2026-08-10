"""Chroma-NR ordering + per-class default (review-driven fix).

Both external reviewers flagged colour "confetti": the saturation boost was amplifying raw R/B
chroma speckle because chroma NR ran *after* saturation (and was off by default). The fix runs
chroma NR *before* saturation and seeds a modest per-class default for OSC broadband classes.
"""
import numpy as np

from lazystretch.objects.model import Parameters
from lazystretch.pipeline.runcore import run_pipeline


def _osc(H=140, W=170, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    gas = 0.30 * np.exp(-(((xx - W / 2) / 45) ** 2 + ((yy - H / 2) / 34) ** 2))
    L = np.clip(0.06 + gas + rng.normal(0, 0.006, (H, W)), 0, 1)
    return np.stack([L, L * 0.82, L * 0.7], axis=-1)


def test_osc_classes_seed_modest_chroma_default():
    assert Parameters.for_object("milkyway").chromaNR == 0.40
    assert Parameters.for_object("emission").chromaNR == 0.30
    assert Parameters.for_object("generic").chromaNR == 0.25
    # star-dominated / narrowband-typical classes stay off (protect star colour)
    assert Parameters.for_object("globular").chromaNR == 0.0
    assert Parameters.for_object("open").chromaNR == 0.0


def test_chroma_nr_runs_before_saturation():
    p = Parameters.for_object("milkyway")          # chromaNR seeded 0.40 > 0 -> step present
    res = run_pipeline(_osc(), p, preview=True)
    chroma = [i for i, s in enumerate(res.steps_run) if "Chroma noise reduction" in s]
    sat = [i for i, s in enumerate(res.steps_run) if "Saturation boost" in s]
    assert chroma and sat, (res.steps_run,)
    assert chroma[0] < sat[0]                       # cleaned before the boost amplifies speckle


def test_chroma_nr_absent_when_dialed_off():
    p = Parameters.for_object("milkyway", chromaNR=0.0)
    res = run_pipeline(_osc(), p, preview=True)
    assert not any("Chroma noise reduction" in s for s in res.steps_run)
