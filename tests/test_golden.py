"""Golden / regression harness (PLAN §9.3, Layer 1).

Golden fixtures live as ``tests/golden/*.expected.json``. Each declares a ``kind``:

  * ``mtf_vectors``  — closed-form MTF checks + find-midtones round-trips. Present now;
    independent of the implementation, so it locks the pinned primitive.
  * ``pi_telemetry`` — statistics MEASURED by PixInsight (via a guarded telemetry mode
    in LazyStretch.js) that the Python core must reproduce from the same raw inputs.
    None ship yet; add them after running the PI export (see tests/golden/README.md).
    ``test_pi_telemetry_present`` skips loudly until they exist.
"""
import glob
import json
import os

import pytest

from lazystretch.data.loader import get_data
from lazystretch.pipeline.params import resolve_effective
from lazystretch.stats.mtf import (
    FIND_MIDTONES_EPS,
    find_midtones_balance,
    mtf,
    mtf_scalar,
)
from lazystretch.stretch.autostretch import solve_stretch

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
FILES = sorted(glob.glob(os.path.join(GOLDEN_DIR, "*.expected.json")))


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _check_mtf_vectors(spec):
    atol = spec.get("atol", 1e-12)
    for c in spec.get("mtf", []):
        got = float(mtf(c["m"], c["x"]))
        assert abs(got - c["y"]) < atol, (c, got)
    for c in spec.get("find_midtones_roundtrip", []):
        m = find_midtones_balance(c["v0"], c["v1"])
        assert abs(mtf_scalar(m, c["v1"]) - c["v0"]) < FIND_MIDTONES_EPS + 1e-9, c


def _check_pi_telemetry(spec):
    """Reproduce PI-measured derived numbers from the same raw inputs.

    Expected sidecar shape (all sections optional)::

        {
          "kind": "pi_telemetry", "image": "M42_...", "atol": 1e-4,
          "stretch":   [{"median":.., "avgDev":.., "shadowsClip":.., "targetBkg":..,
                         "expect": {"c0":.., "m":..}}],
          "effective": [{"class":"emission", "sliders": {...},
                         "expect": {"bkg":.., "sat":.., "clip":.., "bgLevel":.., "contrast":..}}]
        }
    """
    D = get_data()
    atol = spec.get("atol", 1e-4)
    for rec in spec.get("stretch", []):
        res = solve_stretch(
            rec["median"], rec["avgDev"], rec["shadowsClip"], rec["targetBkg"], D
        )
        exp = rec["expect"]
        assert abs(res.c0 - exp["c0"]) < atol, ("c0", rec)
        assert abs(res.m - exp["m"]) < atol, ("m", rec)
    for rec in spec.get("effective", []):
        eff = resolve_effective(D.profile_for(rec["class"]), rec.get("sliders", {}), D)
        exp = rec["expect"]
        for field in ("bkg", "sat", "clip", "bgLevel", "contrast"):
            if field in exp:
                assert abs(getattr(eff, field) - exp[field]) < atol, (field, rec)


_CHECKERS = {"mtf_vectors": _check_mtf_vectors, "pi_telemetry": _check_pi_telemetry}


@pytest.mark.parametrize("path", FILES, ids=[os.path.basename(p) for p in FILES])
def test_golden_file(path):
    spec = _load(path)
    kind = spec.get("kind")
    checker = _CHECKERS.get(kind)
    if checker is None:
        pytest.skip(f"unknown golden kind {kind!r} in {os.path.basename(path)}")
    checker(spec)


def test_pi_telemetry_present():
    have = [p for p in FILES if _load(p).get("kind") == "pi_telemetry"]
    if not have:
        pytest.skip(
            "no PI telemetry sidecars yet — run the PixInsight telemetry export over "
            "the sample masters and drop *.expected.json here (see tests/golden/README.md)"
        )
