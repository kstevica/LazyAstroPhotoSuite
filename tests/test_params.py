"""Tests for effective-value resolution + class predicates (pipeline/params.py)."""
import pytest

from lazystretch.data.loader import get_data
from lazystretch.pipeline.params import (
    adaptive_floor_applies_to,
    effective_floor,
    masked_darken_applies_to,
    resolve_effective,
    star_protect_applies_to,
)

D = get_data()


def test_resolve_effective_zero_sliders_equal_profile():
    prof = D.profile_for("galaxy")
    eff = resolve_effective(prof, {})
    assert eff.bkg == prof.bkg
    assert eff.sat == prof.sat
    assert eff.clip == prof.clip
    assert eff.bgLevel == prof.bgLevel
    assert eff.contrast == prof.contrast


def test_resolve_effective_nudge():
    prof = D.profile_for("emission")
    eff = resolve_effective(prof, {"satAdj": 0.5, "brightAdj": 0.05})
    assert abs(eff.sat - (prof.sat + 0.5)) < 1e-12
    assert abs(eff.bkg - (prof.bkg + 0.05)) < 1e-12


def test_resolve_effective_clamps():
    prof = D.profile_for("emission")
    assert resolve_effective(prof, {"satAdj": 5.0}).sat == 1.50      # effSat hi
    assert resolve_effective(prof, {"blackAdj": -5.0}).clip == -3.00  # effClip lo
    assert resolve_effective(prof, {"brightAdj": 5.0}).bkg == 0.50    # effBkg hi
    assert resolve_effective(prof, {"bgAdj": -5.0}).bgLevel == 0.015  # effBgLevel lo


def test_effective_floor():
    assert abs(effective_floor(0.20, 0.03) - 0.23) < 1e-12
    assert effective_floor(0.39, 0.05) == 0.40   # clamp hi
    assert effective_floor(0.0, 0.0) == 0.015    # clamp lo


def test_predicates_match_source_sets():
    assert adaptive_floor_applies_to("emission")
    assert adaptive_floor_applies_to("generic")
    assert not adaptive_floor_applies_to("galaxy")
    assert not adaptive_floor_applies_to("reflection")

    for c in ("emission", "reflection", "generic", "open"):
        assert masked_darken_applies_to(c)
    assert not masked_darken_applies_to("galaxy")
    assert not masked_darken_applies_to("globular")

    for c in ("emission", "reflection", "planetary", "generic"):
        assert star_protect_applies_to(c)
    assert not star_protect_applies_to("galaxy")
    assert not star_protect_applies_to("open")
