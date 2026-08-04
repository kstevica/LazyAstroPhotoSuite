"""Tests for the shared data loader (data/loader.py)."""
import json

import pytest

from lazystretch.data.loader import (
    SUPPORTED_SCHEMA_VERSION,
    ClassProfile,
    get_data,
    load_data,
)


def test_schema_and_class_list():
    D = get_data()
    assert D.schema_version == SUPPORTED_SCHEMA_VERSION == 1
    assert D.source_pi_version == "1.4.1"
    assert D.class_list == (
        "galaxy", "emission", "reflection", "planetary", "globular", "open", "generic",
    )
    assert len(D.profiles) == 7


def test_profile_known_values_verbatim():
    D = get_data()
    g = D.profile_for("galaxy")
    assert g.bkg == 0.16 and g.clip == -1.30 and g.sat == 0.45
    assert g.hdrLayers == 8 and g.starLevel == 0.60 and g.starReduce is True
    e = D.profile_for("emission")
    assert e.clip == -1.05 and e.bgLevel == 0.20 and e.hdrLayers is None
    glob = D.profile_for("globular")
    assert glob.scnr is False and glob.starLevel == 1.00
    assert isinstance(g, ClassProfile)


def test_catalogs():
    D = get_data()
    assert 42 in D.catalogs["messierClass"]["emission"]
    assert 31 in D.catalogs["messierClass"]["galaxy"]
    assert 1 in D.catalogs["messierClass"]["supernova"]
    assert 2118 in D.catalogs["reflectionCat"]["IC"]   # Witch Head
    assert 1848 in D.catalogs["emissionCat"]["IC"]     # Soul
    assert 7000 in D.catalogs["emissionCat"]["NGC"]    # North America


def test_predicates_frozensets():
    D = get_data()
    assert D.class_predicates["adaptiveFloorAppliesTo"] == frozenset({"emission", "generic"})
    assert D.class_predicates["starProtectAppliesTo"] == frozenset(
        {"emission", "reflection", "planetary", "generic"}
    )


def test_recipe_ranges_and_persisted():
    D = get_data()
    assert D.recipe_ranges["satAdj"] == (-0.5, 0.5)
    assert D.recipe_ranges["brightAdj"] == (-0.10, 0.10)
    assert D.recipe_ranges["starsAdj"] == (-0.3, 0.5)   # new in 1.4.x
    assert "smallStars" in D.persisted_reals
    assert {"chromaNR", "deepen", "starsAdj"}.issubset(D.persisted_reals)   # new in 1.4.x
    assert len(D.persisted_reals) == 12
    assert {"darkLaneGC", "removeStars"}.issubset(D.recipe_bools)           # new in 1.4.x
    assert len(D.recipe_bools) == 20


def test_profile_for_unknown_falls_back_to_generic():
    D = get_data()
    assert D.profile_for("nonsense") is D.profiles["generic"]


def test_schema_mismatch_raises(tmp_path):
    D = get_data()
    # Write a copy with a bumped schema_version and confirm it is rejected.
    from lazystretch.data.loader import _DATA_PATH
    raw = json.loads(_DATA_PATH.read_text())
    raw["schema_version"] = 999
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_data(bad)
