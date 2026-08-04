"""Tests for identification: catalog, classify, palettes, and the solver contract."""
import math

import numpy as np
import pytest

from lazystretch.identify import (
    CatalogRow,
    angular_sep_deg,
    detect_data_type,
    get_catalog,
    object_class,
    palette_key,
    palettes_for,
)
from lazystretch.identify.solver import solve_from_header
from lazystretch.io.image_io import LoadedImage


# --- angular separation ------------------------------------------------------

def test_angular_sep_zero_and_known():
    assert angular_sep_deg(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-9)
    # 1 hour of RA at the equator = 15 degrees.
    assert angular_sep_deg(0.0, 0.0, 15.0, 0.0) == pytest.approx(15.0, abs=1e-6)
    # points 1 deg apart in declination.
    assert angular_sep_deg(30.0, 10.0, 30.0, 11.0) == pytest.approx(1.0, abs=1e-6)


# --- catalog -----------------------------------------------------------------

def test_catalog_loads_and_finds_by_id():
    cat = get_catalog()
    assert len(cat.rows) > 5000            # Messier + NGC-IC
    m42 = cat.find_by_name_or_id("M42")
    assert m42 is not None and m42.id.upper().replace(" ", "") == "M42"
    assert cat.find_by_name_or_id("M 31").id.upper().replace(" ", "") == "M31"


def test_catalog_find_by_common_name():
    cat = get_catalog()
    r = cat.find_by_name_or_id("Andromeda Galaxy")
    assert r is not None and "M31" in r.id.upper().replace(" ", "")


def test_find_in_field_ranks_and_dedupes():
    cat = get_catalog()
    m31 = cat.find_by_name_or_id("M31")
    cands = cat.find_in_field(m31.ra, m31.dec, 1.0)  # 1-degree field radius
    assert cands, "expected M31 in its own field"
    # M31 should rank at or near the top (big + bright + central + Messier bonus).
    top_ids = {c.obj.id.upper().replace(" ", "") for c in cands[:3]}
    assert "M31" in top_ids
    # dedup: no two candidates share a 0.01-deg position bucket.
    keys = [(round(c.obj.ra * 100), round(c.obj.dec * 100)) for c in cands]
    assert len(keys) == len(set(keys))


# --- object_class ------------------------------------------------------------

def _row(id="", commonName="", axisRatio=float("nan"), diameter=float("nan")):
    return CatalogRow(id=id, ra=0.0, dec=0.0, mag=float("nan"),
                      diameter=diameter, axisRatio=axisRatio,
                      commonName=commonName, source="M")


def test_object_class_messier_and_ngc_ic():
    assert object_class(_row(id="M42")) == "emission"
    assert object_class(_row(id="M31")) == "galaxy"
    assert object_class(_row(id="M13")) == "globular"
    assert object_class(_row(id="M45")) == "open"
    assert object_class(_row(id="M1")) == "emission"          # supernova -> emission
    assert object_class(_row(id="IC2118")) == "reflection"    # Witch Head (reflectionCat)
    assert object_class(_row(id="IC1848")) == "emission"      # Soul (emissionCat)
    assert object_class(_row(id="NGC7000")) == "emission"     # North America


def test_object_class_name_keywords_and_fallbacks():
    assert object_class(_row(id="NGC0000", commonName="Some Galaxy")) == "galaxy"
    assert object_class(_row(id="NGC0000", commonName="Reflection blob")) == "reflection"
    assert object_class(_row(id="NGC0000", commonName="Big Nebula")) == "emission"
    assert object_class(_row(id="NGC0000", axisRatio=3.0, diameter=5.0)) == "galaxy"
    assert object_class(_row(id="NGC0000")) == "generic"
    assert object_class(None) == "generic"


# --- data type + palettes ----------------------------------------------------

def test_detect_data_type():
    assert detect_data_type("Ha", False) == "narrowband"
    assert detect_data_type("OIII", False) == "narrowband"
    assert detect_data_type("Red", False) == "broadband"
    assert detect_data_type("RGB", True) == "osc"
    assert detect_data_type(None, True) == "osc"
    assert detect_data_type(None, False) == "broadband"


def test_palette_helpers():
    assert palette_key("SHO (Hubble)") == "SHO"
    assert palette_key("HOO") == "HOO"
    assert palette_key("Bicolor") == "HOO"
    assert palette_key("RGB") is None
    assert "SHO (Hubble)" in palettes_for("narrowband")


# --- solver fast path (synthetic WCS) ----------------------------------------

def test_solve_from_header_synthetic_wcs():
    # A minimal TAN WCS centred at RA=83.8, Dec=-5.4 (near M42), 2 arcsec/px.
    scale = 2.0 / 3600.0
    hdr = {
        "NAXIS": 2, "NAXIS1": 1000, "NAXIS2": 800,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": 500.0, "CRPIX2": 400.0,
        "CRVAL1": 83.8, "CRVAL2": -5.4,
        "CD1_1": -scale, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": scale,
    }
    img = LoadedImage(data=np.zeros((800, 1000)), header={k.upper(): v for k, v in hdr.items()})
    res = solve_from_header(img)
    assert res is not None and res.solved and res.fromExisting
    assert res.ra == pytest.approx(83.8, abs=0.05)
    assert res.dec == pytest.approx(-5.4, abs=0.05)
    assert res.pixScaleAsec == pytest.approx(2.0, abs=0.02)
    assert res.width == 1000 and res.height == 800
    assert math.isfinite(res.fovDeg) and res.fovDeg > 0


def test_solve_from_header_none_without_wcs():
    img = LoadedImage(data=np.zeros((100, 100)), header={"FILTER": "Ha"})
    assert solve_from_header(img) is None
