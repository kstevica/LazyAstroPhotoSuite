"""Tests for .lsrecipe read/write/apply (io/recipes.py)."""
import json

import pytest

from lazystretch.data.loader import get_data
from lazystretch.io.recipes import (
    apply_recipe,
    load_recipe,
    recipe_from_params,
    save_recipe,
)
from lazystretch.objects.model import Parameters

D = get_data()


def test_recipe_roundtrip(tmp_path):
    p = Parameters.for_object("emission", satAdj=0.2, dehaze=0.5, chromaNR=0.6,
                              darkLaneGC=True, reduceCast=True, palette="SHO (Hubble)")
    path = tmp_path / "r.lsrecipe"
    save_recipe(path, p, D)
    recipe = load_recipe(path)
    assert recipe["lazystretch_recipe"] == 1
    assert "chromaNR" in recipe and "deepen" in recipe and "darkLaneGC" in recipe  # 1.4.x fields

    q = Parameters.for_object("generic")           # a different base class
    n = apply_recipe(q, recipe, D)
    assert n >= 20
    assert q.object_class == "emission" and q.palette == "SHO (Hubble)"
    assert abs(q.satAdj - 0.2) < 1e-9 and abs(q.dehaze - 0.5) < 1e-9 and abs(q.chromaNR - 0.6) < 1e-9
    assert q.darkLaneGC is True and q.reduceCast is True


def test_recipe_excludes_machine_specific():
    p = Parameters.for_object("emission")
    r = recipe_from_params(p, D)
    for k in ("inputStretched", "workOnClone", "preferOnline", "ha", "oiii", "sii"):
        assert k not in r


def test_apply_recipe_clamps_out_of_range():
    p = Parameters.for_object("emission")
    apply_recipe(p, {"lazystretch_recipe": 1, "satAdj": 99.0, "deepen": -5.0}, D)
    assert p.satAdj == 0.5     # recipeRanges satAdj [-0.5, 0.5]
    assert p.deepen == 0.0     # deepen [0, 1]


def test_apply_recipe_tolerant_to_unknown_and_missing():
    p = Parameters.for_object("emission")
    n = apply_recipe(p, {"lazystretch_recipe": 1, "unknownKey": 5, "satAdj": 0.1}, D)
    assert n == 1 and abs(p.satAdj - 0.1) < 1e-9   # unknown ignored; missing keep defaults


def test_load_recipe_rejects_non_recipe(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"foo": 1}))
    with pytest.raises(ValueError):
        load_recipe(bad)


def test_cli_save_and_load_recipe(tmp_path):
    from lazystretch.cli import build_parser, _make_params
    rp = tmp_path / "cli.lsrecipe"
    # build params with some dials, save
    args = build_parser().parse_args(["x.fits", "--class", "galaxy", "--sat", "0.3", "--dark-lane"])
    p = _make_params(args, "galaxy")
    save_recipe(rp, p, D)
    # load via --load-recipe onto a fresh emission base
    args2 = build_parser().parse_args(["x.fits", "--class", "emission", "--load-recipe", str(rp)])
    q = _make_params(args2, "emission")
    assert q.object_class == "galaxy"        # recipe's class wins
    assert abs(q.satAdj - 0.3) < 1e-9 and q.darkLaneGC is True
