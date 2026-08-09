"""Load and validate the shared data layer (``lazystretch_data.json``).

This is the single source of truth for every tunable number (PLAN §9.1). Both the
PixInsight ``.js`` and this port are meant to read the same JSON, so a value tweak
propagates with zero re-porting. ``schema_version`` changes only on a *shape* change;
a mismatch here is a hard, explicit error rather than a silent misread.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

SUPPORTED_SCHEMA_VERSION = 1
_DATA_PATH = Path(__file__).with_name("lazystretch_data.json")


@dataclass(frozen=True)
class ClassProfile:
    """One object-class processing profile (LazyStretch.js:948-956)."""

    bkg: float
    clip: float
    sat: float
    contrast: float
    bgLevel: float
    lc: float
    scnr: bool
    localContrast: bool
    bxt: bool
    nr: bool
    hdr: bool
    hdrLayers: Optional[int]
    starReduce: bool
    starLevel: float
    haloTamer: bool = True      # reflection filter-ring suppression (v1.5.1; classes opt out)


@dataclass(frozen=True)
class LazyStretchData:
    """The whole shared data layer, as immutable typed structures."""

    schema_version: int
    source_pi_version: str
    source_sha1: str
    class_list: tuple
    profiles: dict          # class name -> ClassProfile
    slider_map: dict        # adj name -> {"field": str, "effClamp": (lo, hi)}
    recipe_ranges: dict     # dial name -> (lo, hi)
    persisted_reals: tuple
    recipe_bools: tuple
    class_predicates: dict  # predicate name -> frozenset[str]
    catalogs: dict          # messierClass / reflectionCat / emissionCat
    stretch: dict
    auto_assess: dict
    measure_dust: dict
    adaptive_floor: dict

    def profile_for(self, cls: str) -> ClassProfile:
        """``LS.profileFor`` — the class profile, falling back to ``generic``."""
        return self.profiles.get(cls, self.profiles["generic"])


def _strip_comments(d: dict) -> dict:
    """Drop ``_comment`` / underscore-prefixed annotation keys from a dict."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def load_data(path: "str | Path | None" = None) -> LazyStretchData:
    """Parse, validate and structure the shared data JSON."""
    p = Path(path) if path is not None else _DATA_PATH
    with open(p, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    ver = raw.get("schema_version")
    if ver != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"{p.name}: schema_version {ver!r} is not supported by this build "
            f"(expected {SUPPORTED_SCHEMA_VERSION}). Update the loader/dataclasses "
            f"and bump SUPPORTED_SCHEMA_VERSION — see PLAN §9.4 step 4."
        )

    profiles = {
        name: ClassProfile(**prof) for name, prof in raw["profiles"].items()
    }
    slider_map = {
        adj: {"field": spec["field"], "effClamp": tuple(spec["effClamp"])}
        for adj, spec in _strip_comments(raw["sliderMap"]).items()
    }
    recipe_ranges = {
        k: tuple(v) for k, v in _strip_comments(raw["recipeRanges"]).items()
    }
    class_predicates = {
        k: frozenset(v) for k, v in raw["classPredicates"].items()
    }

    return LazyStretchData(
        schema_version=ver,
        source_pi_version=raw.get("source_pi_version", "?"),
        source_sha1=raw.get("source_sha1", "?"),
        class_list=tuple(raw["classList"]),
        profiles=profiles,
        slider_map=slider_map,
        recipe_ranges=recipe_ranges,
        persisted_reals=tuple(raw["persistedReals"]),
        recipe_bools=tuple(raw["recipeBools"]),
        class_predicates=class_predicates,
        catalogs=raw["catalogs"],
        stretch=_strip_comments(raw["stretch"]),
        auto_assess=_strip_comments(raw["autoAssess"]),
        measure_dust=_strip_comments(raw["measureDust"]),
        adaptive_floor=_strip_comments(raw["adaptiveFloor"]),
    )


@lru_cache(maxsize=1)
def get_data() -> LazyStretchData:
    """Return the bundled shared data, parsed once and cached."""
    return load_data()
