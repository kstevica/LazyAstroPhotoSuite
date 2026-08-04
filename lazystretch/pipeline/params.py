"""Effective-value resolution + class predicates.

  * :func:`resolve_effective` — LazyStretch.js:3226-3231 (``effBkg`` … ``effContrast``)
  * :func:`effective_floor`   — the adaptive floor (``effBgLevel + raise``, clamped)
  * class predicates          — LazyStretch.js:2213-2238

Each Adjustment dial is a *relative nudge* on the class profile: ``eff = clip(profile
+ adj, lo, hi)``. Modelled as a pure, data-driven function so it is identical across
the GUI and the headless CLI and trivially unit-testable (PLAN §4.7).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from ..data.loader import ClassProfile, LazyStretchData, get_data


@dataclass(frozen=True)
class Effective:
    """The resolved "look" numbers fed into the pipeline."""

    bkg: float
    sat: float
    clip: float
    bgLevel: float
    contrast: float


def resolve_effective(
    profile: ClassProfile,
    sliders: Optional[Mapping[str, float]] = None,
    data: "LazyStretchData | None" = None,
) -> Effective:
    """Resolve the five effective values from a class profile + adjustment dials.

    ``sliders`` maps dial names (``satAdj``, ``brightAdj``, ``bgAdj``, ``blackAdj``,
    ``contrastAdj``) to nudges; missing dials default to 0 (= class default).
    """
    if data is None:
        data = get_data()
    if sliders is None:
        sliders = {}

    eff = {}
    for adj, spec in data.slider_map.items():
        field = spec["field"]
        lo, hi = spec["effClamp"]
        base = getattr(profile, field)
        eff[field] = float(np.clip(base + float(sliders.get(adj, 0.0)), lo, hi))

    return Effective(
        bkg=eff["bkg"],
        sat=eff["sat"],
        clip=eff["clip"],
        bgLevel=eff["bgLevel"],
        contrast=eff["contrast"],
    )


def effective_floor(
    eff_bg_level: float,
    raise_amount: float,
    data: "LazyStretchData | None" = None,
) -> float:
    """``effFloor = clip(effBgLevel + raise, 0.015, 0.40)`` (LazyStretch.js floor step).

    Uses the same clamp as ``effBgLevel`` (the ``bgAdj`` effective clamp).
    """
    if data is None:
        data = get_data()
    lo, hi = data.slider_map["bgAdj"]["effClamp"]
    return float(np.clip(eff_bg_level + raise_amount, lo, hi))


def _predicate(name: str, cls: str, data: "LazyStretchData | None") -> bool:
    if data is None:
        data = get_data()
    return cls in data.class_predicates[name]


def adaptive_floor_applies_to(cls: str, data: "LazyStretchData | None" = None) -> bool:
    """Floor RAISE only for emission / generic wide fields (LazyStretch.js:2213-2216)."""
    return _predicate("adaptiveFloorAppliesTo", cls, data)


def masked_darken_applies_to(cls: str, data: "LazyStretchData | None" = None) -> bool:
    """Background darkening THROUGH the faint-signal mask (LazyStretch.js:2225-2228)."""
    return _predicate("maskedDarkenAppliesTo", cls, data)


def star_protect_applies_to(cls: str, data: "LazyStretchData | None" = None) -> bool:
    """Saturation THROUGH the inverted star mask (LazyStretch.js:2236-2238)."""
    return _predicate("starProtectAppliesTo", cls, data)
