"""Read/write ``.lsrecipe`` files — byte-compatible with PixInsight LazyStretch.

Port of the recipe format (LazyStretch.js:200-257): a small JSON object carrying
``lazystretch_recipe: 1`` + version + objectClass + palette + every adjustment dial
(``persistedReals``) + the pipeline toggles (``recipeBools``). Machine/input-specific
settings are excluded so a recipe applies on any machine. Loading is **tolerant**:
unknown keys are ignored and known values are type-checked and clamped to
``recipeRanges`` (forward/backward compatible with PI and future versions).

Because both languages use identical field names, a recipe saved in PixInsight loads
here and vice-versa.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

from .. import __version__
from ..data.loader import LazyStretchData, get_data
from ..objects.model import Parameters

RECIPE_MAGIC = "lazystretch_recipe"


def recipe_from_params(params: Parameters, data: "LazyStretchData | None" = None,
                       *, include_state: bool = False) -> Dict:
    """Build a recipe dict from a Parameters object (LazyStretch.js:218-227).

    ``include_state`` adds input-state / debug bools (``inputStretched``, ``debugBackground``)
    that a *portable* ``.lsrecipe`` must not carry (they describe the input, not the look) but
    that the on-disk **history** should, so restoring a run reproduces its exact settings.
    """
    if data is None:
        data = get_data()
    o: Dict = {
        RECIPE_MAGIC: 1,
        "version": f"lazystretch-py {__version__}",
        "objectClass": params.object_class,
        "palette": params.palette,
    }
    for k in data.persisted_reals:
        o[k] = float(getattr(params, k))
    for k in data.recipe_bools:
        o[k] = bool(getattr(params, k))
    # Port-only finishing dials (not in PI's persistedReals — kept explicit so the PI-mirror
    # data layer stays verbatim; PI ignores unknown keys, and we default them when absent).
    o["highlights"] = float(params.highlights)
    o["transparency"] = float(params.transparency)
    o["dimCore"] = float(params.dimCore)
    o["starShrink"] = float(params.starShrink)
    o["snrProtect"] = float(params.snrProtect)
    o["significance"] = float(params.significance)
    o["meteorStrength"] = float(params.meteorStrength)
    o["structure"] = float(params.structure)
    o["deepenBackground"] = float(params.deepenBackground)
    o["nightscapeBrightness"] = float(params.nightscapeBrightness)
    # Port-only pipeline bools (not in PI's recipeBools) — saved explicitly so the GUI's
    # history/recipe round-trips every exposed option, not just the PI-mirror set.
    for key in _PORT_BOOLS:
        o[key] = bool(getattr(params, key))
    if include_state:                            # history only — kept out of portable recipes
        for key in _STATE_BOOLS:
            o[key] = bool(getattr(params, key))
    return o


# Portable processing bools the GUI exposes that PI's recipeBools omits.
_PORT_BOOLS = ("useClassicalDeconv", "developForeground", "fixChromatic")
# Input-state / debug bools: history captures them, portable .lsrecipe files do not.
_STATE_BOOLS = ("inputStretched", "debugBackground")


def save_recipe(path: "str | Path", params: Parameters,
                data: "LazyStretchData | None" = None) -> str:
    """Write ``params`` as a ``.lsrecipe`` JSON file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(recipe_from_params(params, data), fh, indent=2)
    return str(path)


def load_recipe(path: "str | Path") -> Dict:
    """Read + validate a ``.lsrecipe`` file; returns the raw dict (js:232-235)."""
    with open(path, "r", encoding="utf-8") as fh:
        o = json.load(fh)
    if not isinstance(o, dict) or o.get(RECIPE_MAGIC) != 1:
        raise ValueError(f"{Path(path).name}: not a LazyStretch recipe file")
    return o


def apply_recipe(params: Parameters, recipe: Dict,
                 data: "LazyStretchData | None" = None) -> int:
    """Apply a parsed recipe onto ``params`` in place; returns the count applied.

    Tolerant, mirroring ``applyRecipe`` (js:232-257): each real is type-checked and
    clamped to ``recipeRanges``; each bool is applied only if boolean; objectClass is
    accepted only if in ``classList``; unknown keys are ignored.
    """
    if data is None:
        data = get_data()
    n = 0
    for k in data.persisted_reals:
        v = recipe.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            rng = data.recipe_ranges.get(k)
            setattr(params, k, min(max(float(v), rng[0]), rng[1]) if rng else float(v))
            n += 1
    for k in data.recipe_bools:
        v = recipe.get(k)
        if isinstance(v, bool):
            setattr(params, k, v)
            n += 1
    # Port-only finishing dials (see recipe_from_params): tolerant + clamped, applied only
    # when present so a PI recipe (which lacks them) keeps the Parameters default.
    for key in ("highlights", "transparency", "dimCore", "starShrink", "snrProtect",
                "significance", "meteorStrength", "structure", "deepenBackground",
                "nightscapeBrightness"):
        v = recipe.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            setattr(params, key, min(max(float(v), 0.0), 1.0))
            n += 1
    for key in (*_PORT_BOOLS, *_STATE_BOOLS):    # port-only + state bools (see recipe_from_params)
        v = recipe.get(key)
        if isinstance(v, bool):
            setattr(params, key, v)
            n += 1
    oc = recipe.get("objectClass")
    if isinstance(oc, str) and oc in data.class_list:
        params.object_class = oc
        n += 1
    pal = recipe.get("palette")
    if isinstance(pal, str) and pal:
        params.palette = pal
        n += 1
    return n
