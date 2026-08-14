"""The Develop tool registry.

Every interactive tool is an :class:`Op`: a name, a human label, a category, a list
of :class:`ParamSpec` (so the GUI can build a control panel generically), and a pure
``fn(img, params) -> img``. Submodules register their ops at import time; importing
this package (``lazystretch.develop.ops``) populates :data:`REGISTRY`.

The GUI never hard-codes a tool: it walks ``by_category()`` to build the toolbox and
``op.params`` to build each tool's sliders/checkboxes/curve editors. Adding a tool is
one ``register(Op(...))`` call — no GUI change required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

# Category display order in the toolbox.
CATEGORY_ORDER = ["Geometry", "Tone", "Color", "Detail", "Background", "Lighthouse"]


@dataclass
class ParamSpec:
    """One tunable of a tool. ``kind`` tells the GUI which widget to build."""
    key: str
    label: str
    kind: str = "float"            # float | int | bool | choice | curve | rect
    lo: float = 0.0
    hi: float = 1.0
    default: object = 0.0
    decimals: int = 2
    choices: Optional[List[str]] = None   # for kind == "choice" (values are indices)
    tooltip: str = ""

    def coerce(self, v):
        if self.kind == "bool":
            return bool(v)
        if self.kind == "int":
            return int(round(float(v)))
        if self.kind == "choice":
            return int(v)
        if self.kind in ("curve", "rect"):
            return v
        return float(v)


@dataclass
class Op:
    """A registered tool."""
    name: str
    label: str
    category: str
    fn: Callable[[np.ndarray, dict], np.ndarray]
    params: List[ParamSpec] = field(default_factory=list)
    heavy: bool = False            # run in a worker thread (slow: wavelets, ABE, NR)
    needs_color: bool = False      # disabled / no-op on mono images
    tooltip: str = ""

    def defaults(self) -> dict:
        return {p.key: p.default for p in self.params}

    def merged(self, params: Optional[dict]) -> dict:
        """Fill missing keys with defaults and coerce every value to its kind."""
        out = self.defaults()
        if params:
            out.update(params)
        for p in self.params:
            if p.key in out and out[p.key] is not None:
                out[p.key] = p.coerce(out[p.key])
        return out


REGISTRY: Dict[str, Op] = {}


def register(op: Op) -> Op:
    if op.name in REGISTRY:
        raise ValueError(f"duplicate Develop op: {op.name}")
    REGISTRY[op.name] = op
    return op


def get(name: str) -> Op:
    return REGISTRY[name]


def all_ops() -> List[Op]:
    return list(REGISTRY.values())


def by_category() -> "Dict[str, List[Op]]":
    """Ops grouped by category, in CATEGORY_ORDER (unknown categories appended)."""
    groups: Dict[str, List[Op]] = {}
    for op in REGISTRY.values():
        groups.setdefault(op.category, []).append(op)
    ordered: Dict[str, List[Op]] = {}
    for cat in CATEGORY_ORDER:
        if cat in groups:
            ordered[cat] = groups[cat]
    for cat, ops in groups.items():
        if cat not in ordered:
            ordered[cat] = ops
    return ordered


# --- import side-effects: each module registers its ops ---------------------
from . import geometry   # noqa: E402,F401
from . import tone       # noqa: E402,F401
from . import color      # noqa: E402,F401
from . import detail     # noqa: E402,F401
from . import background  # noqa: E402,F401
from . import lighthouse  # noqa: E402,F401

__all__ = [
    "ParamSpec", "Op", "REGISTRY", "register", "get", "all_ops", "by_category",
    "CATEGORY_ORDER",
]
