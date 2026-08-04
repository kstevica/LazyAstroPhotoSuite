"""Curated starting recipes (LazyStretch.js:960-1005, v1.4.0).

Target-intrinsic starting points for well-known objects, applied as: memory > curated >
class defaults. Each entry carries only intrinsic settings — condition-driven choices
(LP gradient, frame fill, SNR) belong to the runtime gates / Analyze. ``curated_for``
matches on the leading catalog designation only, tolerating spacing.
"""
from __future__ import annotations

from typing import Dict, List, Optional

CURATED_RECIPES: List[Dict] = [
    {
        "ids": ["M42", "M43"], "name": "Orion Nebula",
        "settings": {"dehaze": 0.10, "gradientCleanup": 0.40},
        "why": "haze stays low - the warm outer wings are the subject; HDR handles the core",
    },
    {
        "ids": ["IC434", "NGC2023", "NGC2024", "IC431", "IC432", "IC435"],
        "name": "Horsehead / Flame region",
        "settings": {"satAdj": 0.18, "brightAdj": 0.01, "bgAdj": 0.08, "blackAdj": -0.19,
                     "contrastAdj": 0.15, "gradientCleanup": 0.50, "dehaze": 0.79,
                     "starsAdj": 0.5, "reduceCast": True},
        "why": "lifted soft background keeps the molecular dust visible; high haze neutralises the brown",
    },
    {
        "ids": ["IC1318", "NGC6910"], "name": "Sadr region (wall-to-wall Ha)",
        "settings": {"autoAssess": False, "cropPercent": 10, "bgAdj": 0.12, "useMask": False,
                     "adaptiveFloor": False, "darkLaneGC": True, "doSCNR": True,
                     "gradientCleanup": 0.50, "dehaze": 0.15, "satAdj": 0.30, "reduceCast": False},
        "why": "no empty sky - the median IS the subject: floor raised, median-based tools off, dark-lane on",
    },
    {
        "ids": ["IC2118"], "name": "Witch Head Nebula",
        "settings": {"satAdj": 0.40, "brightAdj": 0.02, "bgAdj": 0.15, "blackAdj": 0.08,
                     "contrastAdj": 0.15, "gradientCleanup": 0.40, "dehaze": 0.40, "smallStars": 0.35,
                     "starsAdj": 0.30, "useMask": False, "adaptiveFloor": False, "reduceCast": False,
                     "enhanceEmission": True, "darkLaneGC": True},
        "why": "dark moody dust field - floor raised so global darkening can't crush the frame-filling dust",
    },
    {
        "ids": ["NGC2237", "NGC2238", "NGC2239", "NGC2244", "NGC2246"], "name": "Rosette Nebula",
        "settings": {"satAdj": 0.20, "contrastAdj": 0.12},
        "why": "class defaults nearly fit; gentle saturation + contrast - let Analyze add the LP/SNR tools",
    },
]


def _norm_id(s: str) -> str:
    """Leading catalog designation, spacing removed: 'IC 434 — Horsehead' -> 'IC434'."""
    return s.split("—")[0].strip().upper().replace(" ", "")


def curated_for(object_id: Optional[str]) -> Optional[Dict]:
    """Return the curated recipe whose ids include ``object_id``, or ``None`` (js:983)."""
    if not object_id:
        return None
    key = _norm_id(object_id)
    for r in CURATED_RECIPES:
        if any(_norm_id(i) == key for i in r["ids"]):
            return r
    return None


def apply_preset(params, settings: Dict) -> None:
    """Apply a curated ``settings`` dict onto a Parameters object (in place)."""
    for k, v in settings.items():
        if hasattr(params, k):
            setattr(params, k, v)
