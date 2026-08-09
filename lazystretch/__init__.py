"""LazyStretch — standalone Python port of the PixInsight LazyStretch suite.

Source of truth: ``reference/pixinsight/LazyStretch.js`` (PI v1.7.0) + ``LazyMoonSun.js``.
This package exposes the deterministic, headless "brain": the pinned MTF primitives,
image statistics, the auto-stretch, the self-calibrating assessments, the effective-value
resolution, the PROC pin ledger, and the LazyMoonSun burst-stacking engine. No PixInsight
dependency. A launcher shell hosts LazyStretch + LazyMoonSun (LazyStack in progress).
"""
from __future__ import annotations

__version__ = "0.1.0"

from .data.loader import ClassProfile, LazyStretchData, get_data, load_data
from .pipeline.params import (
    Effective,
    adaptive_floor_applies_to,
    effective_floor,
    masked_darken_applies_to,
    resolve_effective,
    star_protect_applies_to,
)
from .stats.assess import (
    AssessResult,
    DustResult,
    adaptive_floor_raise,
    auto_assess,
    measure_dust,
)
from .stats.measure import (
    avg_dev,
    avg_dev_channel_avg,
    mean_channel_avg,
    median_channel_avg,
)
from .stats.mtf import find_midtones_balance, mtf, mtf_scalar
from .stretch.autostretch import StretchResult, apply_auto_stretch, solve_stretch

__all__ = [
    "__version__",
    # data
    "ClassProfile", "LazyStretchData", "get_data", "load_data",
    # mtf primitives
    "mtf", "mtf_scalar", "find_midtones_balance",
    # statistics
    "avg_dev", "median_channel_avg", "avg_dev_channel_avg", "mean_channel_avg",
    # assessments
    "measure_dust", "adaptive_floor_raise", "auto_assess", "DustResult", "AssessResult",
    # stretch
    "solve_stretch", "apply_auto_stretch", "StretchResult",
    # effective values + predicates
    "Effective", "resolve_effective", "effective_floor",
    "adaptive_floor_applies_to", "masked_darken_applies_to", "star_protect_applies_to",
]
