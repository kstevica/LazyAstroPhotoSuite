"""LazyDevelop — an interactive, PixInsight-style image-*finishing* module.

The Develop window is the "Lightroom for astro" bench: you open an image that has
already been stretched (a LazyStretch result, or any non-linear master) and finish it
by hand — curves, crop, colour, detail (wavelets / HDR / local contrast), gradient
cleanup, masks — in any order, with a linear undo/redo history. The initial stretch
(STF / histogram transformation) is intentionally NOT part of this window; that is
LazyStretch's job. Every tool is a thin front-end over the headless functions already
used by the automated pipeline (``lazystretch.processes.*``), plus faithful ports of
the Lighthouse suite ("Lightroom for PixInsight"): luminosity masks, à-trous wavelet
clarity and OKLab Selective Color.

Public surface:
    DevelopDocument   — the edit document: base image + ordered op history + masks.
    OpInstance        — one applied tool (name, params, optional mask gate).
    ops.REGISTRY      — every available tool, keyed by name (see ops.all_ops()).
"""
from __future__ import annotations

from .document import DevelopDocument, OpInstance
from . import ops  # noqa: F401  (import side-effect: populates ops.REGISTRY)

__all__ = ["DevelopDocument", "OpInstance", "ops"]
