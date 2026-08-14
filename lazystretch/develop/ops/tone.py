"""Tone tools: curves, levels, S-curve contrast, local contrast, HDR, roll-off, deepen.

These are *finishing* tools that assume an already-stretched (non-linear) image — the
initial STF/histogram stretch belongs to LazyStretch, not here.
"""
from __future__ import annotations

import numpy as np

from ...processes import tone as _tone
from ...processes.highlights import highlight_rolloff
from ...processes.deepen import deepen_stretch
from ...processes.multiscale import hdr_core
from ..curves import apply_curve
from . import Op, ParamSpec, register


def _curves(img: np.ndarray, p: dict) -> np.ndarray:
    pts = p.get("points") or [[0.0, 0.0], [1.0, 1.0]]
    modes = ["rgb", "lum", "r", "g", "b"]
    ch = modes[int(p.get("channel", 0))] if img.ndim == 3 else "rgb"
    return apply_curve(img, pts, ch)


def _levels(img: np.ndarray, p: dict) -> np.ndarray:
    """Manual black/white points + gamma — a finishing levels control (not an STF)."""
    black = float(p.get("black", 0.0))
    white = float(p.get("white", 1.0))
    gamma = max(0.05, float(p.get("gamma", 1.0)))
    if white <= black + 1e-4:
        white = black + 1e-4
    out = (np.asarray(img, dtype=np.float32) - black) / (white - black)
    out = np.clip(out, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-4:
        out = np.power(out, 1.0 / gamma, dtype=np.float32)
    return out


def _contrast(img: np.ndarray, p: dict) -> np.ndarray:
    return _tone.contrast(img, float(p.get("pivot", 0.12)), float(p.get("strength", 0.15)))


def _local_contrast(img: np.ndarray, p: dict) -> np.ndarray:
    return _tone.local_contrast(img, float(p.get("amount", 0.12)))


def _hdr(img: np.ndarray, p: dict) -> np.ndarray:
    return hdr_core(img, int(p.get("layers", 6)),
                    compression=float(p.get("compression", 0.85)),
                    detail_boost=float(p.get("detail_boost", 1.0)))


def _rolloff(img: np.ndarray, p: dict) -> np.ndarray:
    return highlight_rolloff(img, knee=float(p.get("knee", 0.82)))


def _deepen(img: np.ndarray, p: dict) -> np.ndarray:
    return deepen_stretch(img, float(p.get("amount", 0.0)))


register(Op(
    "curves", "Curves", "Tone", _curves,
    params=[
        ParamSpec("channel", "Channel", "choice", default=0,
                  choices=["RGB", "Luminance", "Red", "Green", "Blue"]),
        ParamSpec("points", "Curve", "curve", default=[[0.0, 0.0], [1.0, 1.0]],
                  tooltip="Drag points on the curve; double-click to add/remove."),
    ],
    tooltip="Photoshop/Lightroom-style tone curve (RGB, luminance, or a channel).",
))

register(Op(
    "levels", "Levels", "Tone", _levels,
    params=[
        ParamSpec("black", "Black point", "float", 0.0, 0.5, 0.0, 3),
        ParamSpec("white", "White point", "float", 0.5, 1.0, 1.0, 3),
        ParamSpec("gamma", "Gamma (midtones)", "float", 0.2, 3.0, 1.0, 2),
    ],
    tooltip="Manual black/white points and gamma. Finishing only (not an auto-stretch).",
))

register(Op(
    "contrast", "S-curve contrast", "Tone", _contrast,
    params=[
        ParamSpec("pivot", "Pivot (background)", "float", 0.0, 0.5, 0.12, 2),
        ParamSpec("strength", "Strength", "float", -0.3, 0.3, 0.15, 2),
    ],
    tooltip="Gentle S-curve around the background level (Akima-spline contrast).",
))

register(Op(
    "local_contrast", "Local contrast", "Tone", _local_contrast,
    params=[ParamSpec("amount", "Amount", "float", 0.0, 0.6, 0.12, 2)],
    tooltip="Unsharp-mask local contrast / clarity on luminance.",
))

register(Op(
    "hdr", "HDR core compression", "Tone", _hdr, heavy=True,
    params=[
        ParamSpec("layers", "Layers", "int", 3, 8, 6, 0),
        ParamSpec("compression", "Compression", "float", 0.3, 1.0, 0.85, 2),
        ParamSpec("detail_boost", "Detail boost", "float", 0.5, 2.0, 1.0, 2),
    ],
    tooltip="à-trous HDR: compress bright cores, keep detail (hue-preserving).",
))

register(Op(
    "highlight_rolloff", "Highlight roll-off", "Tone", _rolloff,
    params=[ParamSpec("knee", "Knee", "float", 0.1, 0.95, 0.82, 2)],
    tooltip="Dial down blown highlights and reveal core structure.",
))

register(Op(
    "deepen", "Deepen (2nd pass)", "Tone", _deepen,
    params=[ParamSpec("amount", "Amount", "float", 0.0, 1.0, 0.3, 2)],
    tooltip="Deterministic second-pass MTF lift on faint signal.",
))
