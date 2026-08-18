"""Colour tools: saturation, white balance, SCNR, chroma NR, background neutralise,
reduce cast, emission boost. All are RGB-only (no-op copy on mono)."""
from __future__ import annotations

import numpy as np

from ...processes import tone as _tone
from ...processes.scnr import scnr
from ...processes.chromanr import chroma_nr
from ...processes.colorcal import background_neutralize
from ...processes.finishing import reduce_cast, emission_boost
from ...processes.chroma import correct_chromatic_aberration
from . import Op, ParamSpec, register

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _saturation(img: np.ndarray, p: dict) -> np.ndarray:
    return _tone.saturation(img, float(p.get("amount", 0.0)))


def _white_balance(img: np.ndarray, p: dict) -> np.ndarray:
    """Photographic temperature/tint nudge; luminance held roughly constant."""
    if img.ndim != 3:
        return img
    temp = float(p.get("temp", 0.0))   # + warmer (more red), − cooler (more blue)
    tint = float(p.get("tint", 0.0))   # + magenta,          − green
    gain = np.array([1.0 + 0.30 * temp,
                     1.0 - 0.15 * tint,
                     1.0 - 0.30 * temp], dtype=np.float32)
    out = np.asarray(img, dtype=np.float32) * gain
    # Rescale so the frame's mean luminance is unchanged by the colour shift.
    l0 = float((np.asarray(img, dtype=np.float32) * _LUMA).sum(-1).mean())
    l1 = float((out * _LUMA).sum(-1).mean())
    if l1 > 1e-6:
        out *= (l0 / l1)
    return np.clip(out, 0.0, 1.0)


def _scnr(img: np.ndarray, p: dict) -> np.ndarray:
    return scnr(img, amount=float(p.get("amount", 1.0)),
                preserve_lightness=bool(p.get("preserve_lightness", True)))


def _chroma_nr(img: np.ndarray, p: dict) -> np.ndarray:
    return chroma_nr(img, float(p.get("amount", 0.5)))


def _bg_neutralize(img: np.ndarray, p: dict) -> np.ndarray:
    return background_neutralize(img, target=float(p.get("target", 0.05)))


def _reduce_cast(img: np.ndarray, p: dict) -> np.ndarray:
    return reduce_cast(img)


def _fix_chromatic(img: np.ndarray, p: dict) -> np.ndarray:
    return correct_chromatic_aberration(img, strength=float(p.get("strength", 1.0)))


def _emission(img: np.ndarray, p: dict) -> np.ndarray:
    return emission_boost(img)


register(Op(
    "saturation", "Saturation", "Color", _saturation, needs_color=True,
    params=[ParamSpec("amount", "Amount", "float", -1.0, 1.0, 0.0, 2)],
    tooltip="Chroma boost in CIE L*c*h* (hue/lightness preserved).",
))

register(Op(
    "white_balance", "White balance", "Color", _white_balance, needs_color=True,
    params=[
        ParamSpec("temp", "Temperature", "float", -1.0, 1.0, 0.0, 2),
        ParamSpec("tint", "Tint", "float", -1.0, 1.0, 0.0, 2),
    ],
    tooltip="Temperature (red↔blue) and tint (magenta↔green) nudge; luminance held.",
))

register(Op(
    "scnr", "SCNR (remove green)", "Color", _scnr, needs_color=True,
    params=[
        ParamSpec("amount", "Amount", "float", 0.0, 1.0, 1.0, 2),
        ParamSpec("preserve_lightness", "Preserve lightness", "bool", default=True),
    ],
    tooltip="Average-neutral green removal (kills the green cast).",
))

register(Op(
    "chroma_nr", "Chroma noise reduction", "Color", _chroma_nr, needs_color=True, heavy=True,
    params=[ParamSpec("amount", "Amount", "float", 0.0, 1.0, 0.5, 2)],
    tooltip="Multiscale chroma-only NR (removes colour speckle; luminance untouched).",
))

register(Op(
    "bg_neutralize", "Background neutralize", "Color", _bg_neutralize, needs_color=True,
    params=[ParamSpec("target", "Target level", "float", 0.0, 0.2, 0.05, 3)],
    tooltip="Drive the darkest sky background to a neutral grey level.",
))

register(Op(
    "reduce_cast", "Reduce colour cast", "Color", _reduce_cast, needs_color=True,
    params=[],
    tooltip="Pull the faint sky 30% toward grey through a background mask.",
))

register(Op(
    "fix_chromatic", "Fix chromatic aberration", "Color", _fix_chromatic, needs_color=True,
    heavy=True,
    params=[ParamSpec("strength", "Strength", "float", 0.0, 1.0, 1.0, 2)],
    tooltip="Align the R and G channels to B from star offsets — removes lateral-CA colour "
            "fringes and the dark star crescents sharpening can leave.",
))

register(Op(
    "emission_boost", "Enhance emission (Hα)", "Color", _emission, needs_color=True,
    params=[],
    tooltip="Saturation + gentle contrast through a red-excess (Hα) mask.",
))
