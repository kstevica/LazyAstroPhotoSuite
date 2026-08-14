"""Background / gradient tools: ABE-style extraction, gradient cleanup, de-veil.

ABE (polynomial background extraction) is really a linear-stage tool; it is offered
here for touch-ups, but on an already-stretched image prefer *Gradient cleanup* or
*De-veil* for residual sky gradients and milky veils.
"""
from __future__ import annotations

import numpy as np

from ...processes.background import background_extract
from ...processes.finishing import dehaze, gradient_cleanup
from ...processes.deveil import deepen_background
from . import Op, ParamSpec, register


def _abe(img: np.ndarray, p: dict) -> np.ndarray:
    return background_extract(img, int(p.get("degree", 4)),
                              grid=int(p.get("grid", 64)), normalize=True)


def _gradient_cleanup(img: np.ndarray, p: dict) -> np.ndarray:
    return gradient_cleanup(img, float(p.get("strength", 0.5)))


def _dehaze(img: np.ndarray, p: dict) -> np.ndarray:
    return dehaze(img, float(p.get("strength", 0.5)), float(p.get("floor", 0.05)),
                  "generic", gradient=float(p.get("gradient", 0.0)))


def _deveil(img: np.ndarray, p: dict) -> np.ndarray:
    return deepen_background(img, float(p.get("amount", 0.5)),
                             target=float(p.get("target", 0.05)),
                             flatten=float(p.get("flatten", 0.6)))


register(Op(
    "abe", "Background extraction (ABE)", "Background", _abe, heavy=True,
    params=[
        ParamSpec("degree", "Polynomial degree", "int", 1, 6, 4, 0),
        ParamSpec("grid", "Sample grid", "int", 16, 128, 64, 0),
    ],
    tooltip="Fit and subtract a polynomial background surface (per channel).",
))

register(Op(
    "gradient_cleanup", "Gradient cleanup", "Background", _gradient_cleanup,
    params=[ParamSpec("strength", "Strength", "float", 0.0, 1.0, 0.5, 2)],
    tooltip="Robust degree-1 plane-fit gradient removal.",
))

register(Op(
    "dehaze", "Dehaze", "Background", _dehaze,
    params=[
        ParamSpec("strength", "Strength", "float", 0.0, 1.0, 0.5, 2),
        ParamSpec("floor", "Sky floor", "float", 0.0, 0.2, 0.05, 3),
        ParamSpec("gradient", "Gradient", "float", 0.0, 1.0, 0.0, 2),
    ],
    tooltip="Two-pass soft veil subtraction with brownness desaturation.",
))

register(Op(
    "deveil", "De-veil background", "Background", _deveil,
    params=[
        ParamSpec("amount", "Amount", "float", 0.0, 1.0, 0.5, 2),
        ParamSpec("target", "Target floor", "float", 0.0, 0.2, 0.05, 3),
        ParamSpec("flatten", "Flatten tilt", "float", 0.0, 1.0, 0.6, 2),
    ],
    tooltip="Colour-preservingly deepen a milky background floor (self-limiting).",
))
