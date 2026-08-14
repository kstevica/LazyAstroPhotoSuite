"""Detail tools: mid-scale structure boost and multiscale-median noise reduction.

(à-trous wavelet clarity/sharpen lives in ``ops/lighthouse.py`` — it is a faithful
port of the Lighthouse wavelet engine.)
"""
from __future__ import annotations

import numpy as np

from ...processes.multiscale import structure_boost, noise_reduction_mmt
from . import Op, ParamSpec, register


def _structure(img: np.ndarray, p: dict) -> np.ndarray:
    return structure_boost(img, float(p.get("amount", 0.6)))


def _nr(img: np.ndarray, p: dict) -> np.ndarray:
    amount = float(p.get("amount", 1.0))
    nr = noise_reduction_mmt(img)
    if amount >= 0.999:
        return nr
    a = np.asarray(img, dtype=np.float32)
    return np.clip(a + amount * (np.asarray(nr, dtype=np.float32) - a), 0.0, 1.0)


register(Op(
    "structure", "Mid-scale structure", "Detail", _structure, heavy=True,
    params=[ParamSpec("amount", "Amount", "float", 0.0, 1.0, 0.6, 2)],
    tooltip="Boost medium-scale à-trous bands (galaxy arms, dust) — subject-masked.",
))

register(Op(
    "noise_reduction", "Noise reduction (MMT)", "Detail", _nr, heavy=True,
    params=[ParamSpec("amount", "Amount", "float", 0.0, 1.0, 1.0, 2)],
    tooltip="Multiscale-median noise reduction on the two finest scales.",
))
