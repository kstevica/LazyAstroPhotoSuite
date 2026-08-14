"""Detail tools: mid-scale structure boost and multiscale-median noise reduction.

(à-trous wavelet clarity/sharpen lives in ``ops/lighthouse.py`` — it is a faithful
port of the Lighthouse wavelet engine.)
"""
from __future__ import annotations

import numpy as np

from ...processes.multiscale import structure_boost, noise_reduction_mmt
from ...processes.starreduce import shrink_stars
from ...processes.deconv import richardson_lucy
from ...processes.halo import halo_tamer
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


def _star_reduce(img: np.ndarray, p: dict) -> np.ndarray:
    return shrink_stars(img, float(p.get("amount", 0.5)), small=float(p.get("small", 0.5)))


def _deconvolve(img: np.ndarray, p: dict) -> np.ndarray:
    return richardson_lucy(img, sigma=float(p.get("sigma", 1.2)),
                           iterations=int(p.get("iterations", 10)))


def _halo(img: np.ndarray, p: dict) -> np.ndarray:
    return halo_tamer(img, strength=float(p.get("strength", 1.0)))


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

register(Op(
    "star_reduce", "Reduce stars", "Detail", _star_reduce, heavy=True,
    params=[
        ParamSpec("amount", "Amount", "float", 0.0, 1.0, 0.5, 2),
        ParamSpec("small", "Bias to small stars", "float", 0.0, 1.0, 0.5, 2),
    ],
    tooltip="Size/brightness-tiered star reduction (no StarNet): crush the carpet, "
            "keep medium stars, preserve bright anchors.",
))

register(Op(
    "deconvolve", "Deconvolution (sharpen)", "Detail", _deconvolve, heavy=True,
    params=[
        ParamSpec("sigma", "PSF sigma", "float", 0.5, 3.0, 1.2, 2),
        ParamSpec("iterations", "Iterations", "int", 3, 30, 10, 0),
    ],
    tooltip="Richardson–Lucy deconvolution with a Gaussian PSF (recovers sharpness).",
))

register(Op(
    "halo_tamer", "Halo Tamer", "Detail", _halo, heavy=True,
    params=[ParamSpec("strength", "Strength", "float", 0.0, 1.0, 1.0, 2)],
    tooltip="Detect and suppress reflection filter-ring halos around bright stars.",
))
