"""Geometry tools: interactive crop, fractional edge crop, rotate/flip."""
from __future__ import annotations

import numpy as np

from . import Op, ParamSpec, register


def _crop_rect(img: np.ndarray, p: dict) -> np.ndarray:
    """Crop to a fractional rectangle ``{x0,y0,x1,y1}`` in [0,1] of the frame."""
    r = p.get("rect") or {}
    x0 = float(r.get("x0", 0.0)); y0 = float(r.get("y0", 0.0))
    x1 = float(r.get("x1", 1.0)); y1 = float(r.get("y1", 1.0))
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    h, w = img.shape[:2]
    ix0 = int(round(x0 * w)); ix1 = int(round(x1 * w))
    iy0 = int(round(y0 * h)); iy1 = int(round(y1 * h))
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:      # degenerate selection → no-op
        return img
    return np.ascontiguousarray(img[iy0:iy1, ix0:ix1])


def _edge_crop(img: np.ndarray, p: dict) -> np.ndarray:
    """Trim an equal fraction off all four edges (percent → fraction)."""
    frac = float(p.get("percent", 0.0)) / 100.0
    if frac <= 0.0:
        return img
    h, w = img.shape[:2]
    my = int(round(frac * h)); mx = int(round(frac * w))
    if h - 2 * my < 2 or w - 2 * mx < 2:
        return img
    return np.ascontiguousarray(img[my:h - my, mx:w - mx])


def _orient(img: np.ndarray, p: dict) -> np.ndarray:
    out = img
    # "rotate" is a choice index: 0=0°, 1=90° CCW, 2=180°, 3=270° CCW → np.rot90 k.
    k = int(p.get("rotate", 0)) % 4
    if k:
        out = np.rot90(out, k=k)
    if p.get("flip_h"):
        out = out[:, ::-1]
    if p.get("flip_v"):
        out = out[::-1, :]
    return np.ascontiguousarray(out)


register(Op(
    "crop", "Crop", "Geometry", _crop_rect,
    params=[ParamSpec("rect", "Rectangle", "rect",
                      default={"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
                      tooltip="Drag a rectangle on the image to set the crop.")],
    tooltip="Crop to a rectangle you drag on the image.",
))

register(Op(
    "edge_crop", "Edge crop", "Geometry", _edge_crop,
    params=[ParamSpec("percent", "Trim each edge %", "float", 0.0, 25.0, 0.0, 1)],
    tooltip="Trim a fixed percentage off all four edges (stacking artefacts).",
))

register(Op(
    "orient", "Rotate / flip", "Geometry", _orient,
    params=[
        ParamSpec("rotate", "Rotate", "choice", default=0,
                  choices=["0°", "90° CCW", "180°", "270° CCW"]),
        ParamSpec("flip_h", "Flip horizontal", "bool", default=False),
        ParamSpec("flip_v", "Flip vertical", "bool", default=False),
    ],
    tooltip="Rotate in 90° steps and/or mirror the frame.",
))
