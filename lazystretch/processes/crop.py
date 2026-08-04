"""Fractional edge crop — PI ``Pipe.cropEdges`` (LazyStretch.js:2284-2301).

Trims a fixed fraction off **each** edge to remove ragged low-coverage stacking
borders and heavy vignette corners (the root cause of edge gradients and
GradientCorrection corner blowups). Runs first in the pipeline so background
modelling sees clean data.

Fidelity note on the ``fraction`` argument
------------------------------------------
Inside PI's ``cropEdges`` the parameter is a **ratio in [0, 1)** — the margin is
``mx = floor(w * fraction)`` with **no** ``/100`` inside the function. The
percentage-to-ratio conversion happens at the call site::

    let c = Pipe.cropEdges( target, effCrop / 100 );   // LazyStretch.js:3332

where ``effCrop`` is the user-facing percentage (0..10). This port keeps that
contract: ``crop_edges(img, fraction)`` expects the **ratio** (e.g. pass
``effCrop / 100.0``). The caller owns the ``/100`` exactly as PI does.

PI's ``Crop`` with negative ``AbsolutePixels`` margins ``-mx`` (left/right) and
``-my`` (top/bottom) removes ``mx`` columns from each side and ``my`` rows from
top and bottom. In NumPy row/col terms that is ``img[my:h-my, mx:w-mx]`` with
``w = width = axis 1`` and ``h = height = axis 0``. Crop does not alter pixel
values, so there is no ``[0,1]`` truncation to apply.
"""
from __future__ import annotations

import math

import numpy as np


def _margins(w: int, h: int, fraction: float) -> tuple[int, int]:
    """Per-edge pixel margins, mirroring PI's ``floor(dim * fraction)`` exactly."""
    mx = int(math.floor(w * fraction))
    my = int(math.floor(h * fraction))
    return mx, my


def crop_edges(img, fraction) -> np.ndarray:
    """Trim ``fraction`` (a ratio in [0, 1)) off each edge; return a NEW array.

    Faithful port of ``Pipe.cropEdges`` (LazyStretch.js:2284-2301). Mono
    ``(H, W)`` or RGB ``(H, W, 3)``. The input is never mutated.

    No-op cases (return an unchanged **copy**, matching PI returning ``null`` and
    leaving the view untouched):
      * ``fraction`` not ``> 0`` (covers 0, negatives, and NaN — PI's ``!(x > 0)``);
      * both margins collapse to zero, i.e. ``mx <= 0 and my <= 0``.

    Note the ``and``: PI crops when *either* axis has a positive margin, so a very
    non-square frame can be cropped on one axis only. Replicated here.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D (mono) or 3-D (RGB) image, got shape {a.shape}")

    # PI: if ( !( fraction > 0 ) ) return null;  -> no-op unchanged copy.
    if not (fraction > 0):
        return a.copy()

    h, w = a.shape[0], a.shape[1]
    mx, my = _margins(w, h, fraction)

    # PI: if ( mx <= 0 && my <= 0 ) return null;  -> no-op unchanged copy.
    if mx <= 0 and my <= 0:
        return a.copy()

    cropped = a[my:h - my, mx:w - mx]
    # Slicing yields a view; return an independent contiguous NEW array.
    return np.ascontiguousarray(cropped)


def crop_edges_with_offsets(img, fraction) -> tuple[np.ndarray, dict]:
    """Like :func:`crop_edges`, plus the crop offsets so the caller can fix WCS.

    Returns ``(cropped, info)`` where ``info`` mirrors PI's return dict
    ``{ x, y, newW, newH }`` (LazyStretch.js:2300) using explicit keys:

      * ``x0`` — columns removed from the LEFT edge (== PI ``x`` == ``mx``);
      * ``y0`` — rows removed from the TOP edge (== PI ``y`` == ``my``);
      * ``new_w`` / ``new_h`` — output width/height (== PI ``newW`` / ``newH``);
      * ``cropped`` — ``True`` if any pixels were removed, ``False`` on a no-op.

    WCS update for the caller: the new pixel ``(0, 0)`` was old pixel
    ``(mx, my)``, so with 1-based FITS ``CRPIX``::

        CRPIX1 -= x0    # x0 == mx (columns)
        CRPIX2 -= y0    # y0 == my (rows)

    On a no-op, ``x0 == y0 == 0`` and ``new_w`` / ``new_h`` are the input dims.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D (mono) or 3-D (RGB) image, got shape {a.shape}")

    h, w = a.shape[0], a.shape[1]

    if not (fraction > 0):
        return a.copy(), {"x0": 0, "y0": 0, "new_w": w, "new_h": h, "cropped": False}

    mx, my = _margins(w, h, fraction)
    if mx <= 0 and my <= 0:
        return a.copy(), {"x0": 0, "y0": 0, "new_w": w, "new_h": h, "cropped": False}

    cropped = np.ascontiguousarray(a[my:h - my, mx:w - mx])
    info = {"x0": mx, "y0": my, "new_w": w - 2 * mx, "new_h": h - 2 * my, "cropped": True}
    return cropped, info
