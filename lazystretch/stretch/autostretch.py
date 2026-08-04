"""The auto-stretch — the crown jewel (LazyStretch.js:2083-2151, PLAN §4.4).

Measure the channel-averaged median & average deviation, apply a soft shadows knee,
solve for the midtones that put the background at the target level, and bake in a
linked MTF. Faithful to the v1.2.1 soft-knee stretch (the 1.2.1 "posterization fix"
that replaced the hard black clip).

The scalar solve (:func:`solve_stretch`) is factored out from the array application
(:func:`apply_auto_stretch`) so the Layer-1 golden harness can feed it PI-measured
``median``/``avgDev`` directly and reproduce ``c0``/``m`` without an image (PLAN §9.3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..data.loader import LazyStretchData, get_data
from ..stats.measure import avg_dev_channel_avg, median_channel_avg
from ..stats.mtf import find_midtones_balance, mtf


@dataclass(frozen=True)
class StretchResult:
    """What ``applyAutoStretch`` returns (LazyStretch.js:2150) plus ``knee_ran``."""

    median: float
    avgDev: float
    c0: float
    m: float
    knee_ran: bool


def _soft_clip_scalar(t: float, c0: float, e: float) -> float:
    """``softmax0``-style knee: ((t-c0) + sqrt((t-c0)^2 + e^2)) / (2*(1-c0))."""
    d = t - c0
    return (d + math.sqrt(d * d + e * e)) / (2.0 * (1.0 - c0))


def _knee_width(avg_dev: float, data: LazyStretchData) -> float:
    """e = max(0.30 * avgDev, 1e-5) — floored so e^2 stays representable (js:2109-2112)."""
    return max(data.stretch["kneeWidthFactor"] * avg_dev, data.stretch["kneeEpsilonFloor"])


def solve_stretch(
    median: float,
    avg_dev: float,
    shadows_clip: float,
    target_bkg: float,
    data: "LazyStretchData | None" = None,
) -> StretchResult:
    """Solve the stretch parameters from measured stats — no image required.

    ``shadows_clip`` is in avgDev units (negative). Reproduces ``c0`` and the midtones
    ``m`` exactly as ``applyAutoStretch`` does, including the "solve from the softened
    median" fix (LazyStretch.js:2126-2136).
    """
    if data is None:
        data = get_data()
    c0 = min(max(median + shadows_clip * avg_dev, 0.0), 1.0)
    e = _knee_width(avg_dev, data)
    knee_ran = 0.0 < c0 < 1.0
    if knee_ran:
        soft_med = _soft_clip_scalar(median, c0, e)
        if not math.isfinite(soft_med):
            soft_med = median
    else:
        soft_med = median
    m = find_midtones_balance(target_bkg, soft_med)
    return StretchResult(median=median, avgDev=avg_dev, c0=c0, m=m, knee_ran=knee_ran)


def apply_auto_stretch(
    img,
    shadows_clip: float,
    target_bkg: float,
    data: "LazyStretchData | None" = None,
):
    """Apply the auto-stretch to ``img``.

    Parameters
    ----------
    img : numpy.ndarray
        Linear image, float in ``[0, 1]``, mono ``(H, W)`` or RGB ``(H, W, 3)``.
    shadows_clip : float
        Shadows clip in avgDev units — **negative** (e.g. class ``clip`` -1.05..-1.60).
    target_bkg : float
        Target background level in ``[0, 1]`` (class ``bkg``).

    Returns
    -------
    (numpy.ndarray, StretchResult)
        The stretched image (a new array; the input is not modified) and the measured
        parameters.
    """
    if data is None:
        data = get_data()

    a = np.asarray(img, dtype=np.float64)
    median = median_channel_avg(a)
    avgdev = avg_dev_channel_avg(a)
    res = solve_stretch(median, avgdev, shadows_clip, target_bkg, data)

    out = a.copy()
    if res.knee_ran:
        e = _knee_width(avgdev, data)
        # PI builds the PixelMath expression from decimal-rounded constants —
        # c0.toFixed(8), (e*e).toFixed(12), (2*(1-c0)).toFixed(8) (LazyStretch.js:2117-2118)
        # — so its pixels are computed with those truncated values. Emulate toFixed
        # (decimal string -> double) so the array knee is bit-identical to PI's, not
        # merely equal to ~1e-9. The scalar softMed above intentionally stays full
        # precision, matching PI's own full-precision softMed (js:2131-2133).
        c0r = float(f"{res.c0:.8f}")
        e2r = float(f"{e * e:.12f}")
        denr = float(f"{2.0 * (1.0 - res.c0):.8f}")
        d = out - c0r
        out = (d + np.sqrt(d * d + e2r)) / denr
        out = np.clip(out, 0.0, 1.0)  # PixelMath truncate=true (LazyStretch.js:2122)

    # Degenerate m of 0 or 1 would slam the frame to black/white — skip (js:2138-2149).
    if 0.0 < res.m < 1.0 and res.m != 0.5:
        out = mtf(res.m, out)

    return out, res
