"""Self-calibrating measurements: dust lift + the auto crop/GC picker.

  * :func:`measure_dust`        — LazyStretch.js:2162-2193
  * :func:`adaptive_floor_raise`— LazyStretch.js:2198-2201
  * :func:`auto_assess`         — LazyStretch.js:2544-2636

All constants come from the shared data layer. Pixel-rect rounding uses PI's
``Math.round`` (round-half-up), not Python's banker's rounding, so patch boundaries
match the source exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..data.loader import LazyStretchData, get_data
from ..stats.mtf import find_midtones_balance, mtf_scalar


def _jsround(x: float) -> int:
    """JavaScript ``Math.round`` — floor(x + 0.5) (round half toward +inf)."""
    return int(math.floor(x + 0.5))


@dataclass(frozen=True)
class DustResult:
    lift: float
    cleanSky: float
    typical: float


@dataclass(frozen=True)
class AssessResult:
    cropPercent: int
    useGC: bool
    gradient: float
    vmin: float
    vmax: float


def _nchan(a: np.ndarray) -> int:
    return 1 if a.ndim == 2 else a.shape[2]


def _region_channel_medians(a: np.ndarray, x0: int, y0: int, x1: int, y1: int):
    """Per-channel medians inside the half-open rect (list, one per channel)."""
    if a.ndim == 2:
        return [float(np.median(a[y0:y1, x0:x1]))]
    return [float(np.median(a[y0:y1, x0:x1, c])) for c in range(a.shape[2])]


def measure_dust(img, data: "LazyStretchData | None" = None) -> DustResult:
    """How much brighter the typical background is than this image's own cleanest sky.

    Measured on the (already stretched) image. Zero on a clean field by construction.
    """
    if data is None:
        data = get_data()
    md = data.measure_dust
    a = np.asarray(img, dtype=np.float64)
    H, W = a.shape[0], a.shape[1]
    n_ch = _nchan(a)

    gx, gy = md["gridX"], md["gridY"]
    pw, ph = md["patchW"], md["patchH"]
    span_lo, span_hi = md["spanLo"], md["spanHi"]
    span = span_hi - span_lo
    min_px = md["minPatchPx"]

    vals = []
    for iy in range(gy):
        for ix in range(gx):
            cx = span_lo + span * (ix / (gx - 1))
            cy = span_lo + span * (iy / (gy - 1))
            x0 = max(0, _jsround((cx - pw / 2) * W))
            y0 = max(0, _jsround((cy - ph / 2) * H))
            x1 = min(W, _jsround((cx + pw / 2) * W))
            y1 = min(H, _jsround((cy + ph / 2) * H))
            if x1 - x0 < min_px or y1 - y0 < min_px:
                continue
            chan_meds = _region_channel_medians(a, x0, y0, x1, y1)
            vals.append(sum(chan_meds) / n_ch)

    if len(vals) < md["minValidPatches"]:
        return DustResult(0.0, 0.0, 0.0)

    vals.sort()
    n = len(vals)

    def q(p: float) -> float:
        return vals[min(n - 1, max(0, int(math.floor(p * (n - 1)))))]

    clean_sky = q(md["cleanSkyPercentile"])
    typical = q(md["typicalPercentile"])
    return DustResult(lift=max(0.0, typical - clean_sky), cleanSky=clean_sky, typical=typical)


def adaptive_floor_raise(lift: float, data: "LazyStretchData | None" = None) -> float:
    """Map the dust lift to a bounded background-floor raise: clip(lift, 0, cap)."""
    if data is None:
        data = get_data()
    cap = data.adaptive_floor["raiseCap"]
    return float(min(max(lift, 0.0), cap))


def auto_assess(img, data: "LazyStretchData | None" = None) -> Optional[AssessResult]:
    """Perimeter-opposition crop % + GradientCorrection picker (on the LINEAR image).

    Returns ``None`` for frames too small to place 8 non-degenerate perimeter patches
    (caller keeps manual settings).
    """
    if data is None:
        data = get_data()
    aa = data.auto_assess
    a = np.asarray(img, dtype=np.float64)
    H, W = a.shape[0], a.shape[1]
    n_ch = _nchan(a)

    if W < aa["minFrame"] or H < aa["minFrame"]:
        return None

    sigma = aa["sigma"]            # -2.8
    target = aa["targetBkg"]       # 0.25
    sz = aa["patchSize"]           # 0.09
    inset = aa["inset"]            # 0.02
    h = sz / 2.0

    def mk_rect(x0, y0, x1, y1):
        ax0, ay0 = max(0, _jsround(x0)), max(0, _jsround(y0))
        ax1, ay1 = min(W, _jsround(x1)), min(H, _jsround(y1))
        if ax1 - ax0 < 4:
            ax0 = max(0, min(ax0, W - 4))
            ax1 = ax0 + 4
        if ay1 - ay0 < 4:
            ay0 = max(0, min(ay0, H - 4))
            ay1 = ay0 + 4
        return ax0, ay0, ax1, ay1

    # Per-channel STF params to map region medians into post-stretch (visual) space.
    c0 = []
    mm = []
    for c in range(n_ch):
        chan = a if a.ndim == 2 else a[..., c]
        med = float(np.median(chan))
        dev = float(np.mean(np.abs(chan - med)))
        cc = float(np.clip(med + sigma * dev, 0.0, 1.0))
        c0.append(cc)
        mm.append(find_midtones_balance(target, med - cc))

    def vlum(x0, y0, x1, y1) -> float:
        rx0, ry0, rx1, ry1 = mk_rect(x0, y0, x1, y1)
        s = 0.0
        meds = _region_channel_medians(a, rx0, ry0, rx1, ry1)
        for c in range(n_ch):
            denom = 1.0 - c0[c]
            x = (meds[c] - c0[c]) / denom if denom > 1e-8 else 0.0
            x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
            v = mtf_scalar(mm[c], x)
            if math.isfinite(v):
                s += v
        return s / n_ch

    e = inset
    R = [
        vlum(W * e, H * e, W * (e + sz), H * (e + sz)),              # top-left
        vlum(W * (1 - e - sz), H * e, W * (1 - e), H * (e + sz)),    # top-right
        vlum(W * e, H * (1 - e - sz), W * (e + sz), H * (1 - e)),    # bottom-left
        vlum(W * (1 - e - sz), H * (1 - e - sz), W * (1 - e), H * (1 - e)),  # bottom-right
        vlum(W * (0.5 - h), H * e, W * (0.5 + h), H * (e + sz)),     # top-mid
        vlum(W * (0.5 - h), H * (1 - e - sz), W * (0.5 + h), H * (1 - e)),   # bottom-mid
        vlum(W * e, H * (0.5 - h), W * (e + sz), H * (0.5 + h)),     # left-mid
        vlum(W * (1 - e - sz), H * (0.5 - h), W * (1 - e), H * (0.5 + h)),   # right-mid
    ]

    R.sort()
    drop = aa["dropBrightest"]  # 1 -> use 2nd-brightest as the bright reference
    vmin = R[0]
    vmax = R[len(R) - 1 - drop]
    gradient = (vmax - vmin) / vmax if vmax > 1e-4 else 0.0
    if not gradient > 0:
        gradient = 0.0

    chosen = None
    for tier in sorted(aa["tiers"], key=lambda t: t["minGradient"]):
        if gradient >= tier["minGradient"]:
            chosen = tier
    return AssessResult(
        cropPercent=chosen["cropPercent"],
        useGC=chosen["useGC"],
        gradient=gradient,
        vmin=vmin,
        vmax=vmax,
    )
