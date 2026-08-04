"""Dark-lane-anchored degree-2 background / gradient model.

Ported from PixInsight LazyStretch v1.4.1:
  * ``Pipe.darkLaneModel``    — LazyStretch.js:2985-3134  (fits the model)
  * ``Pipe.darkLaneGradient`` — LazyStretch.js:3138-3177  (subtracts + truncates)

Why this exists (LazyStretch.js:3138-3140 header comment)
--------------------------------------------------------
For "no empty sky" wall-to-wall-nebula fields (Sadr / IC 1318) the global median
*is* the subject, so ordinary ABE / GradientCorrection over-flatten — they read
the bright nebula as background and crush it.  This model instead anchors on the
**dark lanes**: the darkest structures scattered across the frame.  It fits a
gentle degree-2 surface to those anchors and subtracts only the *excess over the
model's minimum*, so the darkest region keeps its level and nothing shifts
globally — it removes the large-scale tilt without eating the nebula.

How the anchors are located (LazyStretch.js:2993-3024)
------------------------------------------------------
The frame is split into ``ZX=8`` by ``ZY=5`` zones (40 zones).  In each zone a
``CS=3`` by ``CS=3`` grid of *candidate* sub-patches is probed; each candidate is
the centered half of its cell (``cw*0.25 .. cw*0.75``) to avoid cell borders, and
is skipped if smaller than 8 px in either axis.  A candidate's "darkness" is the
mean over channels of the per-channel patch **median**; the single **darkest**
candidate becomes that zone's anchor, positioned at the patch centre in
normalized coordinates ``u = cx/W, v = cy/H``.  Up to one anchor per zone (>= 10
required, else the model is skipped).

The fit (LazyStretch.js:3028-3097)
----------------------------------
Basis ``phi(u,v) = [1, u, v, u^2, u*v, v^2]`` (degree-2 total, 6 terms).  The JS
builds the 6x6 normal equations ``A = phi^T phi``, ``b = phi^T y`` and solves them
with Gaussian elimination + partial pivoting.  This port instead solves the same
least-squares problem via a design-matrix ``np.linalg.lstsq`` over the anchor
samples — mathematically identical to the normal-equations solution for a
full-rank overdetermined system, hence **near-exact** on the fit.

Robust rejection (LazyStretch.js:3073-3097): a zone with no clean floor yields an
anchor sitting ON nebula — biased BRIGHT, landing above the surface its
neighbours define.  Up to 3 rounds: fit the channel-mean surface, compute
residuals ``chanMean - model``, take a MAD sigma (``1.4826 * MAD``, floored at
1e-5), and reject anchors more than ``2.0`` sigma **ABOVE** the surface (never
below).  Keep >= 12 for conditioning; stop when nothing more is rejected or the
survivor count would drop below 12.  The JS medians here are ``sorted[floor(n/2)]``
(upper-middle for even n, *not* the two-value average), reproduced exactly by
:func:`_js_median`.

Per-channel finalize (LazyStretch.js:3099-3131)::

    cf = fit( anchors, med[c] )                    # degree-2, 6 terms
    shift = max( 0, max_anchor( eval(cf) - med ) ) # pass at-or-below every anchor
    cf[0] -= shift                                 # clamp (see note below)
    mins[c], maxs[c] = min/max of eval(cf) over a 25x25 grid on [0,1]^2
    tilt[c] = maxs[c] - mins[c]

Application (LazyStretch.js:3143-3171)::

    out[c] = T[c] - ( model_c(u,v) - mins[c] )      # u = x()/width(), v = y()/height()

with ``rescale=false, truncate=true`` — i.e. clip to [0, 1].

Fidelity notes
--------------
* Fit + subtract: **near-exact** (lstsq vs the JS Gaussian elimination differ only
  in floating-point rounding for well-conditioned systems).
* Anchor sample-selection: **approximate**.  The zone / candidate *geometry*
  (rects, half-cell insets, round-half-up, the 8-px floor) is reproduced exactly;
  the only difference is the per-patch statistic — PixInsight ``Image.median()``
  vs :func:`numpy.median`.  These agree for odd pixel counts and agree to within
  the two central pixels for even counts, so anchor placement is faithful but not
  guaranteed bit-identical on ties.
* The clamp ``cf[0] -= shift`` is a **cosmetic no-op on the image**: it shifts
  ``model`` and ``mins[c]`` down by the same constant, which cancels in
  ``model - mins[c]``.  It is applied anyway so the returned ``coeffs``/``mins``
  match the JS telemetry; it only affects the reported numbers, not the pixels.
* The 25x25 min grid spans the closed ``[0,1]^2`` while the applied coordinates
  span ``[0, (W-1)/W] x [0, (H-1)/H]`` (PixInsight ``x()/width()`` never reaches
  1) — reproduced exactly as in the JS.

Applies to mono and RGB alike (this is a background model, not a colour-only op).
The input is never mutated; a NEW array is returned.
"""
from __future__ import annotations

import math

import numpy as np

# Zone grid (ZX x ZY) and per-zone candidate grid (CS x CS) — LazyStretch.js:2994.
_ZX, _ZY, _CS = 8, 5, 3


def _js_round(x: float) -> int:
    """JavaScript ``Math.round`` — round half toward +infinity (all coords are >= 0)."""
    return int(math.floor(x + 0.5))


def _js_median(arr) -> float:
    """JS median as used in the rejection loop: ``sorted[floor(n/2)]``.

    NOT the average of the two central values for even ``n`` — the JS picks the
    single upper-middle element (LazyStretch.js:3086, 3090). Reproduced exactly.
    """
    s = np.sort(np.asarray(arr, dtype=np.float64))
    return float(s[s.size // 2])


def _design(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Degree-2 design matrix, columns ``[1, u, v, u^2, u*v, v^2]`` (LazyStretch.js:3029)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=-1)


def _fit(u: np.ndarray, v: np.ndarray, y: np.ndarray):
    """Least-squares degree-2 fit over anchor samples; ``None`` if rank-deficient.

    Port of ``fitPts``/``solve6`` (LazyStretch.js:3031-3068) as a design-matrix
    ``lstsq`` — same minimizer as the JS normal-equations Gaussian elimination. A
    rank < 6 (or non-finite) solve mirrors the JS ``solve6`` singular return.
    """
    A = _design(u, v)
    y = np.asarray(y, dtype=np.float64)
    if A.shape[0] < 6:
        return None
    coef, _resid, rank, _sv = np.linalg.lstsq(A, y, rcond=None)
    if rank < 6 or not np.all(np.isfinite(coef)):
        return None
    return coef


def _eval(coef, u, v):
    """Evaluate the degree-2 model (LazyStretch.js:3069-3070). ``u``/``v`` broadcast."""
    return (coef[0] + coef[1] * u + coef[2] * v
            + coef[3] * u * u + coef[4] * u * v + coef[5] * v * v)


def dark_lane_model(img) -> dict:
    """Fit the dark-lane-anchored degree-2 background model (``Pipe.darkLaneModel``).

    Port of LazyStretch.js:2985-3134. Locates dark-lane anchors, fits a degree-2
    surface per channel with robust (bright-only) rejection, clamps the surface
    at-or-below every anchor, and measures the model min/max over a dense grid.

    Parameters
    ----------
    img : numpy.ndarray
        Float64 image in ``[0, 1]``; ``(H, W)`` mono or ``(H, W, 3)`` RGB with
        channel order R, G, B == 0, 1, 2. Never mutated.

    Returns
    -------
    dict
        Mirrors the JS return object. On success::

            {"nc": int, "kept": int, "total": int,
             "coeffs": [ndarray(6), ...],   # per channel, clamped
             "mins": [float, ...], "maxs": [float, ...], "tilt": [float, ...]}

        On failure ``{"error": "..."}`` — too few anchors or a singular fit — in
        which case :func:`dark_lane_gradient` returns the image unchanged.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 2:
        chans = [a]
    elif a.ndim == 3:
        chans = [a[..., c] for c in range(a.shape[2])]
    else:
        raise ValueError(f"expected a 2-D (mono) or 3-D (RGB) image, got shape {a.shape}")

    H, W = int(a.shape[0]), int(a.shape[1])
    nc = min(3, len(chans))  # LazyStretch.js:2991 — nc = min(3, numberOfNominalChannels)

    # --- anchors: darkest candidate sub-patch per zone (8x5 zones, 3x3 candidates)
    an_u: list[float] = []
    an_v: list[float] = []
    an_med: list[list[float]] = []
    for zy in range(_ZY):
        for zx in range(_ZX):
            zx0 = (W * zx) // _ZX
            zx1 = (W * (zx + 1)) // _ZX
            zy0 = (H * zy) // _ZY
            zy1 = (H * (zy + 1)) // _ZY
            best = None  # (lum, med:list, u, v)
            for cy in range(_CS):
                for cx in range(_CS):
                    # candidate rect: centered cell, half the cell size (avoids borders)
                    cw = (zx1 - zx0) / _CS
                    ch = (zy1 - zy0) / _CS
                    x0 = _js_round(zx0 + cx * cw + cw * 0.25)
                    x1 = _js_round(zx0 + cx * cw + cw * 0.75)
                    y0 = _js_round(zy0 + cy * ch + ch * 0.25)
                    y1 = _js_round(zy0 + cy * ch + ch * 0.75)
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        continue
                    med = []
                    lum = 0.0
                    for c in range(nc):
                        # PI Rect(x0,y0,x1,y1) is exclusive on x1/y1 -> slice [y0:y1, x0:x1]
                        m = float(np.median(chans[c][y0:y1, x0:x1]))
                        med.append(m)
                        lum += m
                    lum /= nc
                    if best is None or lum < best[0]:
                        best = (lum, med,
                                ((x0 + x1) / 2) / W, ((y0 + y1) / 2) / H)
            if best is not None:
                an_u.append(best[2])
                an_v.append(best[3])
                an_med.append(best[1])

    total = len(an_u)
    if total < 10:  # LazyStretch.js:3025
        return {"error": f"too few dark-lane anchors: {total}"}

    u = np.asarray(an_u, dtype=np.float64)
    v = np.asarray(an_v, dtype=np.float64)
    med = np.asarray(an_med, dtype=np.float64)          # (N, nc)
    chan_mean = med.mean(axis=1)                        # LazyStretch.js:3071

    # --- robust rejection: drop anchors sitting > 2 sigma ABOVE the channel-mean
    #     surface (on-nebula, biased bright). Keep >= 12; up to 3 rounds.
    for _round in range(3):
        cf = _fit(u, v, chan_mean)
        if cf is None:
            break
        res = chan_mean - _eval(cf, u, v)
        med_r = _js_median(res)
        sig = max(_js_median(np.abs(res - med_r)) * 1.4826, 1e-5)
        keep = res <= med_r + 2.0 * sig
        n_keep = int(keep.sum())
        if n_keep < 12 or n_keep == u.size:
            if n_keep >= 12:
                u, v, med, chan_mean = u[keep], v[keep], med[keep], chan_mean[keep]
            break
        u, v, med, chan_mean = u[keep], v[keep], med[keep], chan_mean[keep]

    # --- per-channel fit + downward clamp (surface passes at-or-below every anchor)
    coeffs = []
    for c in range(nc):
        cf = _fit(u, v, med[:, c])
        if cf is None:
            return {"error": "singular fit"}
        shift = max(0.0, float(np.max(_eval(cf, u, v) - med[:, c])))
        cf = cf.copy()
        cf[0] -= shift
        coeffs.append(cf)

    # --- per-channel model min/max over a dense 25x25 grid on the closed [0,1]^2
    g = np.arange(25, dtype=np.float64) / 24.0
    gu, gv = np.meshgrid(g, g)
    mins, maxs = [], []
    for c in range(nc):
        vals = _eval(coeffs[c], gu, gv)
        mins.append(float(vals.min()))
        maxs.append(float(vals.max()))
    tilt = [maxs[c] - mins[c] for c in range(nc)]

    return {"nc": nc, "kept": int(u.size), "total": total,
            "coeffs": coeffs, "mins": mins, "maxs": maxs, "tilt": tilt}


def dark_lane_gradient(img) -> np.ndarray:
    """Subtract the dark-lane-anchored degree-2 gradient (``Pipe.darkLaneGradient``).

    Port of LazyStretch.js:3138-3177. Fits the model via :func:`dark_lane_model`,
    then per channel subtracts the model's **excess over its minimum**
    (``model(u,v) - mins[c]``) with ``u = col/W, v = row/H`` (PixInsight
    ``x()/width()``, ``y()/height()``), truncating to ``[0, 1]``. The darkest model
    region loses nothing, so the field's overall level is preserved.

    If the model is skipped (too few anchors or a singular fit — the JS returns a
    ``"skipped (...)"`` string and leaves the view untouched), the image is
    returned **unchanged** (a NEW copy, un-truncated, since PI never runs
    PixelMath in that path).

    Parameters
    ----------
    img : numpy.ndarray
        Float64 image in ``[0, 1]``; ``(H, W)`` mono or ``(H, W, 3)`` RGB with
        channel order R, G, B == 0, 1, 2. Never mutated.

    Returns
    -------
    numpy.ndarray
        A NEW corrected array. Truncated to ``[0, 1]`` on the applied path
        (``rescale=false, truncate=true``). Applies to mono and RGB alike.
    """
    a = np.asarray(img, dtype=np.float64)
    out = a.copy()

    m = dark_lane_model(a)
    if "error" in m:
        return out  # skipped: view left unchanged (no truncation, as in the JS)

    H, W = int(a.shape[0]), int(a.shape[1])
    nc = m["nc"]
    u = np.arange(W, dtype=np.float64) / W  # x()/width()  in [0, (W-1)/W]
    v = np.arange(H, dtype=np.float64) / H  # y()/height() in [0, (H-1)/H]
    u2 = u * u
    v2 = v * v

    for c in range(nc):
        cf = m["coeffs"][c]
        # separable evaluation of the degree-2 surface over the full (H, W) grid
        model = (cf[0]
                 + cf[1] * u[None, :] + cf[2] * v[:, None]
                 + cf[3] * u2[None, :]
                 + cf[4] * (v[:, None] * u[None, :])
                 + cf[5] * v2[:, None])
        excess = model - m["mins"][c]
        if a.ndim == 2:
            out = out - excess
        else:
            out[..., c] = out[..., c] - excess

    return np.clip(out, 0.0, 1.0)
