"""PixelMath-based finishing steps — de-haze, cast reduction, emission boost, gradient cleanup.

Ported from PixInsight LazyStretch v1.2.1:
  * ``Pipe.dehaze``          — LazyStretch.js:2907-3021
  * ``Pipe.reduceCast``      — LazyStretch.js:2860-2905
  * ``Pipe.emissionBoost``   — LazyStretch.js:2828-2858
  * ``Pipe.gradientCleanup`` — LazyStretch.js:3023-3071

All four steps run late in the pipeline on the *stretched* (non-linear) image. Three
of them are pure PixelMath expressions (dehaze, reduceCast, and the blend inside
gradientCleanup) which port verbatim into NumPy; the pieces that lean on proprietary
PixInsight processes are approximated and flagged inline:

  * **Range masks** (``LS.buildRangeMask`` -> ``RangeSelection``, LazyStretch.js:2508-2532).
    ``RangeSelection`` selects by CIE L\\* lightness with a *fuzziness* soft ramp and a
    *smoothness* convolution. Neither the exact L\\* transfer nor the exact smoothing
    kernel is documented, so :func:`_range_mask` uses the equal-weight lightness
    ``(R+G+B)/3`` (the same ``L`` the surrounding PixelMath uses) with a linear edge ramp
    and a Gaussian smooth. APPROXIMATE — calibrate against PI.
  * **Emission mask** (``LS.buildEmissionMask``, LazyStretch.js:2804-2823) is a raw
    PixelMath red-excess expression and ports EXACTLY.
  * **ColorSaturation / CurvesTransformation** inside emissionBoost use PI's proprietary
    colour engine; approximated (see :func:`emission_boost`).
  * **AutomaticBackgroundExtractor** (degree-1) inside gradientCleanup is approximated by
    a robust least-squares plane fit (see :func:`gradient_cleanup`).

Contract: float64 arrays in [0, 1]; channel order R,G,B == index 0,1,2; inputs are never
mutated; colour-only ops (reduceCast, emissionBoost, and dehaze's pass-2 desaturation)
are a no-op on mono. PixelMath steps set ``truncate = true``, so results are clipped to
[0, 1] exactly where PI clips.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.interpolate import Akima1DInterpolator

from ..stats.measure import region_median_channel_avg


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _nchan(a: np.ndarray) -> int:
    return 1 if a.ndim == 2 else a.shape[2]


def _is_rgb(a: np.ndarray) -> bool:
    return a.ndim == 3 and a.shape[2] >= 3


def _lightness(a: np.ndarray) -> np.ndarray:
    """Equal-weight lightness ``(R+G+B)/3`` used as the ``RangeSelection`` L\\* proxy.

    PI's ``RangeSelection`` with ``toLightness = true`` keys on CIE L\\*; that transfer
    is not reproduced here. The equal-weight mean matches the ``L`` the surrounding
    PixelMath expressions use, so mask and expression stay consistent. APPROXIMATE.
    """
    if a.ndim == 2:
        return a
    return a.mean(axis=2)


def _range_mask(a: np.ndarray, low: float, high: float,
                fuzz: float = 0.15, smooth: float = 12.0) -> np.ndarray:
    """Approximate ``LS.buildRangeMask`` / PI ``RangeSelection`` (LazyStretch.js:2508-2532).

    White (1) where the lightness is inside ``[low, high]``, black (0) outside, with a
    linear ``fuzz``-wide soft ramp at each edge and a Gaussian ``smooth`` blur. Defaults
    (``fuzz = 0.15``, ``smooth = 12``) match the source's ``buildRangeMask`` defaults.

    APPROXIMATE: PI's exact fuzziness curve and smoothness kernel are undocumented; a
    linear ramp + Gaussian (sigma == ``smooth`` px) is used and should be calibrated.
    """
    v = _lightness(a)
    m = np.where((v >= low) & (v <= high), 1.0, 0.0)
    if fuzz > 0.0:
        lo = (v < low) & (v > low - fuzz)
        m = np.where(lo, (v - (low - fuzz)) / fuzz, m)
        hi = (v > high) & (v < high + fuzz)
        m = np.where(hi, ((high + fuzz) - v) / fuzz, m)
    if smooth and smooth > 0.0:
        m = ndimage.gaussian_filter(m, sigma=float(smooth))
    return np.clip(m, 0.0, 1.0)


def _apply_masked(orig: np.ndarray, processed: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    """Blend ``processed`` into ``orig`` through a grayscale mask (``maskInverted = false``).

    PI applies a masked process as ``out = orig*(1 - mask) + processed*mask`` per channel,
    with the single grayscale mask broadcast over all channels.
    """
    if orig.ndim == 3:
        w = mask2d[..., np.newaxis]
    else:
        w = mask2d
    return orig * (1.0 - w) + processed * w


def _jsround(x: float) -> int:
    """JavaScript ``Math.round`` — floor(x + 0.5) (round half toward +inf)."""
    return int(math.floor(x + 0.5))


def _clean_sky(a: np.ndarray) -> float:
    """Darkest-patch sky level == ``LS.measureDust(...).cleanSky`` (LazyStretch.js:2162-2193).

    A 6x4 grid of 7%-wide patches whose centres span 8%..92%; each patch value is the
    channel-averaged median; ``cleanSky`` is the 10th-percentile patch (the darkest,
    dust-free sky). Returns 0.0 when fewer than 4 valid patches fit (tiny image), which
    the caller treats as a sampling failure. Grid literals are inlined verbatim from the
    source (they are hard-coded constants there, not data-driven).
    """
    H, W = a.shape[0], a.shape[1]
    n = _nchan(a)
    gx, gy = 6, 4
    pw, ph = 0.07, 0.07
    vals = []
    for iy in range(gy):
        for ix in range(gx):
            cx = 0.08 + 0.84 * (ix / (gx - 1))
            cy = 0.08 + 0.84 * (iy / (gy - 1))
            x0 = max(0, _jsround((cx - pw / 2) * W))
            y0 = max(0, _jsround((cy - ph / 2) * H))
            x1 = min(W, _jsround((cx + pw / 2) * W))
            y1 = min(H, _jsround((cy + ph / 2) * H))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            vals.append(region_median_channel_avg(a, x0, y0, x1, y1))
    if len(vals) < 4:
        return 0.0
    vals.sort()
    nn = len(vals)
    idx = min(nn - 1, max(0, int(math.floor(0.10 * (nn - 1)))))
    return float(vals[idx])


# ---------------------------------------------------------------------------
# Pipe.dehaze — LazyStretch.js:2907-3021
# ---------------------------------------------------------------------------

def dehaze(img, strength, floor, cls, gradient: float = 0.0) -> np.ndarray:
    """Two-pass soft veil subtraction + class-aware brownness desaturation.

    Port of ``Pipe.dehaze`` (v1.4.1 signature ``Pipe.dehaze(view, strength, floor, cls,
    gradient)``, LazyStretch.js:3796). ``gradient`` is the auto-assess opposition score;
    in v1.4.1 it gates a stronger LP-tint removal only on low-gradient frames
    (``strength >= 0.5 and gradient < 0.45``). That gradient-gated branch is a v1.4.1
    refinement not yet applied here (this step is a documented approximation); the arg is
    threaded so the pipeline passes the real value and the refinement can slot in later.

    Pass 1 (LazyStretch.js:2941-2956) — a global soft-shadows veil subtraction, applied
    to *every* channel identically (so it runs on mono too, it is a luminance op):

        v  = strength * max(0, sky - floor)          # sky = darkest-patch level
        ke = max(0.002, 0.15 * v)                    # knee proportional to the veil
        out = ((T - v) + sqrt((T - v)^2 + ke^2)) / (2 * (1 - v))

    Pass 2 (LazyStretch.js:2972-3012, RGB only) — grey the residual warm tint through a
    range mask (0..0.70, sparing bright cores). Brownness key ``max(0, min(R-G, G-B))``
    (positive only on the monotonic warm ramp R>G>B); on ``cls == "emission"`` it also
    adds cool excess ``max(0, B-R)``:

        L = (R+G+B)/3
        r = key/(L + 0.03) * 6
        a = (strength*0.7) * r / sqrt(1 + r*r)       # soft-clamped desaturation weight
        out_c = T_c*(1 - a) + L*a

    Parameters
    ----------
    img : numpy.ndarray
        Float64 image in [0, 1], mono (H, W) or RGB (H, W, 3).
    strength : float
        De-haze amount in [0, 1]; ``<= 0`` returns an unchanged copy ("off").
    floor : float or None
        Target sky floor, clamped to [0, 0.5]; ``None`` -> 0.05 (LazyStretch.js:2932).
    cls : str
        Object-class string; ``"emission"`` widens the pass-2 desaturation key to cool
        excess.

    Returns
    -------
    numpy.ndarray
        A new array, clipped to [0, 1] where PI truncates. Input never mutated.

    Fidelity: near-exact. Pass-1 and the pass-2 PixelMath are exact; the pass-2 range
    mask is the approximate ``RangeSelection`` proxy (see :func:`_range_mask`).
    """
    a = np.asarray(img, dtype=np.float64)

    # "off" — nothing above zero strength (LazyStretch.js:2909-2910).
    if not (strength is not None and strength > 0):
        return a.copy()

    s = float(np.clip(strength, 0.0, 1.0))
    n = _nchan(a)

    # Sky estimate: darkest-patch level, else fall back to the global channel-averaged
    # median (LazyStretch.js:2920-2931).
    sky = _clean_sky(a)
    if not (sky > 0.0):
        nc = min(3, n)
        if a.ndim == 2:
            sky = float(np.median(a))
        else:
            sky = float(np.mean([np.median(a[..., c]) for c in range(nc)]))

    f = float(np.clip(floor, 0.0, 0.5)) if floor is not None else 0.05
    v = s * max(0.0, sky - f)

    out = a.copy()

    # Pass 1 — soft-knee veil subtraction (LazyStretch.js:2941-2956).
    if v > 0.001:
        ke = max(0.002, 0.15 * v)
        t = out - v
        out = (t + np.sqrt(t * t + ke * ke)) / (2.0 * (1.0 - v))
        np.clip(out, 0.0, 1.0, out=out)

    # Pass 2 — desaturate the residual tint (RGB only; LazyStretch.js:2972-3012).
    if n >= 3:
        mask = _range_mask(out, 0.0, 0.70)   # spare bright cores
        R = out[..., 0]
        G = out[..., 1]
        B = out[..., 2]
        L = (R + G + B) / 3.0
        key = np.maximum(0.0, np.minimum(R - G, G - B))
        if cls == "emission":
            key = key + np.maximum(0.0, B - R)
        r = key / (L + 0.03) * 6.0
        # Tint-desat dose (LazyStretch.js:5555-5566). Milky Way class forces it to ZERO —
        # the gold/blue galactic ramp IS the subject, not a cast to remove.
        dose = 0.0 if cls == "milkyway" else (s * 0.7)
        aw = dose * r / np.sqrt(1.0 + r * r)         # soft clamp r/sqrt(1+r^2)
        processed = np.empty_like(out)
        processed[..., 0] = R * (1.0 - aw) + L * aw
        processed[..., 1] = G * (1.0 - aw) + L * aw
        processed[..., 2] = B * (1.0 - aw) + L * aw
        # copy through any extra (alpha) channels unchanged
        for c in range(3, n):
            processed[..., c] = out[..., c]
        out = _apply_masked(out, processed, mask)
        np.clip(out, 0.0, 1.0, out=out)

    return out


# ---------------------------------------------------------------------------
# Pipe.reduceCast — LazyStretch.js:2860-2905
# ---------------------------------------------------------------------------

def reduce_cast(img) -> np.ndarray:
    """Pull the faint sky background toward neutral through a background mask.

    Port of ``Pipe.reduceCast`` (LazyStretch.js:2860-2905). Measures the channel-averaged
    median ``med``, builds a range mask white on the sky background (lightness 0..med),
    and gently pulls the masked pixels 30% toward grey:

        L      = (R+G+B)/3
        new_c  = T_c*0.70 + L*0.30
        out_c  = orig_c*(1 - mask) + new_c*mask

    Colour-only op: a **no-op on mono** (the pull references all three channels).

    Returns
    -------
    numpy.ndarray
        A new array clipped to [0, 1]. Mono input returned as an unchanged copy.

    Fidelity: near-exact. The pull PixelMath is exact; the background mask is the
    approximate ``RangeSelection`` proxy (see :func:`_range_mask`).
    """
    a = np.asarray(img, dtype=np.float64)

    # Colour-only: no-op on mono / single channel.
    if not _is_rgb(a):
        return a.copy()

    n = _nchan(a)
    # med = channel-averaged median over all channels (LazyStretch.js:2864-2866).
    med = float(np.mean([np.median(a[..., c]) for c in range(n)]))

    mask = _range_mask(a, 0.0, med)   # white on the sky background

    R = a[..., 0]
    G = a[..., 1]
    B = a[..., 2]
    L = (R + G + B) / 3.0
    processed = a.copy()
    processed[..., 0] = R * 0.70 + L * 0.30
    processed[..., 1] = G * 0.70 + L * 0.30
    processed[..., 2] = B * 0.70 + L * 0.30

    out = _apply_masked(a, processed, mask)
    np.clip(out, 0.0, 1.0, out=out)
    return out


# ---------------------------------------------------------------------------
# Pipe.emissionBoost — LazyStretch.js:2828-2858
# ---------------------------------------------------------------------------

# Gentle S-curve control points (LazyStretch.js:2844), Akima subsplines.
_EMISSION_S_X = np.array([0.0, 0.35, 0.65, 1.0])
_EMISSION_S_Y = np.array([0.0, 0.31, 0.69, 1.0])
# Flat ColorSaturation curve value across all hues (LazyStretch.js:2839).
_EMISSION_SAT = 0.35


def emission_boost(img) -> np.ndarray:
    """Boost saturation + local contrast in red-excess (Ha) regions.

    Port of ``Pipe.emissionBoost`` (LazyStretch.js:2828-2858). Builds an emission mask
    (white on RED-dominant pixels), then runs a saturation boost followed by a gentle
    contrast S-curve THROUGH that mask, so emission nebulae separate from surrounding
    brown dust without globally over-saturating.

    Emission mask (``LS.buildEmissionMask``, LazyStretch.js:2804-2823) — ported EXACTLY::

        mask = min(1, max(0, R - max(G, B)) * 5)

    Saturation (``ColorSaturation`` HS = flat 0.35, LazyStretch.js:2838-2841) — APPROXIMATE.
    PI's ColorSaturation curve maps a hue to a proprietary saturation multiplier; here a
    flat curve value ``c`` is applied as a chroma scale ``(1 + c)`` about the per-pixel
    equal-weight mean (preserving that luminance proxy)::

        out_c = L + (T_c - L)*(1 + 0.35)

    Contrast S (``CurvesTransformation`` K, Akima subsplines, LazyStretch.js:2843-2846) —
    near-exact: the same 4 control points evaluated with an Akima spline (PI's
    ``AkimaSubsplines``) applied per channel.

    Both sub-processes are blended through the emission mask, in order (saturation, then
    curve on the saturated result), matching PI's two successive masked ``executeOn`` calls.

    Colour-only op: a **no-op on mono**.

    Returns
    -------
    numpy.ndarray
        A new array clipped to [0, 1]. Mono input returned as an unchanged copy.

    Fidelity: approximate (the ColorSaturation multiplier mapping is the main unknown;
    the Akima S-curve and the emission mask are faithful).
    """
    a = np.asarray(img, dtype=np.float64)

    # Colour-only: no-op on mono / single channel.
    if not _is_rgb(a):
        return a.copy()

    R = a[..., 0]
    G = a[..., 1]
    B = a[..., 2]

    # Emission mask — exact port of buildEmissionMask (LazyStretch.js:2811).
    mask = np.clip(np.maximum(0.0, R - np.maximum(G, B)) * 5.0, 0.0, 1.0)

    # --- Sub-step 1: saturation boost through the mask (APPROXIMATE) ---
    sat_factor = 1.0 + _EMISSION_SAT
    L1 = (R + G + B) / 3.0
    saturated = a.copy()
    saturated[..., 0] = L1 + (R - L1) * sat_factor
    saturated[..., 1] = L1 + (G - L1) * sat_factor
    saturated[..., 2] = L1 + (B - L1) * sat_factor
    np.clip(saturated, 0.0, 1.0, out=saturated)
    step1 = _apply_masked(a, saturated, mask)
    np.clip(step1, 0.0, 1.0, out=step1)

    # --- Sub-step 2: gentle contrast S-curve through the mask (near-exact) ---
    spline = Akima1DInterpolator(_EMISSION_S_X, _EMISSION_S_Y)
    curved = step1.copy()
    for c in range(min(3, _nchan(step1))):
        ch = np.clip(step1[..., c], 0.0, 1.0)
        curved[..., c] = spline(ch)
    np.clip(curved, 0.0, 1.0, out=curved)
    step2 = _apply_masked(step1, curved, mask)
    np.clip(step2, 0.0, 1.0, out=step2)

    return step2


# ---------------------------------------------------------------------------
# Pipe.gradientCleanup — LazyStretch.js:3023-3071
# ---------------------------------------------------------------------------

def _robust_plane_fit(chan: np.ndarray) -> np.ndarray:
    """Robust degree-1 (plane) background model of a single channel.

    Approximates ``AutomaticBackgroundExtractor`` with ``polyDegree = 1``: a smooth plane
    that cannot overfit or remove dust. Fit by iteratively-reweighted least squares on a
    subsample, rejecting bright (object/star) outliers more aggressively than dark ones —
    a loose analogue of ABE's asymmetric ``unbalance``/``deviation`` sample rejection.
    """
    H, W = chan.shape
    ys, xs = np.mgrid[0:H, 0:W]
    xn = xs / max(W - 1, 1)
    yn = ys / max(H - 1, 1)

    # Subsample to ~40k points for the fit (the plane is smooth; full resolution is
    # unnecessary and slow on large frames).
    step = max(1, int(math.sqrt(H * W / 40000.0)))
    xf = xn[::step, ::step].ravel()
    yf = yn[::step, ::step].ravel()
    zf = chan[::step, ::step].ravel()

    A = np.column_stack([np.ones_like(xf), xf, yf])
    keep = np.ones(zf.shape, dtype=bool)
    coef = np.array([float(np.median(zf)), 0.0, 0.0])
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(A[keep], zf[keep], rcond=None)
        resid = zf - A @ coef
        med = float(np.median(resid[keep]))
        sigma = 1.4826 * float(np.median(np.abs(resid[keep] - med))) + 1e-12
        # Asymmetric clip: objects sit ABOVE the background, so reject the bright side
        # tightly (+1.5 sigma) and keep the dark side wide (-3 sigma).
        new_keep = (resid < med + 1.5 * sigma) & (resid > med - 3.0 * sigma)
        if int(new_keep.sum()) < 8 or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    return coef[0] + coef[1] * xn + coef[2] * yn


def gradient_cleanup(img, strength) -> np.ndarray:
    """Robust degree-1 plane-fit gradient removal, blended by strength.

    Port of ``Pipe.gradientCleanup`` (LazyStretch.js:3023-3071). Models a LOW-ORDER
    (linear, degree-1) background per channel, subtracts it while preserving the overall
    level (ABE ``normalize = true`` -> add back the model mean), then blends the corrected
    result into the target by ``strength`` (LazyStretch.js:3046-3052)::

        corrected_c = T_c - plane_c + mean(plane_c)
        out_c       = T_c*(1 - s) + corrected_c*s

    Applies to both mono and RGB (ABE runs per channel).

    Parameters
    ----------
    img : numpy.ndarray
        Float64 image in [0, 1], mono (H, W) or RGB (H, W, 3).
    strength : float
        Blend amount in [0, 1]; ``<= 0`` returns an unchanged copy ("off").

    Returns
    -------
    numpy.ndarray
        A new array clipped to [0, 1] (PixelMath truncate). Input never mutated.

    Fidelity: near-exact. The subtract-and-blend PixelMath is exact; the degree-1 model
    is a robust least-squares plane approximating ABE's exact sample rejection.
    """
    a = np.asarray(img, dtype=np.float64)

    # "off" (LazyStretch.js:3025-3026).
    if not (strength is not None and strength > 0):
        return a.copy()

    s = float(np.clip(strength, 0.0, 1.0))

    if a.ndim == 2:
        plane = _robust_plane_fit(a)
        corrected = a - plane + float(plane.mean())
        out = a * (1.0 - s) + corrected * s
    else:
        out = a.copy()
        for c in range(a.shape[2]):
            plane = _robust_plane_fit(a[..., c])
            corrected = a[..., c] - plane + float(plane.mean())
            out[..., c] = a[..., c] * (1.0 - s) + corrected * s

    np.clip(out, 0.0, 1.0, out=out)
    return out
