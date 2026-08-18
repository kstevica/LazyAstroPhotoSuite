"""Tone-shaping finishing steps: contrast, saturation, local contrast.

Ports the three post-stretch "finishing" processes LazyStretch drives, from
PixInsight LazyStretch v1.2.1 (``reference/pixinsight/LazyStretch.js``):

  * :func:`contrast`       — ``Pipe.contrast``       (LazyStretch.js:2468-2481)
  * :func:`saturation`     — ``Pipe.saturation``     (LazyStretch.js:2457-2466)
  * :func:`local_contrast` — ``Pipe.localContrast``  (LazyStretch.js:2444-2455)

Each PI process is a proprietary module; we reimplement the *observable* pixel
behaviour in NumPy/SciPy. :func:`contrast` reproduces PI's CurvesTransformation
control points exactly and interpolates them with an Akima spline (near-exact);
:func:`saturation` and :func:`local_contrast` are documented approximations of
processes whose exact colour space / algorithm PI does not publish.

Contract (see the porting brief):
  * ``img`` is float64, values in ``[0, 1]``, shape ``(H, W)`` mono or
    ``(H, W, 3)`` RGB with channel order R, G, B == index 0, 1, 2.
  * Inputs are never mutated; a new array is always returned.
  * Colour-only ops (:func:`saturation`) are a safe no-op on mono input.
  * Output is truncated to ``[0, 1]`` (PI ``truncate=true`` / process clamping).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.ndimage import gaussian_filter

# Rec. 709 luminance coefficients. PI's ColorSaturation / LocalHistogramEqualization
# work in a CIE-based space using the active RGB working space's luminance
# coefficients (sRGB default ~= these). Rec. 709 is used here as a faithful,
# well-defined stand-in; calibrate against PI's RGBWS if exactness is required.
_LUM = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# --- LocalHistogramEqualization parameters (LazyStretch.js:2447-2450) ----------
_LHE_RADIUS = 64          # P.radius
_LHE_SLOPE_LIMIT = 1.5    # P.slopeLimit (contrast limit; see note in local_contrast)
_LHE_DEFAULT_AMOUNT = 0.12  # P.amount default when caller passes None
_LHE_OVERSHOOT = 3.0      # soft-clip detail beyond this × its 90th-pctile scale (star deringing)


def _luminance(a: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance of an ``(H, W, 3)`` array -> ``(H, W)``."""
    return a[..., 0] * _LUM[0] + a[..., 1] * _LUM[1] + a[..., 2] * _LUM[2]


def contrast(img, pivot, strength) -> np.ndarray:
    """Gentle S-curve contrast pivoted at the background (``Pipe.contrast``).

    Faithful port of LazyStretch.js:2468-2481. Builds the *exact* four
    CurvesTransformation control points PI builds::

        lo = max(0.02, pivot)              # toe just above background
        hi = min(0.92, pivot + 0.45)       # shoulder in the midtones
        K  = [(0, 0),
              (lo, max(0, lo - strength*lo)),        # darken toe
              (hi, min(1, hi + strength*(1 - hi))),  # lift shoulder
              (1, 1)]

    PI uses ``CurvesTransformation.AkimaSubsplines``; we interpolate the same
    points with :class:`scipy.interpolate.Akima1DInterpolator` (Akima's 1970
    method) and apply the resulting curve identically to every channel — PI's
    ``P.K`` is the combined RGB/K curve, so mono and each RGB channel get the
    same map. Output is clamped to ``[0, 1]`` (PI ``executeOn(view, true)`` /
    truncate); the Akima spline can overshoot below 0 between knots, so the clamp
    is load-bearing, not cosmetic.

    Fidelity: **near-exact**. The only divergence from PI is Akima-subsplines vs.
    SciPy's Akima (they differ, if at all, only in boundary-slope handling); the
    control points are identical.
    """
    a = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)

    lo = max(0.02, float(pivot))
    hi = min(0.92, float(pivot) + 0.45)
    y_lo = max(0.0, lo - float(strength) * lo)
    y_hi = min(1.0, hi + float(strength) * (1.0 - hi))
    pts = [(0.0, 0.0), (lo, y_lo), (hi, y_hi), (1.0, 1.0)]

    # Guard the pathological ordering (pivot near/above 0.92 makes lo >= hi):
    # keep only strictly-increasing-x knots so the interpolator is well-posed.
    xs: list[float] = []
    ys: list[float] = []
    for x, y in pts:
        if not xs or x > xs[-1]:
            xs.append(x)
            ys.append(y)

    if len(xs) < 2:
        return a.copy()  # degenerate; identity map

    if len(xs) >= 3:
        curve = Akima1DInterpolator(np.asarray(xs), np.asarray(ys))
        flat = curve(a.ravel())
    else:
        # SciPy's Akima is unreliable with 2 knots; fall back to linear.
        flat = np.interp(a.ravel(), xs, ys)

    out = flat.reshape(a.shape)
    return np.clip(out, 0.0, 1.0)


# --- sRGB <-> CIE Lab (D65) for the ColorSaturation chroma scale --------------------
_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])

# ColorSaturation chroma-scale calibration (P1, against the golden set): PI's flat curve
# at `amount` maps to a c* multiplier. A gentler-than-linear gain matches PI better than
# 1+amount (which over-saturated colourful targets like the Witch Head and under-saturated
# subtle ones like M42 when done in RGB); tune _SAT_GAIN against calibration/survey.py.
_SAT_GAIN: float = 0.85


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, np.power((np.clip(c, 0, None) + 0.055) / 1.055, 2.4))


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _lab_f(t):
    d = 6.0 / 29.0
    return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)


def _lab_finv(t):
    d = 6.0 / 29.0
    return np.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))


def _srgb_to_lab(rgb):
    xyz = np.einsum("hwc,dc->hwd", _srgb_to_linear(rgb), _M_RGB2XYZ) / _WHITE_D65
    fx, fy, fz = _lab_f(xyz[..., 0]), _lab_f(xyz[..., 1]), _lab_f(xyz[..., 2])
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def _lab_to_srgb(L, a_, b_):
    fy = (L + 16.0) / 116.0
    xyz = np.stack([_lab_finv(fy + a_ / 500.0), _lab_finv(fy), _lab_finv(fy - b_ / 200.0)],
                   axis=-1) * _WHITE_D65
    return _linear_to_srgb(np.einsum("hwc,dc->hwd", xyz, _M_XYZ2RGB))


def saturation(img, amount) -> np.ndarray:
    """Saturation boost in CIE L*c*h* (``Pipe.saturation``, LazyStretch.js:2457-2466).

    PI runs ColorSaturation with a flat curve ``HS = [[0, amount], [0.5, amount],
    [1, amount]]`` — the same lift across all hues, ``hueShift = 0`` — scaling **chroma**
    in a CIE L*c*h* space (lightness + hue preserved). This ports that faithfully: convert
    sRGB -> CIE Lab, scale ``(a*, b*)`` (i.e. c*) by ``1 + _SAT_GAIN * amount``, convert
    back — a perceptually uniform boost, unlike the old RGB distance-from-gray scale which
    over-saturated already-colourful pixels (and clipped them) and under-saturated subtle
    ones. No-op on mono. Output clamped to [0, 1].

    Fidelity: near-exact colour space; the ``amount`` -> c* gain (``_SAT_GAIN``) is
    calibrated against PI output (see calibration/README.md).
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] != 3:
        return a.copy()  # mono / non-RGB: colour op is a no-op
    if abs(float(amount)) < 1e-9:
        return a.copy()

    L, aa, bb = _srgb_to_lab(np.clip(a, 0.0, 1.0))
    f = 1.0 + _SAT_GAIN * float(amount)
    out = _lab_to_srgb(L, aa * f, bb * f)
    return np.clip(out, 0.0, 1.0)


def local_contrast(img, amount=_LHE_DEFAULT_AMOUNT) -> np.ndarray:
    """Local contrast enhancement (``Pipe.localContrast``).

    Approximates LazyStretch.js:2444-2455, where PI runs
    LocalHistogramEqualization with ``radius=64``, ``slopeLimit=1.5``,
    ``circularKernel=true`` and ``amount`` (default ``0.12`` when the caller
    passes ``None`` — mirrored here). LHE is a contrast-limited adaptive
    histogram equalization over a circular neighbourhood.

    Implementing CLAHE from scratch (circular kernel, per-tile histograms, slope
    clipping) is heavy and skimage is not a dependency, so we use the
    large-scale unsharp-mask approximation explicitly sanctioned by the brief:

        local_mean = Gaussian blur of the lightness at sigma = radius / 2
        detail     = lightness - local_mean
        out        = lightness + amount * detail

    ``sigma = radius / 2`` matches the variance of a uniform disk of the given
    radius (var = R^2/4). For RGB we enhance the Rec. 709 luminance and add the
    luminance delta back to each channel (colour-preserving, like LHE run on
    lightness); for mono we enhance the single channel directly. Output is
    clamped to ``[0, 1]``.

    Fidelity: **approximate**. This is a high-pass/local-contrast stand-in, not a
    true local histogram equalization: (1) ``slopeLimit`` (LHE's contrast cap)
    is NOT modelled beyond the final ``[0, 1]`` clamp; (2) ``amount`` here scales
    a high-pass layer, whereas PI's ``amount`` is the opacity of an equalized
    layer, so equal numeric amounts differ in strength; (3) a circular kernel is
    approximated by a separable Gaussian. Expect mild ringing around bright stars
    (an unsharp artifact). Calibrate the gain/sigma against PI if needed.
    """
    if amount is None:
        amount = _LHE_DEFAULT_AMOUNT
    amount = float(amount)
    sigma = _LHE_RADIUS / 2.0

    a = np.asarray(img, dtype=np.float64)

    def _enhance(ch: np.ndarray) -> np.ndarray:
        local_mean = gaussian_filter(ch, sigma=sigma, mode="reflect")
        detail = ch - local_mean
        # Overshoot limit (models LHE's slopeLimit, unmodelled before): our unsharp stand-in
        # rings around bright stars — large NEGATIVE detail in the ring becomes a dark halo /
        # crescent. Soft-clip the detail at a multiple of its structure-scale magnitude (90th
        # percentile of |detail|, so the sky-noise floor doesn't set the scale): nebula-scale
        # contrast passes ~linearly, extreme star overshoot is compressed. Symmetric — it tames
        # both the bright ring and the dark halo, and never over-darkens a pixel.
        scale = float(np.percentile(np.abs(detail), 90)) + 1e-6
        lim = _LHE_OVERSHOOT * scale
        detail = lim * np.tanh(detail / lim)
        return ch + amount * detail

    if a.ndim == 2:
        out = _enhance(a)
        return np.clip(out, 0.0, 1.0)

    if a.ndim != 3 or a.shape[2] != 3:
        # Fall back to per-channel enhancement for non-standard channel counts.
        out = np.stack([_enhance(a[..., c]) for c in range(a.shape[2])], axis=-1)
        return np.clip(out, 0.0, 1.0)

    lum = _luminance(a)
    delta = _enhance(lum) - lum
    out = a + delta[..., None]
    return np.clip(out, 0.0, 1.0)
