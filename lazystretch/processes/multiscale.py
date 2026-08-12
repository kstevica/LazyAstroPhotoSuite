"""Multiscale pipeline steps — HDR core compression and the MMT noise-reduction
fallback (ports of LazyStretch v1.2.1).

Ported from PixInsight LazyStretch v1.2.1:
  * ``Pipe.hdrCore``          — LazyStretch.js:3104-3117  (HDRMultiscaleTransform,
                                ``medianTransform = false``  →  a-trous / starlet path)
  * ``Pipe.noiseReduction``  — LazyStretch.js:3081-3101   (the NON-AI fallback:
                                MultiscaleMedianTransform; the NoiseXTerminator branch
                                at 3083-3089 is deliberately NOT ported)

IMPORTANT — these two steps use *different* multiscale transforms:

  * :func:`hdr_core` is a **LINEAR** decomposition. ``P.medianTransform = false``
    (js:3111) selects the à-trous (starlet) B3-spline wavelet transform. It is used to
    COMPRESS the dynamic range of large-scale bright structure (galaxy bulges, HII
    cores) while preserving fine detail.

  * :func:`noise_reduction_mmt` is a **MEDIAN** decomposition. The fallback builds a
    ``MultiscaleMedianTransform`` (js:3090, ``P.transform = ...MultiscaleMedianTransform``)
    — a non-linear median pyramid — and applies k-sigma noise reduction on the two
    finest layers only.

Both are marked ``approximate``:  HDRMultiscaleTransform and MultiscaleMedianTransform
are proprietary PixInsight processes operating in an undocumented internal colour
space with undisclosed strength/deringing/shrinkage curves. The transforms below are
faithful to the *documented behaviour and the exact layer configuration in the source*,
but must be calibrated against a real PI install before they are trusted numerically.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# ----------------------------------------------------------------------------
# à-trous (B3-spline) linear wavelet primitives — used by hdr_core.
# ----------------------------------------------------------------------------

# B3-spline scaling kernel [1 4 6 4 1] / 16 (the standard starlet / à-trous kernel).
_B3 = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0


def _dilate_kernel(k: np.ndarray, step: int) -> np.ndarray:
    """Insert ``step-1`` zero 'holes' between kernel taps (the *à trous* — with holes)."""
    if step <= 1:
        return k
    out = np.zeros((len(k) - 1) * step + 1, dtype=np.float64)
    out[::step] = k
    return out


def _atrous_smooth(c: np.ndarray, step: int) -> np.ndarray:
    """One separable B3 smoothing at the given hole spacing (mirror boundary, as PI)."""
    w = _dilate_kernel(_B3, step)
    t = ndi.convolve1d(c, w, axis=0, mode="mirror")
    t = ndi.convolve1d(t, w, axis=1, mode="mirror")
    return t


def _atrous_residual(L: np.ndarray, layers: int) -> np.ndarray:
    """Return the à-trous residual (smooth approximation) after ``layers`` scales.

    Only the residual is needed by :func:`hdr_core` (it treats the residual as the
    large-scale 'base' and everything finer as multiplicative detail), so the detail
    planes are never materialised. The residual after ``n`` scales holds structures
    larger than ~2**n px — exactly what ``numberOfLayers`` controls in PI.
    """
    c = L.astype(np.float64, copy=True)
    for i in range(layers):
        c = _atrous_smooth(c, 1 << i)  # hole spacing 1, 2, 4, ... 2**(layers-1)
    return c


# ----------------------------------------------------------------------------
# Pipe.hdrCore  (LazyStretch.js:3104-3117)  — HDRMultiscaleTransform, a-trous path.
# ----------------------------------------------------------------------------

# HDR calibration constants (tuned against PI's ls-m42-1.4.1 reference core, P1).
# The base (large-scale à-trous residual) is compressed toward its median by
# _HDR_BASE_COMPRESSION; the fine detail (L - base) is added back at full absolute
# strength times _HDR_DETAIL_BOOST. Compressing the base while KEEPING (or boosting)
# absolute detail is what reveals structure in bright cores — the HDRMT signature.
_HDR_BASE_COMPRESSION: float = 0.85   # 0 = flat base, 1 = no compression
# Calibrated against the PI golden set (calibration/survey.py): at 0.65 the base was
# over-compressed, pulling bright cores ~0.08 below PI (systemic clum bias -19%). 0.85
# only rescales the large-scale base — fine detail rides _HDR_DETAIL_BOOST untouched —
# so it restores core brightness (clum bias -19% -> ~-11%) with detailRMS preserved.
# 1.0 = preserve absolute detail (no boost). Base compression alone reveals structure by
# lowering the large-scale DC, so relative detail rises without pushing bright peaks over
# 1.0. The 10-target golden survey showed boost>1 over-crisps the core AND adds to highlight
# clipping; on the M42 core, boost 1.0 gives detailRMS 0.039 (PI target 0.041) with no clip.
_HDR_DETAIL_BOOST: float = 1.00

# Small floor for the hue-preserving ratio at black pixels.
_HDR_EPS: float = 1e-4


def _lightness(img: np.ndarray) -> np.ndarray:
    """Lightness proxy for the RGB path (js:3112 ``toLightness = true``).

    PI's ``toLightness`` uses CIE L* in the configured RGBWorkingSpace, whose *default*
    luminance coefficients are equal (1/3, 1/3, 1/3). We approximate lightness by that
    equal-weight mean and skip the L* non-linearity — sufficient for a hue-preserving
    (ratio-scaled) reconstruction; flagged approximate.
    """
    return img.mean(axis=2)


def hdr_core(img, layers, compression: float = _HDR_BASE_COMPRESSION,
             detail_boost: float = _HDR_DETAIL_BOOST) -> np.ndarray:
    """Compress the dynamic range of large-scale bright structure (a-trous / linear).

    Port of ``Pipe.hdrCore`` (LazyStretch.js:3104-3117). Because ``medianTransform =
    false`` (js:3111) this is the à-trous (B3-spline / starlet) LINEAR wavelet path.
    The large-scale residual (structures larger than ~2**layers px) is the base; the
    remainder ``L - base`` is the fine detail. **Linear** base/detail HDR: the base is
    compressed toward its median (bright cores pulled down, faint large-scale lifted)
    and the detail is added back at FULL ABSOLUTE strength × ``detail_boost``. This is
    the HDRMultiscaleTransform signature — a compressed core REVEALS its fine structure.
    (Calibrated against PI's ``ls-m42-1.4.1`` reference core, P1; see ``calibration/``.)

    An earlier revision reconstructed in log/multiplicative space, which shrank absolute
    detail with the dimmed base and FLATTENED the core — the opposite of HDRMT. The
    ``test_hdr_calibration`` regression guards against that returning.

    Honours the source flags approximately:
      * ``toLightness``  (js:3112): RGB is compressed on a lightness channel.
      * ``preserveHue``  (js:3113): channels are ratio-scaled by new/old lightness, so
        R:G:B ratios (hue and relative saturation) are preserved exactly.
      * ``luminanceMask`` (js:3114): the effect is masked by the original lightness, so
        the faint background is protected and only bright structure is compressed.
      * ``deringing``    (js:3115): NOT modelled (would clamp ringing at bright edges).

    Parameters
    ----------
    img : numpy.ndarray
        float64 in [0, 1]; mono ``(H, W)`` or RGB ``(H, W, 3)`` (R,G,B = 0,1,2).
    layers : int
        Number of wavelet layers (``P.numberOfLayers``). js:3109 defaults to 6 when the
        argument is <= 0; more layers reach larger structures (galaxy bulge ≈ 8).

    Returns
    -------
    numpy.ndarray
        A new array (input never mutated), clipped to [0, 1].

    Fidelity: approximate — calibrate ``_HDR_BASE_COMPRESSION`` and the lightness model
    against PI HDRMultiscaleTransform output.
    """
    a = np.asarray(img, dtype=np.float64)
    n = (int(layers) if (layers is not None and int(layers) > 0) else 6)  # js:3109

    is_rgb = (a.ndim == 3)
    L = _lightness(a) if is_rgb else a.astype(np.float64, copy=True)

    # Large-scale base = à-trous residual after n scales; fine detail = the remainder.
    base = np.clip(_atrous_residual(L, n), 0.0, 1.0)
    detail = L - base

    # Linear base/detail HDR: compress the base toward its median (bright large-scale
    # cores pulled down, faint large-scale lifted) and add the detail back at FULL
    # ABSOLUTE strength (x boost). Unlike a multiplicative/log reconstruction — which
    # shrinks absolute detail with the dimmed base and FLATTENS the core — this keeps
    # fine structure at full contrast, so a compressed core REVEALS its detail (HDRMT).
    med = float(np.median(base))
    base_c = med + (base - med) * compression
    hdr = np.clip(base_c + detail * detail_boost, 0.0, 1.0)

    # luminanceMask (js:3114): protect the dark background — apply the effect through a
    # smooth mask keyed on the LARGE-SCALE brightness (base), so the sky is left put and
    # only bright structure is compressed. Smoothstep 0.10 -> 0.55.
    t = np.clip((base - 0.10) / 0.45, 0.0, 1.0)
    mask = t * t * (3.0 - 2.0 * t)
    newL = L * (1.0 - mask) + hdr * mask
    newL = np.clip(newL, 0.0, 1.0)

    if not is_rgb:
        return newL

    # preserveHue: scale each channel by the lightness ratio (js:3113). Where the
    # original lightness is ~0 there is no colour to preserve → leave the pixel black.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(L > _HDR_EPS, newL / L, 0.0)
    out = a * ratio[..., None]
    return np.clip(out, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Scale-separated (mid-scale) structure enhancement — P1 (roadmap-next).
#
# HDR core (above) compresses the LARGE-scale base; LHE adds SINGLE-scale local contrast.
# Neither selectively lifts the MEDIUM scales — spiral-arm segments, dust-lane topology,
# Milky-Way dust filaments — which is the top visible lever both ChatGPT critiques flagged.
# structure_boost() does a full à-trous starlet decomposition and applies a per-scale gain:
# PROTECT the very-small scales (1–2 px stars/noise, gain 1), ENHANCE the medium band
# (arms/dust), PRESERVE the very-large residual (core/halo continuity, gain 1). The boost
# is masked to the subject by the large-scale base (the dark sky is left alone, so noise is
# not amplified) and reconstructed hue-preservingly (ratio-scaled channels).
# ----------------------------------------------------------------------------


def _atrous_planes(L: np.ndarray, n: int):
    """Full starlet transform: return (``[w_1..w_n]`` detail planes, large-scale residual).

    ``w_i = c_{i-1} - c_i`` is the detail at scale ``i`` (structures ~``2**(i-1)`` px); the
    residual ``c_n`` holds everything larger. ``sum(w_i) + residual == L`` exactly, so the
    reconstruction is lossless and a per-scale gain simply re-weights each band.
    """
    c = L.astype(np.float64, copy=True)
    planes = []
    for i in range(int(n)):
        c_next = _atrous_smooth(c, 1 << i)          # hole spacing 1, 2, 4, ...
        planes.append(c - c_next)
        c = c_next
    return planes, c


def structure_boost(img: np.ndarray, amount: float = 0.6, *, n: int = 6,
                    medium=(3, 4, 5), fine_protect=(1, 2), fine_gain: float = 1.0,
                    bg_lo: float = 0.06, bg_hi: float = 0.22) -> np.ndarray:
    """Enhance MEDIUM-scale structure (arms/dust) while protecting fine and large scales.

    ``amount`` (0..1) scales the medium-band gain (medium detail ×(1 + amount)); ``medium``
    is the set of 1-based à-trous scales treated as "medium" (~4–32 px at ``n=6``); scales in
    ``fine_protect`` keep ``fine_gain`` (1.0 = untouched; <1 would gently damp star/noise
    speckle); the large-scale residual is always preserved. The added detail is gated by a
    smoothstep on the large-scale base (``bg_lo``→``bg_hi``) so only the subject is enhanced,
    never the dark sky. Hue-preserving. Input never mutated; output clipped to [0, 1].
    """
    a = np.asarray(img, dtype=np.float64)
    amount = float(np.clip(amount, 0.0, 2.0))
    if amount <= 0:
        return a.copy()
    is_rgb = (a.ndim == 3)
    L = _lightness(a) if is_rgb else a.astype(np.float64, copy=True)
    planes, resid = _atrous_planes(L, n)

    med = set(int(s) for s in medium)
    fine = set(int(s) for s in fine_protect)
    add = np.zeros_like(L)
    for i, w in enumerate(planes):
        s = i + 1                                    # 1-based scale index
        if s in med:
            add += amount * w                        # boost medium band
        elif s in fine and fine_gain != 1.0:
            add += (fine_gain - 1.0) * w             # optional gentle fine-scale damping

    t = np.clip((resid - bg_lo) / max(bg_hi - bg_lo, 1e-6), 0.0, 1.0)
    mask = t * t * (3.0 - 2.0 * t)                   # subject mask from the large-scale base
    newL = np.clip(L + add * mask, 0.0, 1.0)

    if not is_rgb:
        return newL
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(L > _HDR_EPS, newL / L, 0.0)
    return np.clip(a * ratio[..., None], 0.0, 1.0)


# ----------------------------------------------------------------------------
# Pipe.noiseReduction fallback (LazyStretch.js:3090-3101) — MultiscaleMedianTransform.
# ----------------------------------------------------------------------------

# Exact per-layer config from the source (js:3092-3096). Format per js:3094:
#   [enabled, biasEnabled, bias, nrEnabled, nrThreshold, nrAmount, nrIterations]
# All 6 layers start neutral (bias 0, NR off); NR is switched on for the two finest.
_MMT_LAYERS = [
    # layer 0 (finest):  js:3095
    dict(enabled=True, bias_enabled=True, bias=0.000, nr=True, nr_threshold=3.0, nr_amount=0.60, nr_iters=2),
    # layer 1:           js:3096
    dict(enabled=True, bias_enabled=True, bias=0.000, nr=True, nr_threshold=2.5, nr_amount=0.40, nr_iters=1),
    # layers 2..5:       js:3092-3093 (neutral pass-through)
    dict(enabled=True, bias_enabled=True, bias=0.000, nr=False, nr_threshold=3.0, nr_amount=1.00, nr_iters=0),
    dict(enabled=True, bias_enabled=True, bias=0.000, nr=False, nr_threshold=3.0, nr_amount=1.00, nr_iters=0),
    dict(enabled=True, bias_enabled=True, bias=0.000, nr=False, nr_threshold=3.0, nr_amount=1.00, nr_iters=0),
    dict(enabled=True, bias_enabled=True, bias=0.000, nr=False, nr_threshold=3.0, nr_amount=1.00, nr_iters=0),
]


def _mmt_median(c: np.ndarray, scale: int) -> np.ndarray:
    """One MultiscaleMedian smoothing step: square median of radius ``2**(scale-1)``.

    Layer j (1-indexed) uses a square structuring element of side ``2**j + 1``
    (3x3, 5x5, 9x9, ...), mirror boundary — the standard dyadic multiscale median
    transform (Starck & Murtagh). Approximate: PI's exact structuring element is undocumented.
    """
    size = (1 << scale) + 1  # scale 1 -> 3, scale 2 -> 5, ...
    return ndi.median_filter(c, size=size, mode="mirror")


def _reduce_layer(w: np.ndarray, threshold: float, amount: float, iterations: int) -> np.ndarray:
    """k-sigma soft shrinkage of a median detail layer (approximates PI's layer NR).

    For each iteration: estimate the layer noise sigma via the MAD (1.4826 * MAD),
    then attenuate coefficients smoothly by a factor rising from ``1 - amount`` at zero
    to ``1`` once ``|w| >= threshold * sigma`` — noise-like coefficients are damped,
    genuine structure passes. PI's exact shrinkage curve is proprietary (approximate).
    """
    out = w.astype(np.float64, copy=True)
    for _ in range(int(iterations)):
        med = np.median(out)
        mad = np.median(np.abs(out - med))
        sigma = 1.4826 * mad
        if sigma <= 0.0:
            break
        f = np.clip(np.abs(out) / (threshold * sigma), 0.0, 1.0)
        out = out * ((1.0 - amount) + amount * f)
    return out


def _denoise_channel(ch: np.ndarray) -> np.ndarray:
    """Apply the MMT median-pyramid noise reduction to a single 2-D channel.

    Only the layers with NR enabled are active (js:3095-3096); deeper layers are
    neutral (bias 0, NR off) and therefore mathematical pass-throughs in the additive
    median reconstruction, so their (expensive) large-radius medians are skipped:

        image = residual + sum_j w_j        (w_j = c_{j-1} - c_j, additive transform)

    Modifying only w_1, w_2 gives  new = c_2 + w_1' + w_2'  because the untouched
    deeper detail planes plus the residual telescope back to c_2.
    """
    # Deepest layer that actually changes anything (NR on, or a non-zero bias).
    active = [i for i, ly in enumerate(_MMT_LAYERS)
              if ly["nr"] or (ly["bias_enabled"] and ly["bias"] != 0.0)]
    if not active:
        return ch.copy()
    deepest = max(active)  # 0-indexed; layer index i corresponds to median scale i+1

    # Build median approximations c_0..c_{deepest+1} and detail planes w_1..w_{deepest+1}.
    c = ch.astype(np.float64, copy=True)
    planes = []           # planes[i] == w_{i+1} == c_i - c_{i+1}
    for i in range(deepest + 1):
        c_next = _mmt_median(c, scale=i + 1)
        planes.append(c - c_next)
        c = c_next
    residual = c          # == c_{deepest+1}; carries all deeper (untouched) structure

    # Recombine, transforming only the configured layers.
    out = residual
    for i in range(deepest + 1):
        w = planes[i]
        ly = _MMT_LAYERS[i]
        if ly["nr"]:
            w = _reduce_layer(w, ly["nr_threshold"], ly["nr_amount"], ly["nr_iters"])
        # bias == 0 for every layer here, so no bias term is applied (js:3092-3096).
        out = out + w
    return out


def noise_reduction_mmt(img) -> np.ndarray:
    """Gentle MultiscaleMedianTransform noise reduction (the NON-AI fallback).

    Port of the fallback branch of ``Pipe.noiseReduction`` (LazyStretch.js:3090-3101).
    The NoiseXTerminator branch (js:3083-3089) is intentionally NOT ported — this is
    the plain multiscale-median denoiser used when NoiseX is unavailable. Six median
    layers are configured; NR runs on the two finest only (threshold 3.0 / amount 0.60
    / 2 iters, and threshold 2.5 / amount 0.40 / 1 iter — js:3095-3096). Each channel is
    processed independently, so this is a valid op on both mono and RGB.

    Parameters
    ----------
    img : numpy.ndarray
        float64 in [0, 1]; mono ``(H, W)`` or RGB ``(H, W, 3)``.

    Returns
    -------
    numpy.ndarray
        A new array (input never mutated), clipped to [0, 1].

    Fidelity: approximate — median structuring elements and the k-sigma shrinkage curve
    are faithful in form to the documented MMT but not bit-exact; calibrate against PI.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 2:
        out = _denoise_channel(a)
    else:
        out = np.stack([_denoise_channel(a[..., c]) for c in range(a.shape[2])], axis=2)
    return np.clip(out, 0.0, 1.0)
