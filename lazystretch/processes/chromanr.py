"""Chroma-only noise reduction — port of ``Pipe.chromaNR`` (LazyStretch v1.4.1).

Ported from PixInsight LazyStretch v1.4.1:
  * ``Pipe.chromaNR`` — LazyStretch.js:4037-4055.

A ``MultiscaleMedianTransform`` applied to the CHROMINANCE only
(``P.toChrominance = true``, ``P.toLuminance = false`` — js:4052-4053): colour
speckle is smoothed while the luminance (structure) is left untouched, so detail
CANNOT smear by construction. It runs POST-stretch, after the saturation boost
that amplifies the speckle (js:4034-4036).

The exact per-layer configuration is taken verbatim from the source (js:4044-4049).
All six layers start neutral ``[enabled, biasEnabled, bias, nrEnabled, nrThreshold,
nrAmount, nrIterations] = [true, true, 0.0, false, 3.0, 1.00, 0]`` (js:4044-4045);
NR is then switched on for the THREE finest layers, with an ``amount``-scaled dose
(``a = clip(amount, 0, 1)`` — js:4039):

  * layer 0 (js:4047): threshold 3.0, amount ``clip(0.50 + 0.50*a, 0, 1)``,
                       iterations ``3 if a >= 0.7 else 2``
  * layer 1 (js:4048): threshold 2.5, amount ``clip(0.30 + 0.50*a, 0, 1)``,
                       iterations ``2 if a >= 0.5 else 1``
  * layer 2 (js:4049): threshold 2.0, amount ``0.50 * a``, iterations 1
  * layers 3..5:       neutral pass-through (NR off).

The median-multiscale machinery (dyadic square-median smoothing and the k-sigma
soft-shrinkage layer NR) is REUSED verbatim from :mod:`.multiscale` — the exact
primitives ``noise_reduction_mmt`` uses — rather than re-derived here.

Fidelity: approximate. Two deliberate stand-ins for the proprietary process:
  1. Luminance/chroma split. PI's ``toChrominance`` operates on the CIE a*/b*
     chrominance in the configured RGBWorkingSpace; we split on the equal-weight
     RGBWS default instead — ``L = mean(R, G, B)`` and ``chroma_c = channel - L``
     (a linear opponent split, NOT the CIE Lab non-linearity).
  2. The MMT median structuring elements and the k-sigma shrinkage curve are
     faithful in FORM to the documented MMT but not bit-exact (see
     :mod:`.multiscale`).
Calibrate against a real PI install before trusting numerically.
"""
from __future__ import annotations

import numpy as np

# REUSE the exact median-multiscale primitives that noise_reduction_mmt uses
# (js:4051 P.transform = MultiscaleMedianTransform): the dyadic square-median
# smoothing step and the k-sigma soft-shrinkage layer NR.
from .multiscale import _mmt_median, _reduce_layer


def _denoise_chroma_channel(c: np.ndarray, layers: list[dict]) -> np.ndarray:
    """MMT median-pyramid NR on one (signed) chroma channel.

    Mirrors ``multiscale._denoise_channel`` but drives an arbitrary, amount-derived
    layer configuration. Only the NR-enabled layers are active (js:4047-4049); the
    deeper layers are neutral (bias 0, NR off) and telescope back exactly, so their
    (expensive) large-radius medians are skipped:

        image = residual + sum_j w_j        (w_j = c_{j-1} - c_j, additive transform)

    Modifying only the shallow detail planes leaves ``residual`` (which carries all
    deeper, untouched chroma structure) intact. The channel is SIGNED here
    (chroma = channel - L), which the median filter and MAD-based shrinkage handle
    without change.
    """
    # Deepest layer that actually changes anything (NR on, or a non-zero bias).
    active = [i for i, ly in enumerate(layers)
              if ly["nr"] or (ly["bias_enabled"] and ly["bias"] != 0.0)]
    if not active:
        return c.copy()
    deepest = max(active)  # 0-indexed; layer index i -> median scale i+1

    # Build median approximations c_0..c_{deepest+1} and detail planes w_1..w_{deepest+1}.
    cur = c.astype(np.float64, copy=True)
    planes = []           # planes[i] == w_{i+1} == c_i - c_{i+1}
    for i in range(deepest + 1):
        c_next = _mmt_median(cur, scale=i + 1)
        planes.append(cur - c_next)
        cur = c_next
    residual = cur        # == c_{deepest+1}; carries all deeper (untouched) chroma

    # Recombine, transforming only the configured layers.
    out = residual
    for i in range(deepest + 1):
        w = planes[i]
        ly = layers[i]
        if ly["nr"]:
            w = _reduce_layer(w, ly["nr_threshold"], ly["nr_amount"], ly["nr_iters"])
        # bias == 0 for every layer here, so no bias term is applied (js:4044-4049).
        out = out + w
    return out


def chroma_nr(img, amount) -> np.ndarray:
    """Denoise colour speckle only, leaving luminance untouched.

    Port of ``Pipe.chromaNR`` (LazyStretch.js:4037-4055). A MultiscaleMedianTransform
    is applied to the chrominance only (``toChrominance = true``, ``toLuminance =
    false`` — js:4052-4053). We split RGB into a luminance/chroma pair on the
    equal-weight RGBWS default, ``L = mean(R, G, B)`` and ``chroma_c = channel - L``,
    denoise each chroma channel with the median multiscale transform using the exact
    per-layer, amount-scaled configuration from the source (js:4044-4049), then
    recombine ``channel = L + denoised_chroma``.

    Parameters
    ----------
    img : numpy.ndarray
        float64 in [0, 1]; mono ``(H, W)`` or RGB ``(H, W, 3)`` (R,G,B = 0,1,2).
    amount : float
        Dose in [0, 1]; clamped as ``a = clip(amount, 0, 1)`` (js:4039). Scales the
        NR strength (and iteration count) over the three finest scales.

    Returns
    -------
    numpy.ndarray
        A new array (input never mutated), clipped to [0, 1]. Colour-only op:
        a no-op copy on mono, and a no-op copy when ``amount <= 0`` (js:4040-4041).

    Fidelity: approximate — the RGB-mean opponent chroma split stands in for PI's
    CIE a*/b* chrominance, and the MMT median/shrinkage primitives are faithful in
    form but not bit-exact. See the module docstring.
    """
    a_in = np.asarray(img, dtype=np.float64)

    # a = Math.range( amount, 0, 1 ) (js:4039); off when not ( a > 0 ) (js:4040-4041).
    a = float(np.clip(amount, 0.0, 1.0))

    # Colour-only op: no-op copy on mono, and no-op copy when amount <= 0.
    if a_in.ndim != 3 or not (a > 0.0):
        return a_in.copy()

    # Exact per-layer config from the source (js:4044-4049). Format per js:4046:
    #   [enabled, biasEnabled, bias, nrEnabled, nrThreshold, nrAmount, nrIterations]
    # Six layers start neutral (NR off); NR is switched on for the three finest.
    layers = [
        # layer 0 (finest): js:4047
        dict(bias_enabled=True, bias=0.000, nr=True, nr_threshold=3.0,
             nr_amount=float(np.clip(0.50 + 0.50 * a, 0.0, 1.0)),
             nr_iters=(3 if a >= 0.7 else 2)),
        # layer 1: js:4048
        dict(bias_enabled=True, bias=0.000, nr=True, nr_threshold=2.5,
             nr_amount=float(np.clip(0.30 + 0.50 * a, 0.0, 1.0)),
             nr_iters=(2 if a >= 0.5 else 1)),
        # layer 2: js:4049
        dict(bias_enabled=True, bias=0.000, nr=True, nr_threshold=2.0,
             nr_amount=0.50 * a, nr_iters=1),
        # layers 3..5: neutral pass-through (js:4044-4045)
        dict(bias_enabled=True, bias=0.000, nr=False, nr_threshold=3.0, nr_amount=1.00, nr_iters=0),
        dict(bias_enabled=True, bias=0.000, nr=False, nr_threshold=3.0, nr_amount=1.00, nr_iters=0),
        dict(bias_enabled=True, bias=0.000, nr=False, nr_threshold=3.0, nr_amount=1.00, nr_iters=0),
    ]

    # Luminance/chroma split on the equal-weight RGBWS default (stand-in for CIE a*/b*).
    L = a_in.mean(axis=2)

    out = np.empty_like(a_in)
    for ch in range(a_in.shape[2]):
        chroma = a_in[..., ch] - L
        denoised = _denoise_chroma_channel(chroma, layers)
        out[..., ch] = L + denoised

    return np.clip(out, 0.0, 1.0)
