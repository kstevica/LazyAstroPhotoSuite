"""Advisor / Analyze-Frame engine — the dry-run measurement + recommendation report.

Ported from PixInsight LazyStretch v1.4.1:
  * :func:`measure_noise` — ``LS.measureNoise`` (LazyStretch.js:2554-2582)
  * :func:`analyze_view`  — ``LS.analyzeView``  (LazyStretch.js:2633-2815)

``analyzeView`` is a *dry run*: nothing is processed. It measures the (linear)
master and translates the numbers into the same evidence-only rules the runtime
advisor uses — a report of plain-language ``lines`` plus a list of one-click
``recommendations`` (``{key, val, label}``, the JS ``changes``). A change is
recommended ONLY where a field dataset produced the tell; everything else is a
measured fact. All thresholds, keys and values below are transcribed verbatim
from the source so the recommendation logic is bit-for-bit the same.

Deviations from the JS (documented, deliberate)
-----------------------------------------------
* **Noise estimate (``measureNoise``)** — PixInsight's ``Image.noiseMRS`` (the
  official multiresolution-support NoiseEvaluation) and ``Image.noiseKSigma``
  fallback (LazyStretch.js:2566-2577) are PI-native and have no NumPy equivalent.
  This port substitutes a **robust MAD-based high-frequency sigma**: the first
  a-trous B3-spline wavelet detail layer, with ``sigma = 1.4826 * MAD(detail)``
  de-scaled by the layer's noise-propagation factor ``||delta - h*h||_2``. MAD
  rejects stars / edges the same way the multiresolution support excludes
  significant structure, so the estimate tracks the sky-noise floor — but it is
  an approximation, not the exact MRS number. The report keeps the ``"MRS sigma"``
  wording for parity with the JS console output; the ``usedKSigma`` /
  ``"(k-sigma fallback)"`` suffix is dropped (there is a single estimator, no
  fallback ladder).
* **``vis`` mapper** — ``autoAssess`` returns a scalar linear→visual (post-stretch)
  mapper (``visScalar``, LazyStretch.js:3510-3522) used by the dust-lift line. The
  ported :func:`~lazystretch.stats.assess.auto_assess` does not expose it, so it is
  rebuilt here from the same per-channel STF params (``-2.8`` sigma, ``0.25`` target;
  LazyStretch.js:3442-3453), reproducing ``visScalar`` exactly.
* **Star census** (``LS.starCensus``, LazyStretch.js:2775-2789) — needs a star
  detector; SKIPPED per spec. A single ``"Stars: census unavailable"`` line is
  emitted in its place.
* **Colour-calibration tail** (LazyStretch.js:2791-2801) and the **per-object
  memory / curated tail** (2803-2812) reference ``parameters.solution`` /
  ``rememberPerObject`` / ``hasObjectMemory`` — fields not present in the ported
  ``Parameters`` model — so they are omitted. The measurement + recommendation
  core (gradient, dust, noise, chromatic tilt) is reproduced in full.

Field-name note: the JS ``parameters.objectClass`` is ``params.object_class`` in
the ported model; the five dial / toggle fields consulted here
(``cropPercent``, ``autoAssess``, ``inputStretched``, ``useMask``, ``dehaze``,
``chromaNR``, ``gradientCleanup``, ``reduceCast``, ``darkLaneGC``) keep their JS
(camelCase) names. Inputs are never mutated.

Contract: float64 arrays in ``[0, 1]``, ``(H, W)`` mono or ``(H, W, 3)`` RGB with
channel order R, G, B == 0, 1, 2. This module only *reads* the image and returns
measurements; it produces no image output, so there is nothing to clip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..stats.assess import auto_assess, measure_dust
from ..stats.mtf import find_midtones_balance, mtf_scalar
from ..processes.darklane import dark_lane_model


# --- class label (LazyStretch.js:1066-1078 — pure switch, no data-layer entry) ---
_CLASS_LABELS = {
    "galaxy": "Galaxy",
    "emission": "Emission nebula",
    "reflection": "Reflection nebula",
    "planetary": "Planetary nebula",
    "globular": "Globular cluster",
    "open": "Open cluster",
}


def _class_label(cls: str) -> str:
    """``LS.classLabel`` — human label for a class, ``"Generic / unknown"`` default."""
    return _CLASS_LABELS.get(cls, "Generic / unknown")


def _adaptive_floor_applies_to(cls: str) -> bool:
    """``LS.adaptiveFloorAppliesTo`` (LazyStretch.js:2827-2830) — emission / generic."""
    return cls == "emission" or cls == "generic"


@dataclass(frozen=True)
class AnalyzeResult:
    """Result of :func:`analyze_view` — the JS ``{ lines, changes }`` object.

    Attributes
    ----------
    lines : list[str]
        Plain-language analysis report, one console line per entry.
    recommendations : list[dict]
        One-click dial changes, each ``{"key": paramName, "val": value,
        "label": reason}`` (the JS ``changes``). ``key`` matches a
        :class:`~lazystretch.objects.model.Parameters` field name; ``val`` is the
        recommended value. Empty when class defaults + runtime gates already fit.
    """

    lines: list[str] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #

# B3-spline scaling kernel, first a-trous scale (LazyStretch's MRS uses the
# multiscale linear transform; this is its finest scaling filter).
_B3 = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
# Noise-propagation factor for the first detail layer  w1 = x - smooth(x)
# = conv(x, delta - h⊗h): for unit-variance white noise, std(w1) = ||k||_2.
# So sigma_x = std(w1) / ||k||_2, and robust std(w1) = 1.4826 * MAD(w1).
_H2 = np.outer(_B3, _B3)
_DELTA = np.zeros_like(_H2)
_DELTA[_H2.shape[0] // 2, _H2.shape[1] // 2] = 1.0
_NOISE_FACTOR = float(np.sqrt(np.sum((_DELTA - _H2) ** 2)))  # ~= 0.8908


def measure_noise(img) -> list[float]:
    """Per-channel noise sigma estimate (``LS.measureNoise``, LazyStretch.js:2554-2582).

    **Approximation of PI's MRS/k-sigma** (see module docstring): a robust
    MAD-based high-frequency sigma. Per channel: form the first a-trous B3-spline
    wavelet detail ``w1 = channel - smooth(channel)`` (separable convolution,
    reflect boundary), take ``1.4826 * MAD(w1)`` for a robust standard deviation,
    and de-scale by the detail layer's noise-propagation factor to recover the
    input noise sigma. MAD is robust to stars / edges, mirroring the way the
    multiresolution support excludes significant structure.

    Parameters
    ----------
    img : numpy.ndarray
        Float64 image in ``[0, 1]``; ``(H, W)`` mono or ``(H, W, 3)`` RGB. Never
        mutated.

    Returns
    -------
    list[float]
        One sigma per channel (length 1 for mono, 3 for RGB), in image units —
        the same units the Analyze report ratios against the darkest-patch sky.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 2:
        chans = [a]
    elif a.ndim == 3:
        chans = [a[..., c] for c in range(a.shape[2])]
    else:
        raise ValueError(f"expected a 2-D (mono) or 3-D (RGB) image, got shape {a.shape}")

    sigma: list[float] = []
    for ch in chans:
        smooth = ndimage.convolve1d(ch, _B3, axis=0, mode="reflect")
        smooth = ndimage.convolve1d(smooth, _B3, axis=1, mode="reflect")
        detail = ch - smooth
        mad = float(np.median(np.abs(detail - np.median(detail))))
        robust_std = 1.4826 * mad
        sigma.append(robust_std / _NOISE_FACTOR if _NOISE_FACTOR > 0 else robust_std)
    return sigma


# --------------------------------------------------------------------------- #
# vis mapper — rebuilds autoAssess's scalar linear->visual mapper (js:3510-3522)
# --------------------------------------------------------------------------- #

def _build_vis(a: np.ndarray, data=None):
    """Rebuild ``autoAssess``'s ``visScalar`` (LazyStretch.js:3442-3522).

    Per channel: ``c0 = clip(median - 2.8*avgDev, 0, 1)`` and
    ``mm = findMidtonesBalance(0.25, median - c0)`` (avgDev = mean |x - median|).
    ``vis(x) = mean_c MTF(mm[c], clip((x - c0[c]) / (1 - c0[c]), 0, 1))``. The
    ``-2.8`` sigma / ``0.25`` target come from the data layer when supplied, else
    the JS literals. Identical STF params to the ported ``auto_assess``.
    """
    nch = 1 if a.ndim == 2 else a.shape[2]
    sigma_clip = -2.8
    target_bkg = 0.25
    if data is not None:
        try:
            sigma_clip = data.auto_assess["sigma"]
            target_bkg = data.auto_assess["targetBkg"]
        except Exception:
            pass

    c0: list[float] = []
    mm: list[float] = []
    for c in range(nch):
        chan = a if a.ndim == 2 else a[..., c]
        med = float(np.median(chan))
        dev = float(np.mean(np.abs(chan - med)))
        cc = float(min(max(med + sigma_clip * dev, 0.0), 1.0))
        c0.append(cc)
        mm.append(find_midtones_balance(target_bkg, med - cc))

    def vis(x: float) -> float:
        s = 0.0
        for c in range(nch):
            denom = 1.0 - c0[c]
            t = (x - c0[c]) / denom if denom > 1e-8 else 0.0
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            v = mtf_scalar(mm[c], t)
            if math.isfinite(v):
                s += v
        return s / nch

    return vis


# --------------------------------------------------------------------------- #
# Analyze
# --------------------------------------------------------------------------- #

def analyze_view(img, params, data=None) -> AnalyzeResult:
    """Dry-run measurement + recommendation report (``LS.analyzeView``, js:2633-2815).

    Measures the (linear) master and reports gradient, dust lift, noise and
    chromatic tilt, emitting one-click ``recommendations`` where a validated
    field case fires. Nothing is processed and ``params`` is **not mutated**.

    Parameters
    ----------
    img : numpy.ndarray
        Float64 image in ``[0, 1]``; ``(H, W)`` mono or ``(H, W, 3)`` RGB.
    params : lazystretch.objects.model.Parameters
        Pipeline state. Read (never written): ``object_class``, ``object_id``,
        ``object_name``, ``cropPercent``, ``autoAssess``, ``inputStretched``,
        ``useMask``, ``dehaze``, ``chromaNR``, ``gradientCleanup``, ``reduceCast``,
        ``darkLaneGC``.
    data : lazystretch.data.loader.LazyStretchData, optional
        Shared data layer (constants for ``auto_assess`` / ``measure_dust`` and the
        ``vis`` STF params). ``None`` lets the callees load the bundled default.

    Returns
    -------
    AnalyzeResult
        ``.lines`` (report) and ``.recommendations`` (``{key, val, label}`` changes).
    """
    a_img = np.asarray(img, dtype=np.float64)
    W = int(a_img.shape[1])
    H = int(a_img.shape[0])
    is_color = a_img.ndim == 3

    lines: list[str] = []
    changes: list[dict] = []
    cls = params.object_class

    # --- header: object + frame (LazyStretch.js:2640-2649) -------------------
    oid = (params.object_id or "").strip()
    if len(oid) > 0:
        obj = oid + ((" - " + params.object_name) if params.object_name else "")
    else:
        obj = "(not identified — class-aware advice needs Identify)"
    lines.append("Object: " + obj + "  [class: " + _class_label(cls) + "]")

    frame = f"Frame: {W} x {H} px, " + ("RGB" if is_color else "mono")
    if params.inputStretched:
        frame += "  (marked already stretched - readings taken as-is)"
    lines.append(frame)

    # --- gradient (8 perimeter regions, post-stretch visual space) -----------
    a = None
    try:
        a = auto_assess(a_img, data)
    except Exception:
        a = None
    if a is not None:
        g = a.gradient
        if g >= 0.45:
            sev = "strong (Bortle 7/8-class light pollution)"
        elif g >= 0.20:
            sev = "moderate"
        else:
            sev = "low (clean sky)"
        lines.append(f"Gradient: {g:.2f} - {sev}.")
        gc_name = "GradientCorrection" if a.useGC else "ABE"
        if params.autoAssess:
            lines.append(f"   Auto-assess will crop {a.cropPercent:.0f}% and use {gc_name}.")
        else:
            lines.append(
                f"   Auto-assess is OFF: it would pick crop {a.cropPercent:.0f}% + {gc_name} "
                f"(your manual {params.cropPercent:.0f}% is kept)."
            )
        if g >= 0.50:
            lines.append("   The adaptive shadow floor will be auto-skipped at run time (under this much LP")
            lines.append("   its dust-lift measure protects the light pollution — IC 1396 case).")
            if g >= 0.85:
                lines.append(f"   NOTE: {g:.2f} is beyond the earlier field-tested range (max 0.73): GC alone")
                lines.append("   leaves residual mottle / LP tint at this dose.")
                if params.gradientCleanup < 0.3:
                    changes.append({
                        "key": "gradientCleanup", "val": 0.4,
                        "label": (f"Gradient cleanup 0.40 (now {params.gradientCleanup:.2f}) - flattens the "
                                  "residual mottle GC leaves at extreme LP (Rosette 0.93 case)"),
                    })
                if not params.reduceCast:
                    changes.append({
                        "key": "reduceCast", "val": True,
                        "label": ("Reduce background color cast ON - masked desat of the residual LP tint "
                                  "(Rosette 0.93 case)"),
                    })
                if not params.darkLaneGC:
                    changes.append({
                        "key": "darkLaneGC", "val": True,
                        "label": ("Dark-lane gradient model ON - at this LP dose GC leaves a large residual "
                                  "tilt the model removes (Rosette 0.93 logged 0.040-0.044 per channel "
                                  "post-GC), and it is a proven no-op on clean frames"),
                    })
            if (cls == "emission" or cls == "generic") and params.useMask:
                changes.append({
                    "key": "useMask", "val": False,
                    "label": ("Protect faint signal OFF - under a strong gradient the mask keeps the LP as a "
                              "milky veil (IC 1396 recipe)"),
                })
            if is_color and params.dehaze < 0.5:
                changes.append({
                    "key": "dehaze", "val": 0.70,
                    "label": (f"Neutralize haze 0.70 (now {params.dehaze:.2f}) - clips the veil and removes "
                              "the LP tint (IC 1396 / IC 2118 recipes)"),
                })
    else:
        lines.append("Gradient: frame too small to assess.")

    # --- dust (linear measureDust, mapped onto the post-stretch scale) -------
    d = None
    if not params.inputStretched:
        try:
            d = measure_dust(a_img, data)
        except Exception:
            d = None
    if d is not None and d.typical > 0 and a is not None:
        vis = _build_vis(a_img, data)
        lift = max(0.0, vis(d.typical) - vis(d.cleanSky))
        if lift >= 0.015:
            if a.gradient >= 0.50:
                desc = "large, but under this gradient that is light pollution, not dust"
            elif _adaptive_floor_applies_to(cls):
                desc = "dusty field (Horsehead-class); the adaptive floor will preserve it"
            else:
                desc = ("elevated - but for this class much of it is the object's own halo "
                        "(the floor is class-skipped)")
        elif lift >= 0.008:
            desc = "mild dust / background structure"
        else:
            desc = "clean, flat background"
        lines.append(f"Dust lift: ~{lift:.3f} after stretch - {desc}.")

    # --- noise (MRS-substitute sigma vs the darkest-patch sky level) ---------
    if not params.inputStretched:
        nz = None
        try:
            nz = measure_noise(a_img)
        except Exception:
            nz = None
        if nz and len(nz) > 0:
            s = sum(nz) / len(nz)
            ratio = (100.0 * s / d.cleanSky) if (d is not None and d.cleanSky > 1e-7) else 0.0
            if ratio >= 5:
                bucket = "LOW-SNR data (Rosette-class; deep masters measure ~0.3%)"
            elif ratio >= 2:
                bucket = "moderate SNR"
            elif ratio > 0:
                bucket = "deep, clean data"
            else:
                bucket = ""
            line = f"Noise: MRS sigma {s:.6f}"
            if ratio > 0:
                line += f" = {ratio:.1f}% of the sky level"
            if len(bucket) > 0:
                line += " - " + bucket
            lines.append(line + ".")
            if ratio >= 5 and is_color and params.chromaNR < 0.3:
                changes.append({
                    "key": "chromaNR", "val": 0.6,
                    "label": (f"Chroma NR 0.60 (now {params.chromaNR:.2f}) - colour speckle dominates at this "
                              "SNR; chroma-only smoothing cleaned the Rosette at 0.7 with zero structure cost"),
                })
            elif ratio >= 2 and is_color:
                lines.append("   If colour speckle shows after processing, the Chroma NR dial smooths it")
                lines.append("   without touching luminance structure (try 0.5-0.7).")

    # --- chromatic tilt (dark-lane anchor scan on the raw master) ------------
    if not params.inputStretched:
        m = None
        try:
            m = dark_lane_model(a_img)
        except Exception:
            m = None
        if m is not None and not m.get("error"):
            t = [f"{m['tilt'][c]:.4f}" for c in range(m["nc"])]
            prefix = "R/G/B " if m["nc"] >= 3 else ""
            lines.append(
                f"Chromatic tilt: {prefix}{' / '.join(t)} across the frame "
                f"({m['kept']} anchors, {m['total'] - m['kept']} rejected as on-nebula)."
            )
            lines.append("   GC/ABE removes normal tilt on frames WITH sky. Only if nebula fills the ENTIRE frame,")
            lines.append("   use the wall-to-wall recipe: dark-lane model ON, Background +0.10..+0.12,")
            lines.append("   faint-mask and adaptive floor OFF, crop 10, SCNR ON (IC 1318 recipe).")

    # --- star census: SKIPPED (no star detector in the NumPy port) -----------
    lines.append("Stars: census unavailable (no star detector in the NumPy port).")

    return AnalyzeResult(lines=lines, recommendations=changes)
