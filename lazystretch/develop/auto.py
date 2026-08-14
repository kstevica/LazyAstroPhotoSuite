"""Auto-develop: analyse a (stretched) image and suggest an ordered finishing recipe.

``analyze(img)`` measures the WHOLE image and returns a list of suggested steps —
``{"name", "params", "reason"}`` — already sorted into the correct finishing order:

    gradient cleanup → background neutralise → SCNR → de-veil → chroma NR →
    noise reduction → local contrast → contrast → saturation → highlight roll-off

Each step is included only when its measurement crosses a conservative threshold, and
its strength scales with the measured amount, so a clean image yields few (or no)
suggestions. The GUI runs this on the full frame once, then previews the recipe on a
downscaled proxy for speed; every op name/param here matches the Develop op registry.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# Canonical finishing order — suggestions are sorted by this.
_ORDER = [
    "gradient_cleanup", "bg_neutralize", "scnr", "deveil", "chroma_nr",
    "noise_reduction", "local_contrast", "contrast", "saturation", "highlight_rolloff",
]


def _luma(a3: np.ndarray) -> np.ndarray:
    return a3[..., 0] * _LUMA[0] + a3[..., 1] * _LUMA[1] + a3[..., 2] * _LUMA[2]


def _highpass(x: np.ndarray) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    return x - uniform_filter(x, 3, mode="nearest")


def _robust_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x)))          # x is ~zero-mean (a high-pass)


def _bg_channels(a3: np.ndarray) -> "tuple[float, float, float]":
    """Per-channel background level (35th percentile — the sky dominates an astro frame)."""
    return tuple(float(np.percentile(a3[..., c], 35)) for c in range(3))  # type: ignore


def _gradient_ptp(L: np.ndarray) -> float:
    """Peak-to-peak of a robust degree-1 plane fit to the luminance (star-safe)."""
    h, w = L.shape
    g = 24
    hh, ww = (h // g) * g, (w // g) * g
    if hh < g or ww < g:
        return 0.0
    # block medians on a g×g grid (median → robust to stars)
    grid = np.median(L[:hh, :ww].reshape(g, hh // g, g, ww // g), axis=(1, 3))
    ys, xs = np.meshgrid(np.linspace(0, 1, g), np.linspace(0, 1, g), indexing="ij")
    A = np.column_stack([xs.ravel(), ys.ravel(), np.ones(g * g)])
    coef, *_ = np.linalg.lstsq(A, grid.ravel(), rcond=None)
    corners = [coef[0] * x + coef[1] * y + coef[2] for x in (0.0, 1.0) for y in (0.0, 1.0)]
    return float(max(corners) - min(corners))


def analyze(img: np.ndarray) -> List[Dict]:
    """Return an ordered list of suggested finishing steps for ``img`` (float [0,1])."""
    a = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    is_rgb = a.ndim == 3 and a.shape[2] == 3
    a3 = a if is_rgb else np.repeat(np.atleast_3d(a), 3, axis=2)
    L = _luma(a3)
    bg = float(np.median(L))
    steps: List[Dict] = []

    def add(name, params, reason):
        steps.append({"name": name, "params": params, "reason": reason})

    # 1. Gradient ------------------------------------------------------------
    try:
        ptp = _gradient_ptp(L)
        if ptp > 0.02:
            add("gradient_cleanup", {"strength": round(float(min(1.0, ptp / 0.08)), 2)},
                f"Background gradient ≈ {ptp * 100:.0f}%")
    except Exception:
        pass

    # 2/3. Colour cast + green (RGB only) -----------------------------------
    if is_rgb:
        r, gch, b = _bg_channels(a3)
        spread = max(r, gch, b) - min(r, gch, b)
        if spread > 0.008:
            add("bg_neutralize", {"target": round(float(np.clip(min(r, gch, b), 0.0, 0.2)), 3)},
                f"Background colour cast (Δ {spread * 100:.1f}%)")
        green_excess = gch - 0.5 * (r + b)
        if green_excess > 0.006:
            add("scnr", {"amount": round(float(min(1.0, green_excess / 0.03)), 2)},
                "Residual green cast")

    # 4. De-veil (milky, warm background) -----------------------------------
    if is_rgb and bg > 0.08:
        r, gch, b = _bg_channels(a3)
        if (r - b) > 0.01:
            add("deveil", {"amount": round(float(min(0.8, (bg - 0.05) / 0.15)), 2)},
                f"Milky background floor ({bg * 100:.0f}%)")

    # 5. Chroma noise (RGB only) --------------------------------------------
    if is_rgb:
        try:
            cn = 0.5 * (_robust_sigma(_highpass(a3[..., 0] - a3[..., 1]))
                        + _robust_sigma(_highpass(a3[..., 1] - a3[..., 2])))
            if cn > 0.010:
                add("chroma_nr", {"amount": round(float(min(1.0, cn / 0.03)), 2)},
                    "Colour speckle in the background")
        except Exception:
            pass

    # 6. Luminance noise (measured in the background) -----------------------
    try:
        hp = _highpass(L)
        m = L < (bg + 0.05)
        vals = hp[m] if int(m.sum()) > 500 else hp
        ln = _robust_sigma(vals)
        if ln > 0.006:
            add("noise_reduction", {"amount": round(float(min(0.85, ln / 0.02)), 2)},
                "Background luminance noise")
    except Exception:
        pass

    # 7. Local contrast (soft / low mid-scale structure) --------------------
    try:
        from scipy.ndimage import uniform_filter
        mid = float(np.std(L - uniform_filter(L, 16, mode="nearest")))
        if mid < 0.020:
            add("local_contrast", {"amount": 0.12}, "Low local contrast (soft)")
    except Exception:
        pass

    # 8. Global contrast (compressed tonal range) ---------------------------
    spread99 = float(np.percentile(L, 99.5) - np.percentile(L, 50))
    if spread99 < 0.35:
        add("contrast", {"pivot": round(bg, 2), "strength": 0.12}, "Flat tonal range")

    # 9. Saturation (muted colour) ------------------------------------------
    if is_rgb:
        bright = L > (bg + 0.05)
        if int(bright.sum()) < 500:
            bright = L > np.percentile(L, 80)
        sat = float(np.mean((a3.max(-1) - a3.min(-1))[bright])) if bright.any() else 0.0
        if sat < 0.10:
            amt = float(np.clip(0.15 + (0.10 - sat) / 0.10 * 0.30, 0.0, 0.5))
            add("saturation", {"amount": round(amt, 2)}, "Muted colour")

    # 10. Highlights (blown cores) ------------------------------------------
    blown = float(np.mean(L > 0.98))
    if blown > 0.002:
        add("highlight_rolloff", {"knee": 0.82}, f"{blown * 100:.1f}% blown highlights")

    steps.sort(key=lambda s: _ORDER.index(s["name"]))
    return steps


# How each auto step is gated by a semantic mask (name, invert). GLOBAL steps are absent
# on purpose — gradient_cleanup (plane fit), bg_neutralize (global colour offset), scnr
# (green removal must reach stars) and contrast (global tone) would seam or misbehave if
# masked, so they run whole-frame.
_SEMANTIC_GATES = {
    # colour — don't over-saturate bright star cores
    "saturation":        ("Star cores", True),
    # detail / noise — protect the detailed nebula + stars, work the rest
    "chroma_nr":         ("Detail (protect)", True),
    "noise_reduction":   ("Detail (protect)", True),
    # structure — only inside the subject
    "local_contrast":    ("Nebulosity", False),
    "structure":         ("Nebulosity", False),
    # background / veil — deepen the veil but protect the bright nebula
    "deveil":            ("Nebulosity", True),
    "dehaze":            ("Nebulosity", True),
    "reduce_cast":       ("Sky", False),
    # highlights — only the blown cores
    "highlight_rolloff": ("Cores", False),
    # emission — only where Hα actually is
    "emission_boost":    ("Hα (red)", False),
}


def auto_develop_plan(img: np.ndarray, snr=None, existing=None) -> dict:
    """Recipe (``analyze``) + the semantic masks each step should be gated by.

    Returns ``{"steps": [{name, params, reason, mask, mask_invert}], "masks": {name: array}}``.
    Only the masks the caller must ADD are in ``masks``; if a gate's mask is already in the
    library (``existing`` = a collection of current mask names) it is reused by name and NOT
    recomputed — so masks you generated/refined with "Auto masks" are the ones used.
    """
    from .semantic import segment
    from .masks import combine_masks

    existing = set(existing or ())
    steps = analyze(img)
    try:
        sem = segment(img, snr)
    except Exception:
        sem = {}

    # Composite protection masks derived from the semantic set.
    available = dict(sem)
    if "Nebulosity" in sem and "Stars" in sem:
        available["Detail (protect)"] = combine_masks(sem["Nebulosity"], sem["Stars"], "union")
    if "Bright stars" in sem and "Cores" in sem:
        available["Star cores"] = combine_masks(sem["Bright stars"], sem["Cores"], "union")

    used: dict = {}
    for s in steps:
        s.setdefault("mask", None)
        s.setdefault("mask_invert", False)
        gate = _SEMANTIC_GATES.get(s["name"])
        if gate is None:
            continue
        mname, invert = gate
        if mname in existing:                 # already in the library → reuse it by name
            s["mask"] = mname
            s["mask_invert"] = invert
        elif mname in available:              # not in the library → we'll add our own
            s["mask"] = mname
            s["mask_invert"] = invert
            used[mname] = available[mname]
    return {"steps": steps, "masks": used}
