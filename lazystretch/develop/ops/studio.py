"""Studio tools — the "creative darkroom" bench.

Two flagship tools:

* **Selective Color** — Photoshop-style per-colour-group CMYK correction. Colour
  classification is done in OKLab via cosine similarity between a pixel's chroma
  direction and each group's hue direction (no atan2 → no hue wraparound), with
  per-group half-widths, chroma-proportional weighting, a chroma noise gate and a
  shadow guard. One op edits one group; stack several for a full grade.
* **Wavelet clarity** — multi-scale à-trous (starlet) sharpening + denoise, six dyadic
  layers (1,2,4,8,16,32 px), per-layer Sharpen/Denoise and a master Strength.

Both run on luminance and ratio-scale RGB to preserve hue, like the rest of the suite.
"""
from __future__ import annotations

import numpy as np

from ...processes.multiscale import _atrous_planes
from ..spikes import render_spikes
from . import Op, ParamSpec, register

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _smoothstep(e0, e1, x):
    if e1 == e0:
        return (x >= e1).astype(np.float64)
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luma(a):
    return a[..., 0] * _LUMA[0] + a[..., 1] * _LUMA[1] + a[..., 2] * _LUMA[2]


# ============================================================================
# Selective Color (OKLab)
# ============================================================================

# key, kind (0=hue, 1=tonal), OKLab hue direction (ua, ub), cos(halfWidth), tone.
# The hue directions / half-widths come from a reference OKLab selective-colour model.
_SC_GROUPS = [
    ("Reds",     0,  0.872633,  0.488376, 0.151117,  0),
    ("Yellows",  0, -0.338233,  0.941063, 0.420635,  0),
    ("Greens",   0, -0.793304,  0.608826, 0.657706,  0),
    ("Cyans",    0, -0.966962, -0.254922, 0.343740,  0),
    ("Blues",    0, -0.103625, -0.994616, 0.228065,  0),
    ("Magentas", 0,  0.851392, -0.524530, 0.309356,  0),
    ("Whites",   1,  0.0,       0.0,      0.0,        1),
    ("Neutrals", 1,  0.0,       0.0,      0.0,        0),
    ("Blacks",   1,  0.0,       0.0,      0.0,       -1),
]
_SC_GROUP_NAMES = [g[0] for g in _SC_GROUPS]

_SC_CHROMA_FULL_MIN = 0.06
_SC_CHROMA_FULL_MAX = 0.30
_SC_CHROMA_NOISE0 = 0.010
_SC_CHROMA_NOISE1 = 0.025
_SC_GUARD_MAX_LUM = 0.60
_SC_GUARD_RAMP = 0.60
_SC_SMOOTH_MAX_PX = 12.0


def _rgb_to_oklab(a):
    """Linear-ish RGB [0,1] → OKLab (L, a, b). Björn Ottosson's matrices."""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = np.cbrt(l); m_ = np.cbrt(m); s_ = np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return L, A, B


def _sc_membership(a, gi, range_pct, guard_pct):
    """Soft membership [0,1] of every pixel in colour group ``gi``."""
    key, kind, ua, ub, cosHW, tone = _SC_GROUPS[gi]
    L, A, B = _rgb_to_oklab(a)
    chroma = np.sqrt(A * A + B * B)
    t = np.clip(range_pct / 100.0, 0.0, 1.0)
    chroma_full = max(0.005, _SC_CHROMA_FULL_MAX - (_SC_CHROMA_FULL_MAX - _SC_CHROMA_FULL_MIN) * t)

    # chroma weight (pale tint → pale correction) and the noise gate.
    w_chroma = np.clip(chroma / chroma_full, 0.0, 1.0)
    noise = _smoothstep(_SC_CHROMA_NOISE0, _SC_CHROMA_NOISE1, chroma)

    if kind == 0:   # hue group
        hue_mult = 0.6 + 0.8 * t
        hw = np.arccos(np.clip(cosHW, -1.0, 1.0)) * hue_mult
        inner = np.cos(np.clip(0.5 * hw, 0.0, np.pi))
        outer = np.cos(np.clip(hw, 0.0, np.pi))
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(chroma > 1e-6, (ua * A + ub * B) / np.maximum(chroma, 1e-9), -1.0)
        member = _smoothstep(outer, inner, cos)
        f = member * w_chroma * noise
    else:           # tonal group (Whites / Neutrals / Blacks)
        achroma = 1.0 - _smoothstep(_SC_CHROMA_NOISE1, chroma_full, chroma)
        if tone > 0:                       # Whites
            f = _smoothstep(0.60, 0.95, L)
        elif tone < 0:                     # Blacks
            f = 1.0 - _smoothstep(0.05, 0.40, L)
        else:                              # Neutrals
            f = np.clip(1.0 - np.abs(L - 0.5) / 0.5, 0.0, 1.0) * achroma

    # shadow guard: exclude the darkest sky (scaled by the user's Guard).
    gt = np.clip(guard_pct / 100.0, 0.0, 1.0)
    if gt > 0:
        thr = gt * _SC_GUARD_MAX_LUM
        f = f * _smoothstep(thr * _SC_GUARD_RAMP, thr, L)
    return np.clip(f, 0.0, 1.0)


def _ink_relative(ch, ink):
    if ink >= 0:
        return ch * (1.0 - ink)
    return ch + (-ink) * (1.0 - ch)


def _apply_selective_color(img, p):
    if img.ndim != 3:
        return img
    a = np.asarray(img, dtype=np.float64)
    gi = int(p.get("group", 0))
    C = float(p.get("cyan", 0.0)); M = float(p.get("magenta", 0.0))
    Y = float(p.get("yellow", 0.0)); K = float(p.get("black", 0.0))
    if C == 0 and M == 0 and Y == 0 and K == 0:
        return a
    absolute = bool(p.get("absolute", False))
    lum_protect = bool(p.get("lum_protect", True))

    f = _sc_membership(a, gi, float(p.get("range", 50.0)), float(p.get("guard", 5.0)))
    smooth = float(p.get("smooth", 5.0))
    if smooth > 0:
        try:
            from scipy.ndimage import gaussian_filter
            f = gaussian_filter(f, sigma=max(0.5, _SC_SMOOTH_MAX_PX * smooth / 100.0))
        except Exception:
            pass

    out = a.copy()
    if absolute:
        out[..., 0] = a[..., 0] - C
        out[..., 1] = a[..., 1] - M
        out[..., 2] = a[..., 2] - Y
        if K != 0:
            out = out - K
    else:
        out[..., 0] = _ink_relative(a[..., 0], C)
        out[..., 1] = _ink_relative(a[..., 1], M)
        out[..., 2] = _ink_relative(a[..., 2], Y)
        if K >= 0:
            out = out * (1.0 - K)
        else:
            out = out + (-K) * (1.0 - out)
    out = np.clip(out, 0.0, 1.0)

    if lum_protect:
        L0 = _luma(a); L1 = _luma(out)
        scale = np.where(L1 > 1e-6, L0 / np.maximum(L1, 1e-6), 1.0)
        out = np.clip(out * scale[..., None], 0.0, 1.0)

    return np.clip(a + f[..., None] * (out - a), 0.0, 1.0)


# ============================================================================
# Wavelet clarity / sharpening (à-trous starlet)
# ============================================================================

_WV_LAYERS = 6


def _wavelet_sharpen(img, p):
    """Per-layer à-trous sharpen + denoise on luminance (hue-preserving on RGB)."""
    strength = float(np.clip(p.get("strength", 100.0) / 100.0, 0.0, 1.0))
    sharpen = [float(p.get(f"sharpen{i}", 0.0)) for i in range(1, _WV_LAYERS + 1)]
    denoise = [float(p.get(f"denoise{i}", 0.0)) for i in range(1, _WV_LAYERS + 1)]
    if not any(s > 0 for s in sharpen) and not any(d > 0 for d in denoise):
        return img
    if not any(d > 0 for d in denoise) and strength <= 0:
        return img

    a = np.asarray(img, dtype=np.float64)
    is_rgb = (a.ndim == 3)
    L = _luma(a) if is_rgb else a.copy()
    planes, resid = _atrous_planes(L, _WV_LAYERS)

    out = resid.copy()
    for i, w in enumerate(planes):
        s = np.clip(sharpen[i] / 100.0, 0.0, 1.0)
        d = np.clip(denoise[i] / 100.0, 0.0, 1.0)
        wi = w
        if d > 0:
            sig = 1.4826 * np.median(np.abs(w)) + 1e-9      # robust noise sigma at this scale
            soft = np.sign(w) * np.maximum(np.abs(w) - 3.0 * sig, 0.0)
            wdn = soft * (1.0 - 0.95 * d)                    # attenuate residual detail
            wi = w + d * (wdn - w)                           # dose by the Denoise amount
        bias = (s ** 1.5) * 3.0 * strength                   # per-layer sharpen bias
        out = out + (1.0 + bias) * wi
    newL = np.clip(out, 0.0, 1.0)

    if not is_rgb:
        return newL
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(L > 1e-6, newL / np.maximum(L, 1e-6), 1.0)
    return np.clip(a * ratio[..., None], 0.0, 1.0)


# ---------------------------------------------------------------- registration
register(Op(
    "selective_color", "Selective Color", "Studio", _apply_selective_color,
    needs_color=True,
    params=[
        ParamSpec("group", "Colour group", "choice", default=0, choices=_SC_GROUP_NAMES),
        ParamSpec("cyan", "Cyan ↔ Red", "float", -1.0, 1.0, 0.0, 2),
        ParamSpec("magenta", "Magenta ↔ Green", "float", -1.0, 1.0, 0.0, 2),
        ParamSpec("yellow", "Yellow ↔ Blue", "float", -1.0, 1.0, 0.0, 2),
        ParamSpec("black", "Black", "float", -1.0, 1.0, 0.0, 2),
        ParamSpec("range", "Range (selection width)", "float", 0.0, 100.0, 50.0, 0),
        ParamSpec("guard", "Shadow guard", "float", 0.0, 100.0, 5.0, 0),
        ParamSpec("smooth", "Selection smoothing", "float", 0.0, 100.0, 5.0, 0),
        ParamSpec("absolute", "Absolute mode", "bool", default=False),
        ParamSpec("lum_protect", "Preserve luminance", "bool", default=True),
    ],
    tooltip="Photoshop-style per-colour-group CMYK correction (OKLab selection).",
))

# ============================================================================
# Star diffraction spikes
# ============================================================================
def _star_spikes(img, p):
    return render_spikes(
        img, list(p.get("stars") or []),
        count=int(p.get("count", 4)), angle_deg=float(p.get("angle", 0.0)),
        thickness=float(p.get("thickness", 1.0)), intensity=float(p.get("intensity", 1.0)),
        colored=bool(p.get("colored", True)), base_len=float(p.get("length", 0.06)),
        fringe=float(p.get("fringe", 0.0)), jitter=float(p.get("jitter", 0.0)),
        asymmetry=float(p.get("asymmetry", 0.0)),
    )


register(Op(
    "star_spikes", "Star spikes", "Studio", _star_spikes, heavy=True,
    params=[
        ParamSpec("stars", "Stars", "stars", default=[]),
        ParamSpec("count", "Spikes (3–32)", "int", 3, 32, 4, 0),
        ParamSpec("length", "Length (default / selected star)", "float", 0.01, 0.35, 0.06, 2),
        ParamSpec("angle", "Rotation°", "float", 0.0, 90.0, 0.0, 0),
        ParamSpec("thickness", "Thickness", "float", 0.3, 4.0, 1.0, 1),
        ParamSpec("intensity", "Intensity", "float", 0.0, 2.0, 1.0, 2),
        ParamSpec("fringe", "Chromatic fringe", "float", 0.0, 1.0, 0.0, 2,
                  tooltip="Blend a warm→cool spectral gradient into the outer arms "
                          "(refractor-style diffraction colour). 0 = off."),
        ParamSpec("jitter", "Arm-length jitter", "float", 0.0, 1.0, 0.0, 2,
                  tooltip="Vary each arm's length a little (0 = all arms equal)."),
        ParamSpec("asymmetry", "Per-arm colour", "float", 0.0, 1.0, 0.0, 2,
                  tooltip="Give each arm a slightly different hue (0 = all arms same colour)."),
        ParamSpec("colored", "Tint by star colour", "bool", default=True),
        ParamSpec("max_stars", "Auto-detect count", "int", 1, 200, 30, 0),
    ],
    tooltip="Diffraction spikes on stars. Auto-detects the brightest stars; click one to "
            "select, click empty to add, drag to move, right-click to remove. Spike count "
            "(3–32), rotation, thickness and intensity are global; Length applies to the "
            "selected star (and sets new stars). Optional chromatic fringe + arm jitter.",
))

register(Op(
    "wavelet_sharpen", "Wavelet clarity", "Studio", _wavelet_sharpen, heavy=True,
    params=[
        ParamSpec("sharpen1", "Sharpen 1 (1px)", "float", 0.0, 100.0, 0.0, 0),
        ParamSpec("sharpen2", "Sharpen 2 (2px)", "float", 0.0, 100.0, 0.0, 0),
        ParamSpec("sharpen3", "Sharpen 3 (4px)", "float", 0.0, 100.0, 0.0, 0),
        ParamSpec("sharpen4", "Sharpen 4 (8px)", "float", 0.0, 100.0, 0.0, 0),
        ParamSpec("sharpen5", "Sharpen 5 (16px)", "float", 0.0, 100.0, 0.0, 0),
        ParamSpec("sharpen6", "Sharpen 6 (32px)", "float", 0.0, 100.0, 0.0, 0),
        ParamSpec("denoise1", "Denoise 1 (1px)", "float", 0.0, 100.0, 0.0, 0),
        ParamSpec("denoise2", "Denoise 2 (2px)", "float", 0.0, 100.0, 0.0, 0),
        ParamSpec("denoise3", "Denoise 3 (4px)", "float", 0.0, 100.0, 0.0, 0),
        ParamSpec("strength", "Master strength", "float", 0.0, 100.0, 100.0, 0),
    ],
    tooltip="Multi-scale à-trous wavelet sharpening + denoise.",
))
