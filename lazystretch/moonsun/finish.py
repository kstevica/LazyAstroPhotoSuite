"""LazyMoonSun solar/lunar finish — deterministic, physics-aware.

Port of ``SUN.finishView`` (LazyMoonSun.js:1207-1810): measure WB + geometry → one MTF
lift → optional solar surface-flatten engine → limb-protected à-trous wavelet sharpen →
radial ring eraser → contrast / LHE / tone / saturation → final tone triple.

Faithful where the .js is pure math (WB, MTF, surface engine, wavelet, ring eraser, golden
tone, contrast, final HT). Two steps stand in for PixInsight processes and are documented
approximations: LHE (LocalHistogramEqualization → masked unsharp local contrast) and
saturation (ColorSaturation → HSV boost). Both default to 0 for the Sun preset.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from . import register as reg

SKY_ANCHOR = 0.60
_NRB = 96


def _noop(_m: str) -> None:
    pass


def mtf(m: float, x: np.ndarray) -> np.ndarray:
    """PI midtones transfer function (SUN.mtf). m<0.5 brightens, m>0.5 darkens."""
    x = np.asarray(x, dtype=np.float64)
    denom = (2.0 * m - 1.0) * x - m
    out = ((m - 1.0) * x) / np.where(denom == 0.0, 1e-12, denom)
    return out


def _ht(img: np.ndarray, sh: float, mid: float, hi: float) -> np.ndarray:
    """HistogramTransformation on the RGB/K channel: clip to [sh,hi], rescale, MTF(mid)."""
    span = max(1e-6, hi - sh)
    x = np.clip((img - sh) / span, 0.0, 1.0)
    return np.clip(mtf(mid, x), 0.0, 1.0)


# --------------------------------------------------------------- geometry / classify


def _p999_hist(v: np.ndarray) -> float:
    return float(np.quantile(v, 0.999))


def classify_body(wk: np.ndarray, lit_thr: float) -> dict:
    """SUN.classifyBody: limb darkens outward → sun; brightens → moon; else unknown."""
    H, W = wk.shape
    R = 6
    offs = [(R, 0), (-R, 0), (0, R), (0, -R), (4, 4), (-4, 4), (4, -4), (-4, -4)]
    ys = np.arange(R, H - R, 2)
    xs = np.arange(R, W - R, 2)
    if ys.size == 0 or xs.size == 0:
        return {"body": "unknown", "ratio": 1.0}
    grid = wk[np.ix_(ys, xs)]
    lit = grid > lit_thr
    is_edge = np.zeros_like(lit)
    for dx, dy in offs:
        nb = wk[np.ix_(ys + dy, xs + dx)]
        is_edge |= (nb <= lit_thr)
    edge = lit & is_edge
    interior = lit & ~is_edge
    ne, ni = int(edge.sum()), int(interior.sum())
    if ne < 50 or ni < 200:
        return {"body": "unknown", "ratio": 1.0}
    ratio = (grid[edge].sum() / ne) / (grid[interior].sum() / ni)
    body = "sun" if ratio < 0.93 else "moon" if ratio > 1.03 else "unknown"
    return {"body": body, "ratio": float(ratio)}


def _disc_geometry(wk: np.ndarray, disc_thr: float) -> Optional[dict]:
    """Centroid, radius, fill, and 96-bin radial median profile of the lit disc."""
    ys, xs = np.nonzero(wk > disc_thr)
    if xs.size < 100:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    n = xs.size
    R = np.sqrt(n / np.pi)
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    max_r = float(r.max())
    vals = wk[ys, xs]
    bins = np.minimum(_NRB - 1, np.floor(r / (R * 1.03) * _NRB).astype(int))
    prof = [None] * _NRB
    for b in range(_NRB):
        sel = vals[bins == b]
        if sel.size > 20:
            prof[b] = float(np.median(sel))
    first_valid = next((p for p in prof if p is not None), 1.0)
    last = first_valid
    filled = []
    for p in prof:
        if p is not None:
            last = p
        filled.append(last)
    fill = n / (np.pi * max_r * max_r) if max_r > 0 else 0.0
    c0 = max(1e-6, max(filled[0], filled[1]))
    return {"cx": cx, "cy": cy, "R": R, "max_r": max_r, "fill": float(fill),
            "prof": filled, "c0": c0, "n": n}


# --------------------------------------------------------------------- à-trous sharpen

_B3 = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0


def _atrous_convolve(a: np.ndarray, step: int) -> np.ndarray:
    """One à-trous (holes) B3-spline smoothing at dilation ``step``, separable."""
    k = np.zeros(4 * step + 1)
    k[::step] = _B3
    from scipy.ndimage import convolve1d
    out = convolve1d(a, k, axis=0, mode="reflect")
    return convolve1d(out, k, axis=1, mode="reflect")


def atrous_sharpen(img: np.ndarray, sharp: float, mask: Optional[np.ndarray] = None,
                   nlayers: int = 6) -> np.ndarray:
    """MultiscaleLinearTransform sharpen: bias detail layers 1-3 (0.6/1.0/0.6 · sharp).

    Starlet à-trous decomposition; each biased layer's coefficients scale by (1+bias),
    with light two-sided deringing (clamp the boosted detail against overshoot). ``mask``
    (0..1, same H×W) limits where the added detail lands (the limb-protection mask).
    """
    a = np.asarray(img, dtype=np.float64)
    gains = {1: 1.0 + 0.6 * sharp, 2: 1.0 + sharp, 3: 1.0 + 0.6 * sharp}

    def _one(plane: np.ndarray) -> np.ndarray:
        c = plane.copy()
        boosted = np.zeros_like(plane)
        added = np.zeros_like(plane)
        for i in range(nlayers):
            step = 1 << i
            c_next = _atrous_convolve(c, step)
            detail = c - c_next
            g = gains.get(i, 1.0)
            if g != 1.0:
                added += (g - 1.0) * detail
            boosted += detail
            c = c_next
        boosted += c                       # residual
        # two-sided deringing: damp the *extra* detail where it overshoots [0,1]
        result = boosted + added
        over = np.maximum(0.0, result - 1.0)
        under = np.maximum(0.0, -result)
        added = added - 0.95 * over + 0.95 * under
        result = boosted + added
        return result

    if a.ndim == 3:
        out = np.stack([_one(a[..., c]) for c in range(a.shape[2])], axis=-1)
    else:
        out = _one(a)
    if mask is not None:
        m = mask[..., None] if (a.ndim == 3 and mask.ndim == 2) else mask
        out = a + m * (out - a)
    return np.clip(out, 0.0, 1.0)


def _interior_mask(shape, cx, cy, R, feather=(40, 10)) -> np.ndarray:
    """Feathered disc-interior mask: 1 inside R-40, cosine ramp to 0 at R-10."""
    H, W = shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r0, r1 = R - feather[0], R - feather[1]
    m = np.where(r < r0, 1.0,
                 np.where(r > r1, 0.0, (1 + np.cos(np.pi * (r - r0) / max(1e-6, r1 - r0))) / 2))
    return m


# ------------------------------------------------------------------- the finish


def finish(img: np.ndarray, params, log: Callable[[str], None] = _noop) -> Optional[dict]:
    """Run the full finish. Returns {'image', 'body', 'do_surface'} or None if no disc."""
    a = np.asarray(img, dtype=np.float64)
    is_color = a.ndim == 3 and a.shape[2] >= 3
    nch = 3 if is_color else 1

    wk, _k = reg.working(a, 1024)
    wkK = wk.shape[1] / a.shape[1] if a.shape[1] else 1.0    # working→full scale
    p999 = _p999_hist(wk)
    if p999 < 0.01:
        log(f"No disc found (p99.9 {p999:.4f}).")
        return None
    disc_thr, sky_thr = 0.5 * p999, 0.10 * p999

    lum_wk = wk                                              # working image is already intensity
    disc_mask = lum_wk > disc_thr
    if disc_mask.sum() < 100:
        log("Disc mask found almost no lit pixels — cannot finish.")
        return None

    # per-channel disc/sky medians (sample the full image at working locations via resample)
    wk_rgb = _working_rgb(a) if is_color else wk[..., None]
    disc_med = [float(np.median(wk_rgb[..., c][disc_mask])) for c in range(nch)]
    sky_sel = (lum_wk < sky_thr)
    sky_med = [float(np.median(wk_rgb[..., c][sky_sel])) if sky_sel.sum() else 0.0
               for c in range(nch)]

    geom = _disc_geometry(lum_wk, disc_thr)
    body = classify_body(lum_wk, disc_thr)
    log(f"Auto-classify: {body['body']} (edge/interior {body['ratio']:.2f}).")

    gains = [1.0, 1.0, 1.0]
    if params.do_wb and nch >= 3:
        for c in range(3):
            gains[c] = disc_med[1] / max(1e-6, disc_med[c])
        log(f"Neutralize: WB gains R {gains[0]:.4f} G 1.0 B {gains[2]:.4f}")

    dm = float(np.mean([disc_med[c] * gains[c] for c in range(nch)]))
    sm = float(np.mean([sky_med[c] * gains[c] for c in range(nch)]))

    do_surface = (params.surface > 0 and geom is not None and geom["fill"] > 0.90
                  and body["body"] != "moon")

    dm = min(dm, 0.98)
    c0 = max(0.0, SKY_ANCHOR * sm)
    x = (dm - c0) / (1.0 - c0)
    t = params.disc_target
    denom = (2 * t - 1) * x - t
    m = (x * (t - 1)) / denom if denom != 0 else -1
    if not (0.0 < m < 1.0):
        log(f"Degenerate midtones solve (m={m:.4f}) — refusing.")
        return None
    log(f"Lift: black point {c0:.5f}, midtones {m:.4f} → disc lands at {t:.2f}")

    out = np.array(a, dtype=np.float64, copy=True)

    if do_surface:
        out = _surface_engine(out, geom, disc_med, m, params, wkK, nch, log)

    if params.do_wb and nch >= 3:
        out[..., 0] *= gains[0]
        out[..., 2] *= gains[2]
        out = np.clip(out, 0.0, 1.0)

    out = _ht(out, c0, m, 1.0)

    if params.sharp > 0:
        mask = None
        if geom is not None and geom["fill"] > 0.90 and geom["R"] > 0:
            mask = _interior_mask(out.shape, geom["cx"] / wkK, geom["cy"] / wkK,
                                  geom["R"] / wkK)
        out = atrous_sharpen(out, params.sharp, mask=mask)
        log(f"Sharpen: wavelet bias L2-4 ({0.6*params.sharp:.2f}/{params.sharp:.2f}/"
            f"{0.6*params.sharp:.2f}){' limb-masked' if mask is not None else ''}")

    if geom is not None and geom["fill"] > 0.90 and geom["R"] > 0:
        out = _ring_eraser(out, geom, wkK, nch, log)

    if params.contrast > 0:
        out = _contrast_scurve(out, params.contrast)
        log(f"Contrast: S-curve {params.contrast:.2f}")

    if params.lhe > 0 and geom is not None and geom["R"] > 0:
        mask = (_interior_mask(out.shape, geom["cx"] / wkK, geom["cy"] / wkK, geom["R"] / wkK)
                if geom["fill"] > 0.90 else None)
        out = _lhe(out, params.lhe, mask)
        log(f"Local contrast: LHE {params.lhe:.2f} (approximation)")

    if params.tone == "golden" and nch >= 3:
        out = _golden_tone(out)
        log("Tone: golden duotone applied (luminance-mapped)")

    if params.sat > 0 and nch >= 3:
        out = _saturation(out, params.sat)
        log(f"Saturation: +{params.sat:.2f} (approximation)")

    if params.ht_sh > 0.0005 or abs(params.ht_mid - 0.5) > 0.0005 or params.ht_hi < 0.9995:
        sh = min(0.2, max(0.0, params.ht_sh))
        mid = min(0.8, max(0.2, params.ht_mid))
        hi = min(1.0, max(0.6, params.ht_hi))
        out = _ht(out, sh, mid, hi)
        log(f"Tone triple: black {sh:.3f}, midtones {mid:.3f}, highlights {hi:.3f}")

    return {"image": np.clip(out, 0.0, 1.0), "body": body["body"], "do_surface": do_surface}


def _working_rgb(a: np.ndarray) -> np.ndarray:
    """Downsample an RGB image to the 1024 working size, per channel."""
    long_edge = max(a.shape[0], a.shape[1])
    if long_edge <= 1024:
        return a[..., :3].astype(np.float64)
    k = 1024 / long_edge
    return np.stack([reg.zoom(a[..., c], k, order=1) for c in range(3)], axis=-1)


def _surface_engine(out, geom, disc_med, m, params, wkK, nch, log) -> np.ndarray:
    """Flatten limb darkening + boost residual contrast on the full-res disc patch."""
    prof, c0p = geom["prof"], geom["c0"]
    full_cx, full_cy = geom["cx"] / wkK, geom["cy"] / wkK
    full_r = (geom["R"] * 1.03) / wkK
    H, W = out.shape[0], out.shape[1]
    px0, px1 = max(0, int(np.floor(full_cx - full_r))), min(W - 1, int(np.ceil(full_cx + full_r)))
    py0, py1 = max(0, int(np.floor(full_cy - full_r))), min(H - 1, int(np.ceil(full_cy + full_r)))

    prof_arr = np.array(prof) / c0p
    wsum = sum(i + 0.5 for i in range(_NRB))
    fmean = sum((i + 0.5) * max(0.15, prof_arr[i]) for i in range(_NRB)) / wsum
    C0c = [disc_med[c] / max(1e-6, fmean) for c in range(nch)]
    K, F = params.surface, params.flatten
    tC = min(0.95, params.disc_target)
    m_inv = 1.0 - m
    g_den = max(1e-6, float(mtf(m_inv, tC)))

    yy, xx = np.mgrid[py0:py1 + 1, px0:px1 + 1]
    r = np.sqrt((xx - full_cx) ** 2 + (yy - full_cy) ** 2)
    inside = r < full_r
    bp = np.clip(r / full_r * _NRB, 0, _NRB - 1)
    bi = np.minimum(_NRB - 1, bp.astype(int))
    bj = np.minimum(_NRB - 1, bi + 1)
    bf = bp - bi
    fl = np.maximum(0.15, prof_arr[bi] * (1 - bf) + prof_arr[bj] * bf)
    gl = mtf(m_inv, tC * np.power(fl, 1 - F)) / g_den
    k_fade = np.clip((fl - 0.42) / 0.13, 0, 1)
    fade = np.clip((fl - 0.25) / 0.17, 0, 1)

    patch = out[py0:py1 + 1, px0:px1 + 1]
    if patch.ndim == 2:
        patch = patch[..., None]
    new = patch.copy()
    for c in range(nch):
        s = patch[..., c]
        ratio = s / np.maximum(1e-6, C0c[c] * fl)
        bfac = np.clip(1 + K * k_fade * (ratio - 1), 0.45, 1.6)
        o = C0c[c] * gl * bfac
        o = s * (1 - fade) + o * fade
        new[..., c] = np.where(inside, np.clip(o, 0, 1), s)
    if out.ndim == 2:
        out[py0:py1 + 1, px0:px1 + 1] = new[..., 0]
    else:
        out[py0:py1 + 1, px0:px1 + 1] = new
    log(f"Surface: flatten {F:.2f}, boost {K:.1f} (disc R {full_r:.0f}px, fill {geom['fill']:.2f})")
    return out


def _ring_eraser(out, geom, wkK, nch, log) -> np.ndarray:
    """Detrend rotationally-symmetric residual lines in the limb zone [0.88R, 1.14R]."""
    cx, cy, R = geom["cx"] / wkK, geom["cy"] / wkK, geom["R"] / wkK
    r_in, r_out, NB = 0.88 * R, 1.14 * R, 120
    bw = (r_out - r_in) / NB
    H, W = out.shape[0], out.shape[1]
    bx0, bx1 = max(0, int(np.floor(cx - r_out))), min(W - 1, int(np.ceil(cx + r_out)))
    by0, by1 = max(0, int(np.floor(cy - r_out))), min(H - 1, int(np.ceil(cy + r_out)))
    yy, xx = np.mgrid[by0:by1 + 1, bx0:bx1 + 1]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    patch = out[by0:by1 + 1, bx0:bx1 + 1]
    lum = patch.mean(axis=2) if patch.ndim == 3 else patch
    band = (r >= r_in) & (r < r_out)
    # clip to [0, NB-1]: bidx is evaluated over the whole patch (incl. r < r_in, which
    # would be a large negative index) before `band` masks it — the out-of-band values
    # are discarded by np.where but must still be valid indices.
    bidx = np.clip(((r - r_in) / bw).astype(int), 0, NB - 1)
    prof = np.zeros(NB)
    last = 0.0
    for i in range(NB):
        sel = lum[band & (bidx == i)]
        if sel.size > 30:
            last = float(np.median(sel))
        prof[i] = last
    SW = 6
    fcorr = np.ones(NB)
    max_dev = 0.0
    for i in range(NB):
        wnd = prof[max(0, i - SW):min(NB, i + SW + 1)]
        ref = float(np.median(wnd))
        f = ref / prof[i] if prof[i] > 1e-5 else 1.0
        f = min(1.06, max(0.94, f))
        edge = min(1.0, min(i, NB - 1 - i) / 8)
        fcorr[i] = 1 + (f - 1) * edge
        max_dev = max(max_dev, abs(f - 1))
    if max_dev <= 0.01:
        return out
    factor = np.where(band, fcorr[bidx], 1.0)
    if patch.ndim == 3:
        out[by0:by1 + 1, bx0:bx1 + 1] = np.clip(patch * factor[..., None], 0, 1)
    else:
        out[by0:by1 + 1, bx0:bx1 + 1] = np.clip(patch * factor, 0, 1)
    log(f"Ring eraser: radial detrend (max symmetric deviation {100*max_dev:.1f}%)")
    return out


def _contrast_scurve(img: np.ndarray, amount: float) -> np.ndarray:
    """CurvesTransformation Akima S-curve through (0,0)(.25,·)(.80,·)(1,1)."""
    from scipy.interpolate import Akima1DInterpolator
    xs = np.array([0.0, 0.25, 0.80, 1.0])
    ys = np.array([0.0, max(0.0, 0.25 - amount * 0.25), min(1.0, 0.80 + amount * 0.20), 1.0])
    f = Akima1DInterpolator(xs, ys)
    return np.clip(f(np.clip(img, 0, 1)), 0.0, 1.0)


def _lhe(img: np.ndarray, amount: float, mask: Optional[np.ndarray]) -> np.ndarray:
    """LocalHistogramEqualization stand-in: masked unsharp local contrast on luminance."""
    a = np.asarray(img, dtype=np.float64)
    lum = a[..., :3].mean(axis=2) if a.ndim == 3 else a
    blur = gaussian_filter(lum, sigma=32.0, mode="nearest")
    boost = amount * (lum - blur)
    if mask is not None:
        boost = boost * mask
    if a.ndim == 3:
        out = a + boost[..., None]
    else:
        out = a + boost
    return np.clip(out, 0.0, 1.0)


def _golden_tone(img: np.ndarray) -> np.ndarray:
    """Golden duotone mapped from luminance (SUN.finishView PixelMath expressions)."""
    a = np.asarray(img, dtype=np.float64)
    L = a[..., :3].mean(axis=2)
    out = np.empty_like(a)
    out[..., 0] = np.minimum(1.0, np.power(L, 0.85) * 1.02)
    out[..., 1] = np.power(L, 1.15) * 0.93
    out[..., 2] = np.power(L, 1.90) * 0.72
    if a.shape[2] > 3:
        out[..., 3:] = a[..., 3:]
    return np.clip(out, 0.0, 1.0)


def _saturation(img: np.ndarray, amount: float) -> np.ndarray:
    """ColorSaturation stand-in: boost chroma about luminance by (1+amount)."""
    a = np.asarray(img, dtype=np.float64)
    L = a[..., :3].mean(axis=2)[..., None]
    out = a.copy()
    out[..., :3] = L + (a[..., :3] - L) * (1.0 + amount)
    return np.clip(out, 0.0, 1.0)
