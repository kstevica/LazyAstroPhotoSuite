"""Experimental "Amplified signal" stacking — physics that keeps more of the photons.

No pixels are invented. Five measured mechanisms, all evidence-logged:

1. **Soft frames kept** — cull no longer discards bloated / bright-sky frames; they join
   the stack with frequency-split weights (their low-frequency signal is as good as the
   best frame's; only their high band is down-weighted by their measured MTF).
2. **Inverse-variance weighting** — every frame weighted 1/σ² from its measured noise
   floor, the maximum-likelihood combination for frames of unequal quality.
3. **Photon-transfer noise model** — the Poisson+Gaussian relation var = a·signal + b is
   regressed from the registered frames themselves (no calibration data needed) and used
   as a physical floor for the clip scale + reported as evidence.
4. **Dither-validated static-pattern removal** — structure that stays fixed in sensor
   coordinates while the sky moves with the dither is pattern, not photons; a temporal
   median (star-safe via a temporal-MAD guard) is soft-thresholded and subtracted.
5. **Fine-grid registration (drizzle-lite)** — undersampled, dithered sets are warped
   once, directly onto a 2× grid, recovering sub-pixel detail the dither actually sampled.

The calibrated variance sidecar (existing noise-map format) plus ``amplified_meta.json``
carry the receipts for every decision.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from ..moonsun import register as fftreg


def _noop(_m: str) -> None:
    pass


def _lum(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    return a.mean(axis=2) if a.ndim == 3 else a


# --------------------------------------------------------------------------- dither

def measure_dither(get: Callable, frames: Sequence, keep: Sequence[int], ref_idx: int,
                   *, max_probe: int = 9, log: Callable[[str], None] = _noop) -> float:
    """Median |shift| (full-res px) of a probe subset vs the reference, via FFT phase
    correlation. Cheap, translation-only — dither is translation to first order."""
    others = [i for i in keep if i != ref_idx]
    if not others:
        return 0.0
    probe = others[:: max(1, len(others) // max_probe)][:max_probe]
    try:
        ref_wk, k = fftreg.working(_lum(get(frames[ref_idx])))
        ref_fft = fftreg.make_ref(fftreg.gradient(ref_wk), ref_wk.shape[1], ref_wk.shape[0])
        mags = []
        for i in probe:
            wk, kk = fftreg.working(_lum(get(frames[i])))
            dx, dy = fftreg.measure_against(ref_fft, fftreg.gradient(wk),
                                           wk.shape[1], wk.shape[0])
            mags.append(float(np.hypot(dx / kk, dy / kk)))
        d = float(np.median(mags)) if mags else 0.0
        log(f"  dither: median offset {d:.1f} px over {len(mags)} probe frame(s)")
        return d
    except Exception as e:
        log(f"  dither probe failed ({e}) — assuming none")
        return 0.0


# ------------------------------------------------------------------- static pattern

def static_pattern(get: Callable, frames: Sequence, keep: Sequence[int], *,
                   fwhm_px: float, dither_px: float,
                   log: Callable[[str], None] = _noop) -> Optional[np.ndarray]:
    """Dither-validated static-pattern estimate in SENSOR space (or None).

    The lever is geometric: the sky moves by the dither between frames, the sensor's
    residual pattern does not. The pattern is band-limited to spatial scales the dither
    can PROVE are static (highpass σ = dither/3): structure smaller than the dither is
    displaced by more than its own size between frames, so the temporal median suppresses
    it if it is sky — anything larger is not separable and is left alone (this is what
    keeps extended nebulosity, which barely moves relative to its own width, safe).
    Star-safety guard: pattern pixels have temporal spread ≈ noise; star-crossed pixels
    have a large temporal MAD and are excluded. The pattern is soft-thresholded at 1σ of
    its own distribution so weak "pattern" (i.e. noise) injects nothing.
    """
    need = max(2.0 * fwhm_px, 3.0)
    smooth_sigma = dither_px / 3.0                          # provable-scale bound
    if dither_px < need or smooth_sigma < 1.2:
        log(f"  static-pattern removal skipped: dither {dither_px:.1f} px "
            f"(needs ≥ {max(need, 3.6):.1f} px — sky and sensor are not separable)")
        return None
    if len(keep) < 5:
        log("  static-pattern removal skipped: needs ≥5 frames")
        return None
    # Banded streaming (mmap where staged): temporal median + MAD per row band, never the
    # full N×H×W cube — same memory discipline as combine_files.
    def _open(h):
        return np.load(str(h), mmap_mode="r") if isinstance(h, (str, Path)) else np.asarray(h)
    handles = [_open(frames[i]) for i in keep]
    H, W = handles[0].shape[:2]
    band = max(1, int(120_000_000 / (len(handles) * W * 8)))
    med = np.empty((H, W), dtype=np.float64)
    tmad = np.empty((H, W), dtype=np.float64)
    for y0 in range(0, H, band):
        y1 = min(H, y0 + band)
        chunk = np.stack([_lum(np.asarray(h[y0:y1], dtype=np.float64)) for h in handles], axis=0)
        med[y0:y1] = np.median(chunk, axis=0)
        tmad[y0:y1] = np.median(np.abs(chunk - med[y0:y1]), axis=0)
    del handles
    smooth = gaussian_filter(med, sigma=smooth_sigma)
    p = med - smooth                                        # static structure ≲ dither scale only
    guard = tmad < 2.5 * (np.median(tmad) + 1e-12)          # star-crossed pixels excluded
    sig = 1.4826 * np.median(np.abs(p[guard] - np.median(p[guard]))) + 1e-12
    p = np.sign(p) * np.maximum(np.abs(p) - sig, 0.0)       # soft threshold: noise adds nothing
    p = np.where(guard, p, 0.0)
    rms = float(np.sqrt(np.mean(p**2)))
    if rms < 0.25 * sig:
        log("  static-pattern removal: no significant fixed pattern found")
        return None
    log(f"  static pattern: RMS {rms:.2e} (threshold σ {sig:.2e}), "
        f"{100.0 * np.mean(p != 0):.1f}% of pixels touched")
    return p.astype(np.float32)


def subtract_pattern(img: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Subtract the (luminance) pattern from every channel, clipped to valid range."""
    a = np.asarray(img, dtype=np.float32)
    p = pattern if a.ndim == 2 else pattern[..., None]
    return np.clip(a - p, 0.0, 1.0)


# ------------------------------------------------------------------ photon transfer

def photon_transfer(handles: Sequence, *, sample_rows: int = 192,
                    log: Callable[[str], None] = _noop) -> Optional[dict]:
    """Regress the Poisson+Gaussian noise relation var = a·signal + b from the
    registered frames themselves (per-pixel temporal statistics, quantile-binned).

    Returns {"a", "b", "gain_full_scale"} in normalized [0,1] units, or None if the
    relation is not measurable (too few frames, flat signal range, negative slope).
    """
    n = len(handles)
    if n < 5:
        return None
    first = np.load(str(handles[0]), mmap_mode="r") if isinstance(handles[0], (str, Path)) \
        else np.asarray(handles[0])
    H = first.shape[0]
    rows = np.linspace(0, H - 1, num=min(sample_rows, H)).astype(int)
    def _rows(h):
        a = np.load(str(h), mmap_mode="r") if isinstance(h, (str, Path)) else np.asarray(h)
        return _lum(np.asarray(a[rows], dtype=np.float64))
    cube = np.stack([_rows(h) for h in handles], axis=0)     # (N, R, W)
    valid = np.isfinite(cube)
    cnt = valid.sum(axis=0)
    ok = cnt >= max(5, n // 2)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(np.where(valid, cube, np.nan), axis=0)
        v = np.nanvar(np.where(valid, cube, np.nan), axis=0, ddof=1)
    m, v = m[ok].ravel(), v[ok].ravel()
    if m.size < 10_000:
        return None
    # LOWER-ENVELOPE fit on the sky-and-faint regime only. Bright pixels (star halos,
    # nebula edges) carry registration-jitter and seeing-variation variance that dwarfs
    # photon noise — the very frames amplified mode keeps make that worse — so: restrict
    # to signal ≤ the 80th percentile, and take each bin's 25th-percentile variance
    # (contamination is strictly additive), de-biased by the χ² quantile so the envelope
    # is unbiased under pure photon statistics.
    from scipy.stats import chi2
    cap = float(np.quantile(m, 0.80))
    m_cap = float(np.quantile(m, 0.85))                    # floor never applies above this
    sel_all = m <= cap
    m2, v2 = m[sel_all], v[sel_all]
    dof = max(int(np.median(cnt[ok])) - 1, 4)
    debias = chi2.ppf(0.25, dof) / dof
    qs = np.quantile(m2, np.linspace(0.02, 0.99, 13))
    centers, envs, counts = [], [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (m2 >= lo) & (m2 < hi)
        if sel.sum() < 500:
            continue
        centers.append(float(np.median(m2[sel])))
        envs.append(float(np.percentile(v2[sel], 25)) / debias)
        counts.append(int(sel.sum()))
    if len(centers) < 4 or (max(centers) - min(centers)) < 1e-4:
        log("  photon transfer: signal range too flat to regress — model skipped")
        return None
    w = np.sqrt(np.asarray(counts, dtype=np.float64))
    c_arr, e_arr = np.asarray(centers), np.asarray(envs)
    a, b = np.polyfit(c_arr, e_arr, 1, w=w)
    # Second pass: bins whose envelope still sits far above the line are wholesale-
    # contaminated (dense halo zones) — drop them and refit on the photon-clean bins.
    pred = a * c_arr + max(b, 0.0)
    ok_bins = e_arr <= 2.5 * np.maximum(pred, 1e-12)
    if ok_bins.sum() >= 4 and not ok_bins.all():
        a, b = np.polyfit(c_arr[ok_bins], e_arr[ok_bins], 1, w=w[ok_bins])
        log(f"  photon transfer: dropped {int((~ok_bins).sum())} contaminated bin(s)")
    if a <= 0:
        log("  photon transfer: non-physical slope — model skipped")
        return None
    b = max(float(b), 0.0)
    gain = 1.0 / float(a)
    log(f"  photon transfer: var = {a:.3e}·signal + {b:.3e}  "
        f"(≈{gain:,.0f} e⁻ full-scale, resampled-grid estimate; floor capped at "
        f"signal {m_cap:.3f})")
    return {"a": float(a), "b": b, "m_cap": m_cap, "gain_full_scale": gain}


# -------------------------------------------------------------------- band weights

def band_weights(measures: Sequence[Optional[dict]], order: Sequence[int],
                 *, ref_idx: Optional[int] = None,
                 log: Callable[[str], None] = _noop) -> dict:
    """Per-frame weights for the two frequency bands, in ``order``.

    Low band: pure inverse variance from each frame's noise floor AFTER the normalize
    step's per-frame rescale (normalize_to_ref multiplies by ≈clip(σ_ref/σ_i, 0.5, 2), so
    the noise actually integrated is σ_i·scale_i — weighting the raw σ_i would misweight
    exactly the frames the mode exists to save). High band: the same, times the frame's
    MTF at the seeing cutoff — a Gaussian PSF of FWHM f has MTF ∝ exp(−0.89·(f/f_best)²)
    at the best frame's resolution scale, so soft frames feed the faint stuff and stay
    out of the detail.
    """
    noise = np.array([(measures[i] or {}).get("noise", np.nan) for i in order], dtype=float)
    fwhm = np.array([(measures[i] or {}).get("fwhm", np.nan) for i in order], dtype=float)
    med_n = np.nanmedian(noise) if np.isfinite(noise).any() else 1.0
    noise = np.where(np.isfinite(noise) & (noise > 0), noise, med_n)
    n_ref = (measures[ref_idx] or {}).get("noise") if ref_idx is not None else None
    if n_ref and np.isfinite(n_ref) and n_ref > 0:
        noise = noise * np.clip(float(n_ref) / noise, 0.5, 2.0)   # post-normalization σ
    w_lo = 1.0 / np.maximum(noise, 1e-6) ** 2
    w_lo /= w_lo.mean()
    fin = fwhm[np.isfinite(fwhm)]
    f_best = float(np.percentile(fin, 10)) if fin.size else 1.0
    fwhm = np.where(np.isfinite(fwhm) & (fwhm > 0), fwhm, f_best)
    mtf = np.exp(-0.89 * ((fwhm / max(f_best, 1e-6)) ** 2 - 1.0))
    w_hi = w_lo * np.clip(mtf, 1e-4, 1.0)
    w_hi /= w_hi.mean()
    log(f"  weights: inverse-variance (spread {w_lo.min():.2f}–{w_lo.max():.2f}); "
        f"high-band MTF vs best FWHM {f_best:.2f}px "
        f"(soft-frame factor down to {np.clip(mtf, 1e-4, 1.0).min():.2f})")
    return {"w_lo": w_lo.tolist(), "w_hi": w_hi.tolist(), "fwhm_best": f_best,
            "fwhm": fwhm.tolist()}


# ---------------------------------------------------------------------- band merge

def band_merge(m_lo: np.ndarray, m_hi: np.ndarray, *, sigma_px: float) -> np.ndarray:
    """Master = lowpass(all-frame mean) + highpass(sharp-weighted mean).

    Both means share the same survivors (one rejection pass); only the frame weights
    differ. The split at the seeing scale hands each band to the mean that is optimal
    for it: every photon deepens the faint structure, only sharp photons draw the detail.
    """
    lo = np.asarray(m_lo, dtype=np.float32)
    hi = np.asarray(m_hi, dtype=np.float32)
    out = np.empty_like(lo)                                 # float32, channel-wise: the 2×
    if lo.ndim == 3:                                        # grid makes float64 whole-frame
        for c in range(lo.shape[2]):                        # temporaries a multi-GB spike
            out[..., c] = gaussian_filter(lo[..., c], sigma_px)
            out[..., c] += hi[..., c] - gaussian_filter(hi[..., c], sigma_px)
    else:
        out[:] = gaussian_filter(lo, sigma_px) + hi - gaussian_filter(hi, sigma_px)
    return np.clip(out, 0.0, 1.0, out=out)


# --------------------------------------------------------------------------- meta

def write_meta(out_dir: Path, meta: dict, log: Callable[[str], None] = _noop) -> None:
    try:
        (Path(out_dir) / "amplified_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        log(f"  amplified receipts: {Path(out_dir) / 'amplified_meta.json'}")
    except Exception as e:                                   # receipts, never fatal
        log(f"  (amplified meta skipped: {e})")
