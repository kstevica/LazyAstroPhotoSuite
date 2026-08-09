"""Sub-frame measurement + MAD cull (SubframeSelector + STK.cull equivalents).

Measures FWHM / eccentricity / star count / background / noise per frame via photutils, then
rejects with named evidence: hard eccentricity (trailed), soft FWHM (bloated) and background
surge in MADs, capped so no run culls more than ``max_reject_frac``. The surviving frames are
ranked to pick the registration reference (typical FWHM, dark sky, round, star-rich).
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np


def _noop(_m: str) -> None:
    pass


def _lum(img: np.ndarray) -> np.ndarray:
    return img[..., :3].mean(axis=2) if img.ndim == 3 else img


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12)


def measure_frame(img: np.ndarray) -> Optional[dict]:
    """Measure one frame; None if photutils is unavailable or no stars are found."""
    try:
        from astropy.stats import sigma_clipped_stats
        from photutils.detection import IRAFStarFinder
    except Exception:
        return None
    lum = _lum(np.asarray(img, dtype=np.float64))
    mean, med, std = sigma_clipped_stats(lum, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        return None
    try:
        sources = IRAFStarFinder(fwhm=3.0, threshold=max(5.0 * std, 1e-4))(lum - med)
    except Exception:
        sources = None
    if sources is None or len(sources) == 0:
        return {"stars": 0, "fwhm": np.nan, "ecc": np.nan, "bg": float(med),
                "noise": float(std), "snr": 0.0}
    fwhm = np.asarray(sources["fwhm"], dtype=np.float64)
    round1 = np.abs(np.asarray(sources["roundness"], dtype=np.float64))
    flux = np.asarray(sources["flux"], dtype=np.float64)
    return {
        "stars": int(len(sources)),
        "fwhm": float(np.median(fwhm)),
        "ecc": float(np.clip(np.median(round1), 0.0, 1.0)),   # roundness→eccentricity proxy
        "bg": float(med),
        "noise": float(std),
        "snr": float(np.median(flux) / std) if std > 0 else 0.0,
    }


def cull(measures: List[dict], params, *, log: Callable[[str], None] = _noop) -> dict:
    """Reject bad frames with named evidence; return keep indices + reference + reasons."""
    n = len(measures)
    idx = [i for i, m in enumerate(measures) if m and m["stars"] > 0]
    # frames the star finder couldn't measure (no round stars) are trailed / clouded: hard reject
    reasons: dict = {i: "no stars detected (trailed / clouded)"
                     for i in range(n) if i not in idx}
    if not idx:
        return {"keep": list(range(n)), "reference": 0, "rejected": reasons, "reasons": reasons}

    fwhm = np.array([measures[i]["fwhm"] for i in idx])
    ecc = np.array([measures[i]["ecc"] for i in idx])
    bg = np.array([measures[i]["bg"] for i in idx])
    fwhm_med, fwhm_mad = float(np.median(fwhm)), _mad(fwhm)
    bg_med, bg_mad = float(np.median(bg)), _mad(bg)

    soft: dict = {}
    for k, i in enumerate(idx):
        if ecc[k] > params.ecc_hard:
            soft[i] = f"trailed (ecc {ecc[k]:.2f} > {params.ecc_hard})"
        elif fwhm[k] > fwhm_med + params.fwhm_mads * fwhm_mad:
            soft[i] = f"bloated (FWHM {fwhm[k]:.2f} vs {fwhm_med:.2f})"
        elif bg[k] > bg_med + params.bg_mads * bg_mad:
            soft[i] = f"background surge (bg {bg[k]:.4f} vs {bg_med:.4f})"

    # cap over-culling of the SOFT rejects (zero-star frames are always out)
    max_rej = int(np.floor(params.max_reject_frac * len(idx)))
    if len(soft) > max_rej:
        ranked = sorted(soft.keys(), key=lambda i: (measures[i]["ecc"], measures[i]["fwhm"]))
        for i in ranked[max_rej:]:
            soft.pop(i)
        log(f"cull capped at {max_rej}/{len(idx)} soft rejects (max {params.max_reject_frac:.0%})")
    reasons.update(soft)

    keep = [i for i in idx if i not in soft]
    if not keep:
        keep = idx

    # reference: closest to the typical FWHM band, then dark sky + round
    def _ref_score(i):
        m = measures[i]
        return (abs(m["fwhm"] - fwhm_med), m["bg"], m["ecc"])
    reference = min(keep, key=_ref_score)
    log(f"kept {len(keep)}/{n}, rejected {len(reasons)}; reference frame index {reference} "
        f"(FWHM {measures[reference]['fwhm']:.2f}, bg {measures[reference]['bg']:.4f})")
    return {"keep": keep, "reference": reference, "rejected": reasons, "reasons": reasons,
            "fwhm_med": fwhm_med, "bg_med": bg_med}
