"""Transient (meteor) detection on the integration transient map — Perseid preservation.

The sigma-clip integrator already *finds* meteors: a meteor is the high-side outlier it rejects
(present in one frame, absent from its neighbours). :func:`integrate.sigma_clip_mean` with
``return_transient`` hands back that discarded light — the brightest high-side rejected excess
``(frame − master)`` in linear RGB, plus the source frame per pixel. Here we pick out the ones
that look like meteors: **elongated**, **single-frame**, not-too-small / not-too-huge connected
components. Compact residuals (cosmic rays, star-seeing outliers) fail the aspect/length test;
diffuse regions fail the area cap. Satellites/aircraft (also single-frame streaks) are left for
the review step / a later multi-frame classifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

_LAYER_NAMES = ("lazystack_master_meteors.npy",)


def _noop(_m: str) -> None:
    pass


def load_meteor_layer(master_path: "str | Path") -> Optional[np.ndarray]:
    """Find + load the meteor layer written beside ``master_path`` (or ``None``)."""
    p = Path(master_path)
    candidates = [p.with_name(n) for n in _LAYER_NAMES]
    candidates.append(p.with_name(p.stem + "_meteors.npy"))
    for c in candidates:
        if c.exists():
            try:
                return np.load(str(c))
            except Exception:
                pass
    return None


def develop_meteor(layer: np.ndarray, strength: float = 1.0, *, k: float = 12.0) -> np.ndarray:
    """Gently, HUE-PRESERVINGLY develop the linear meteor layer for compositing.

    A meteor is already bright; the master's deep-sky MTF would blow it to white and kill its
    colour. Instead apply an asinh to the LUMINANCE and rescale each channel by the same factor —
    brightness comes up, the R/G/B ratios (the real ablation colours) are preserved. Scaled by
    ``strength``. Returns an additive layer to add onto the developed master (then clip).
    """
    a = np.asarray(layer, dtype=np.float64)
    lum = a[..., :3].mean(axis=2) if a.ndim == 3 else a
    f = np.arcsinh(lum * k) / np.arcsinh(k)                     # asinh luminance stretch
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(lum > 1e-4, f / np.where(lum > 1e-4, lum, 1.0), 0.0)
    dev = a * scale[..., None] if a.ndim == 3 else a * scale
    return float(np.clip(strength, 0.0, 1.0)) * dev


def _pca_shape(ys: np.ndarray, xs: np.ndarray) -> Tuple[float, float]:
    """Return (aspect = major/minor, length ≈ 2·major-sigma) of a pixel blob via PCA."""
    if xs.size < 3:
        return 1.0, 0.0
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], axis=0).astype(np.float64)
    cov = pts @ pts.T / xs.size
    ev = np.linalg.eigvalsh(cov)                      # ascending [minor, major]
    minor = float(np.sqrt(max(ev[0], 1e-9)))
    major = float(np.sqrt(max(ev[1], 1e-9)))
    return major / max(minor, 1e-6), 4.0 * major      # ~full extent (±2σ)


def detect_meteors(transient: np.ndarray, frame_idx: np.ndarray, *,
                   thr_floor: float = 0.02, min_area: int = 20, max_area_frac: float = 0.02,
                   min_aspect: float = 3.0, min_length: float = 30.0,
                   min_single_frame: float = 0.6, feather: float = 2.0,
                   log: Callable[[str], None] = _noop) -> Tuple[List[dict], np.ndarray]:
    """Detect meteor trails in the transient map. Returns (meteor dicts, feathered soft mask)."""
    from scipy.ndimage import binary_dilation, gaussian_filter, label
    a = np.asarray(transient, dtype=np.float64)
    lum = a[..., :3].mean(axis=2) if a.ndim == 3 else a
    H, W = lum.shape
    mad = 1.4826 * float(np.median(np.abs(lum - np.median(lum))))
    thr = max(thr_floor, 8.0 * mad)                  # transient is ~0 off-trail, so the floor rules
    lbl, nlab = label(lum > thr, structure=np.ones((3, 3)))
    max_area = int(max_area_frac * H * W)
    meteors: List[dict] = []
    soft = np.zeros((H, W), dtype=np.float64)
    for i in range(1, nlab + 1):
        ys, xs = np.where(lbl == i)
        area = xs.size
        if area < min_area or area > max_area:
            continue
        aspect, length = _pca_shape(ys, xs)
        if aspect < min_aspect or length < min_length:   # compact residual / too short
            continue
        fr = frame_idx[ys, xs]
        fr = fr[fr >= 0]
        if fr.size == 0:
            continue
        vals, counts = np.unique(fr, return_counts=True)
        dom_frac = float(counts.max()) / area
        if dom_frac < min_single_frame:                  # must be one frame (a real transient)
            continue
        meteors.append({"frame": int(vals[np.argmax(counts)]), "area": int(area),
                        "aspect": round(aspect, 2), "length": round(length, 1),
                        "single_frame_frac": round(dom_frac, 2), "peak": float(lum[ys, xs].max()),
                        "bbox": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]})
        soft[ys, xs] = 1.0
    if meteors:
        soft = binary_dilation(soft > 0, iterations=3).astype(np.float64)   # include the glow
        soft = np.clip(gaussian_filter(soft, feather), 0.0, 1.0)            # soft edge, no hard line
        log(f"Meteor detection: {len(meteors)} trail(s) preserved.")
    else:
        log("Meteor detection: no meteor trails found.")
    return meteors, soft


def meteor_layer(transient: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    """The feathered linear-RGB meteor contribution to composite (transient × soft mask)."""
    a = np.asarray(transient, dtype=np.float32)
    m = np.asarray(soft_mask, dtype=np.float32)
    return a * (m[..., None] if a.ndim == 3 else m)
