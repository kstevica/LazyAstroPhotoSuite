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
import json
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

_LAYER_NAMES = ("lazystack_master_meteors.npy",)
_LABELS_NAMES = ("lazystack_master_meteorlabels.npy",)
_META_NAMES = ("lazystack_master_meteors.json",)


def _noop(_m: str) -> None:
    pass


def _beside(master_path: "str | Path", names: Sequence[str], stem_suffix: str) -> Optional["Path"]:
    p = Path(master_path)
    for c in [p.with_name(n) for n in names] + [p.with_name(p.stem + stem_suffix)]:
        if c.exists():
            return c
    return None


def load_meteor_layer(master_path: "str | Path") -> Optional[np.ndarray]:
    """Find + load the meteor layer written beside ``master_path`` (or ``None``)."""
    c = _beside(master_path, _LAYER_NAMES, "_meteors.npy")
    try:
        return np.load(str(c)) if c else None
    except Exception:
        return None


def load_meteor_labels(master_path: "str | Path") -> Optional[np.ndarray]:
    """Find + load the per-meteor id map beside ``master_path`` (or ``None``)."""
    c = _beside(master_path, _LABELS_NAMES, "_meteorlabels.npy")
    try:
        return np.load(str(c)) if c else None
    except Exception:
        return None


def load_meteor_meta(master_path: "str | Path") -> List[dict]:
    """Load the detected-meteor metadata list beside ``master_path`` (``[]`` if none)."""
    c = _beside(master_path, _META_NAMES, "_meteors.json")
    try:
        return json.loads(c.read_text(encoding="utf-8")).get("meteors", []) if c else []
    except Exception:
        return []


def selection_layer(layer: np.ndarray, labels: Optional[np.ndarray],
                    select: Optional[Sequence[int]]) -> np.ndarray:
    """Mask the meteor layer to the selected trail ids (``None`` → all trails)."""
    if select is None or labels is None:
        return layer
    keep = np.isin(np.asarray(labels), list(select))
    a = np.asarray(layer)
    return a * (keep[..., None] if a.ndim == 3 else keep)


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
                   thr_floor: float = 0.006, min_area: int = 20, max_area_frac: float = 0.02,
                   min_aspect: float = 4.0, min_length: float = 40.0, connect: int = 2,
                   min_seg: int = 6, min_single_frame: float = 0.6, feather: float = 2.0,
                   log: Callable[[str], None] = _noop) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """Detect meteor trails in the transient map. Returns (meteor dicts, feathered soft mask,
    per-meteor label map — each trail's pixels carry its 1-based ``id`` for later selection).

    Tuned on a real faint Milky Way meteor (transient excess ~0.005-0.08). Pipeline: threshold at
    ``thr_floor`` → drop tiny (< ``min_seg``) noise specks BEFORE bridging (else dilation merges
    scattered noise into one huge blob) → dilate by ``connect`` px so trail brightness dips don't
    fragment it → label → keep elongated, single-frame, right-sized components.
    """
    from scipy.ndimage import binary_dilation, find_objects, gaussian_filter, grey_dilation, label
    a = np.asarray(transient, dtype=np.float64)
    lum = a[..., :3].mean(axis=2) if a.ndim == 3 else a
    H, W = lum.shape
    mad = 1.4826 * float(np.median(np.abs(lum - np.median(lum))))
    thr = max(thr_floor, 8.0 * mad)                  # transient is ~0 off-trail, so the floor rules
    lbl0, n0 = label(lum > thr, structure=np.ones((3, 3)))
    if n0 == 0:
        log("Meteor detection: no meteor trails found.")
        return [], np.zeros((H, W), dtype=np.float64), np.zeros((H, W), dtype=np.int16)
    keep0 = np.bincount(lbl0.ravel()) >= min_seg     # noise specks are 1-3 px; drop them first
    keep0[0] = False
    binary = keep0[lbl0]
    if connect:
        binary = binary_dilation(binary, iterations=int(connect))   # then bridge trail dips
    lbl, nlab = label(binary, structure=np.ones((3, 3)))
    meteors: List[dict] = []
    soft = np.zeros((H, W), dtype=np.float64)
    labels = np.zeros((H, W), dtype=np.int16)            # per-meteor id map (for selection)
    if nlab:
        # One O(N) pass for all component sizes, then only inspect the size-candidates via their
        # bounding boxes — never `np.where(lbl==i)` over the whole frame per label (that was
        # O(nlab·N): 40 MP × tens-of-thousands of noise specks = tens of minutes).
        sizes = np.bincount(lbl.ravel())
        max_area = int(max_area_frac * H * W)
        cand = np.where((sizes >= min_area) & (sizes <= max_area))[0]
        cand = cand[cand != 0]
        boxes = find_objects(lbl)
        for lab in cand:
            sl = boxes[lab - 1]
            if sl is None:
                continue
            sub = lbl[sl] == lab
            yy, xx = np.where(sub)
            ys, xs = yy + sl[0].start, xx + sl[1].start
            aspect, length = _pca_shape(ys, xs)
            if aspect < min_aspect or length < min_length:   # compact residual / too short
                continue
            fr = frame_idx[ys, xs]
            fr = fr[fr >= 0]
            if fr.size == 0:
                continue
            vals, counts = np.unique(fr, return_counts=True)
            # Fraction of the *transient* pixels from one frame (a real meteor is single-frame; a
            # chance-elongated noise cluster draws from many frames). Denominator excludes the
            # no-transient border pixels the dilation added.
            dom_frac = float(counts.max()) / fr.size
            if dom_frac < min_single_frame:
                continue
            mid = len(meteors) + 1                       # 1-based meteor id
            meteors.append({"id": mid, "frame": int(vals[np.argmax(counts)]), "area": int(xs.size),
                            "aspect": round(aspect, 2), "length": round(length, 1),
                            "single_frame_frac": round(dom_frac, 2), "peak": float(lum[ys, xs].max()),
                            "bbox": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]})
            soft[ys, xs] = 1.0
            labels[ys, xs] = mid
    if meteors:
        # dilate the id map to cover the feathered glow, so selection masks match the composite
        labels = grey_dilation(labels, size=7)
        soft = binary_dilation(soft > 0, iterations=3).astype(np.float64)   # include the glow
        soft = np.clip(gaussian_filter(soft, feather), 0.0, 1.0)            # soft edge, no hard line
        log(f"Meteor detection: {len(meteors)} trail(s) preserved.")
    else:
        log("Meteor detection: no meteor trails found.")
    return meteors, soft, labels


def meteor_layer(transient: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    """The feathered linear-RGB meteor contribution to composite (transient × soft mask)."""
    a = np.asarray(transient, dtype=np.float32)
    m = np.asarray(soft_mask, dtype=np.float32)
    return a * (m[..., None] if a.ndim == 3 else m)
