"""Nightscape (foreground-locked Milky Way) support — sky/foreground segmentation + scoring.

A fixed-tripod nightscape holds two subjects with different needs: the SKY (deep-stack the
Milky Way, aligned on stars) and the FOREGROUND (a static, often moonlit landscape kept sharp
from a single frame). Stacking them as one fails — the sky drifts relative to the static land
(two rigid motions), so a single alignment can't serve both.

The first job is to split the frame into sky vs foreground. On easy data (dark sky, long drift)
motion cues work, but on moonlit / low-drift data they collapse (the bright textured land mimics
stars; the stars barely move). The robust, scene-general cue is the **horizon**: a near-continuous
high-gradient edge between the smooth sky and the structured land. :func:`segment_sky` detects it
per-scanline (so it follows an undulating ridge and vertical intrusions — a peak, a tree, a tower —
without assuming a straight or horizontal horizon), picks the sky side automatically, and exposes a
``bias`` knob so the result is refinable. The mask is feathered for compositing.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, sobel


def _luminance(a: np.ndarray) -> np.ndarray:
    return a[..., :3].mean(axis=2) if a.ndim == 3 else a


def _median_luma(stack: np.ndarray) -> np.ndarray:
    """Median luminance frame from a cube ``(N, H, W[, 3])`` or a single image."""
    a = np.asarray(stack, dtype=np.float64)
    if a.ndim == 4:
        return np.median(_luminance(a.reshape(-1, *a.shape[2:])).reshape(a.shape[0], *a.shape[1:3]), axis=0) \
            if a.shape[-1] == 3 else np.median(a, axis=0)
    if a.ndim == 3 and a.shape[-1] == 3:
        return _luminance(a)                     # single RGB image
    if a.ndim == 3:
        return np.median(a, axis=0)              # mono cube
    return a                                     # single mono image


def _structure(logL: np.ndarray, scale: float) -> np.ndarray:
    """Large-scale edge energy of a log-luminance image (the land is structured, the sky smooth)."""
    s = gaussian_filter(logL, scale)
    return np.hypot(sobel(s, axis=0), sobel(s, axis=1))


def segment_sky(stack: np.ndarray, *, bias: float = 0.0, feather: Optional[float] = None,
                edge_scale: float = 4.0, smooth_frac: float = 0.02
                ) -> Tuple[np.ndarray, dict]:
    """Segment sky (1) vs foreground (0) from a nightscape stack or a single image.

    Returns ``(sky_mask, info)`` where ``sky_mask`` is a feathered float array in [0, 1] (1 = sky)
    and ``info`` carries the detected axis/side, the horizon curve, and its strength.

    Robust to orientation: the split axis and which side is sky are chosen from the large-scale
    edge-energy asymmetry across the borders (the land is structured and touches one side; the sky
    is smooth). The horizon is then found per-scanline as the strongest continuous edge, so it
    tracks an undulating ridge and vertical intrusions. ``bias`` (−1..1) shifts the boundary toward
    the foreground (>0, more sky) or the sky (<0, more foreground) for refinement.
    """
    L = np.clip(_median_luma(stack), 0.0, 1.0)
    H, W = L.shape
    logL = np.log1p(L * 50.0)                     # compress the dark-sky/bright-land dynamic range
    struct = _structure(logL, edge_scale)

    # --- pick the split axis + sky side from where the strong edges concentrate ---
    # The land is a large structured region off to one side; the structure-weighted centroid points
    # at it. (Thin border strips miss it — the ridge/moonglow peaks mid-frame, not at the edge.)
    st = np.clip(struct - np.median(struct), 0.0, None)   # strong edges only
    tot = st.sum() + 1e-12
    cx = float((st.sum(axis=0) * np.arange(W)).sum() / tot)
    cy = float((st.sum(axis=1) * np.arange(H)).sum() / tot)
    off_x, off_y = abs(cx - W / 2) / W, abs(cy - H / 2) / H
    vertical = off_x >= off_y                        # near-vertical horizon → scan per row
    if vertical:
        fg_on_high = cx > W / 2                       # foreground toward higher column
        axis, sky_side = "vertical", ("left" if fg_on_high else "right")
    else:
        fg_on_high = cy > H / 2
        axis, sky_side = "horizontal", ("top" if fg_on_high else "bottom")

    # orient so we scan rows and the foreground is toward higher column index
    Lw = logL if vertical else logL.T
    Sw = struct if vertical else struct.T
    Hs, Ws = Lw.shape
    if not fg_on_high:                                # flip so foreground is at the high-index end
        Lw, Sw = Lw[:, ::-1], Sw[:, ::-1]

    # --- per-scanline horizon: strongest edge, made continuous across scanlines ---
    band = gaussian_filter(Sw, (max(3, int(0.012 * Hs)), 2))
    col = np.argmax(band, axis=1).astype(np.float64)
    strength = band.max(axis=1)
    thr = np.percentile(strength, 25)
    idx = np.arange(Hs)
    good = strength >= thr
    if good.sum() >= 2:
        col = np.interp(idx, idx[good], col[good])
    sm = max(3, int(smooth_frac * Hs) | 1)           # odd window
    col = median_filter(col, sm)
    col = np.clip(col + bias * Ws, 0, Ws - 1)         # refinement shift

    # foreground = at/beyond the horizon column; feather the seam
    xx = np.arange(Ws)[None, :]
    signed = (xx - col[:, None])                      # <0 sky, >0 foreground
    fp = feather if feather is not None else max(2.0, 0.01 * Ws)
    t = np.clip(0.5 - signed / (2.0 * fp), 0.0, 1.0)  # 1 sky → 0 foreground over ~2·fp
    sky_w = t * t * (3.0 - 2.0 * t)

    if not fg_on_high:
        sky_w = sky_w[:, ::-1]
    sky_mask = sky_w if vertical else sky_w.T

    info = {"axis": axis, "sky_side": sky_side, "horizon_strength": float(np.median(strength)),
            "foreground_fraction": float(1.0 - sky_mask.mean())}
    return np.clip(sky_mask, 0.0, 1.0), info


def foreground_sharpness(frame: np.ndarray, fg_mask: np.ndarray) -> float:
    """Focus score of a frame's foreground region (Laplacian energy) — higher = sharper.

    Used to auto-pick the sharpest foreground frame; a single frame is kept (never averaged) so a
    shimmering moonlit sea or moving water isn't smeared.
    """
    L = _luminance(np.asarray(frame, dtype=np.float64))
    lap = (-4.0 * L + np.roll(L, 1, 0) + np.roll(L, -1, 0) + np.roll(L, 1, 1) + np.roll(L, -1, 1))
    w = np.asarray(fg_mask, dtype=np.float64)
    denom = float(w.sum()) + 1e-9
    return float((lap * lap * w).sum() / denom)
