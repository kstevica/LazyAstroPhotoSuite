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


def _noop(_m: str) -> None:
    pass


def load_foreground(master_path: "str | Path"):
    """Load the nightscape foreground layer sitting beside ``master_path`` (or None)."""
    from pathlib import Path
    from ..io.image_io import load_image
    p = Path(master_path).with_name("lazystack_master_foreground.fits")
    try:
        return load_image(str(p)).data if p.exists() else None
    except Exception:
        return None


def load_sky_mask(master_path: "str | Path"):
    """Load the nightscape feathered sky mask (.npy) beside ``master_path`` (or None)."""
    from pathlib import Path
    p = Path(master_path).with_name("lazystack_master_skymask.npy")
    try:
        return np.load(str(p)) if p.exists() else None
    except Exception:
        return None


def _apply_transform_full(T, frame: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Apply an astroalign transform to a full (mono/RGB) frame; mark no-data border as NaN."""
    import astroalign
    if frame.ndim == 3:
        chans, footprint = [], None
        for c in range(frame.shape[2]):
            reg_c, footprint = astroalign.apply_transform(T, frame[..., c], ref[..., c])
            chans.append(reg_c)
        out = np.stack(chans, axis=-1)
        out[np.broadcast_to(footprint[..., None], out.shape)] = np.nan
    else:
        out, footprint = astroalign.apply_transform(T, frame, ref)
        out = np.asarray(out, dtype=np.float64)
        out[footprint] = np.nan
    return out


def register_sky(frames, reference: int, sky_mask: np.ndarray, *, log=_noop):
    """Register every frame to ``frames[reference]`` on SKY STARS ONLY.

    The foreground is replaced by the sky-median before source detection, so astroalign matches
    sky asterisms and is not confused by the bright static land (the cause of whole-frame
    registration failing on nightscapes). The transform is then applied to the FULL frame — the
    sky lines up (stars stack); the static foreground shifts by the small sky drift and is smeared
    in the resulting stack, but that region is discarded (the sharp foreground comes from one
    frame). Returns ``(aligned_frames, kept_indices)``.
    """
    import astroalign
    ref = np.asarray(frames[reference], dtype=np.float64)
    fg = np.asarray(sky_mask) < 0.5

    def masked_luma(im):
        L = _luminance(np.asarray(im, dtype=np.float64)).copy()
        L[fg] = float(np.median(L[~fg])) if np.any(~fg) else 0.0   # blank the land
        return L

    ref_ml = masked_luma(ref)
    aligned, kept = [], []
    for i, f in enumerate(frames):
        f = np.asarray(f, dtype=np.float64)
        if i == reference:
            aligned.append(f)
            kept.append(i)
            continue
        try:
            T, _ = astroalign.find_transform(masked_luma(f), ref_ml)
            aligned.append(_apply_transform_full(T, f, ref))
            kept.append(i)
        except Exception as e:                       # a frame that won't match is dropped, not fatal
            log(f"  nightscape: frame {i} sky-registration failed ({e}) — dropped")
    return aligned, kept


def pick_foreground(frames, fg_mask: np.ndarray):
    """Return ``(best_index, scores)`` — the frame whose foreground is sharpest (single frame)."""
    scores = [foreground_sharpness(f, fg_mask) for f in frames]
    return int(np.argmax(scores)), scores


def build_sky_master(frames, sky_mask: np.ndarray, reference: int, *, log=_noop):
    """Sky-only-register the frames to ``reference`` and sigma-clip integrate → deep sky master.

    Returns the linear deep-sky master in the reference's coordinates (sky sharp; the foreground
    region is smeared and meant to be replaced by the picked foreground frame at composite time).
    """
    from . import integrate as _integ
    aligned, kept = register_sky(frames, reference, sky_mask, log=log)
    log(f"  nightscape: sky-registered {len(kept)}/{len(frames)} frames; integrating deep sky…")
    master = _integ.integrate(aligned)
    return np.clip(np.nan_to_num(master, nan=0.0), 0.0, 1.0), kept


def _resize_mask(mask: np.ndarray, shape) -> np.ndarray:
    """Bilinear-resize a (feathered) mask to ``shape`` exactly."""
    from scipy.ndimage import zoom
    out = zoom(np.asarray(mask, dtype=np.float64),
               (shape[0] / mask.shape[0], shape[1] / mask.shape[1]), order=1)
    out = out[:shape[0], :shape[1]]
    if out.shape != tuple(shape[:2]):
        out = np.pad(out, [(0, shape[0] - out.shape[0]), (0, shape[1] - out.shape[1])], mode="edge")
    return np.clip(out, 0.0, 1.0)


def match_shape(arr: np.ndarray, shape) -> np.ndarray:
    """Resize a mono/RGB array to ``shape`` (H, W) — used to align a Mode-2 foreground/mask."""
    a = np.asarray(arr, dtype=np.float64)
    if a.shape[:2] == tuple(shape[:2]):
        return a
    if a.ndim == 3:
        return np.stack([_resize_mask(a[..., c], shape) for c in range(a.shape[2])], axis=-1)
    return _resize_mask(a, shape)


def plan(small_cube: np.ndarray, full_shape, *, user_fg_small: Optional[np.ndarray] = None,
         bias: float = 0.0) -> Tuple[Optional[int], np.ndarray, dict]:
    """Plan a nightscape stack from a downsampled cube (bounded memory; the caller has the full res).

    Returns ``(best_local_index, sky_mask_full, info)``. In Mode 1 (auto) the mask comes from the
    stack and ``best_local_index`` is the sharpest-foreground frame (index into the cube). In Mode 2
    the mask comes from ``user_fg_small`` (the user's foreground, downsampled) and the index is None
    (the caller supplies the foreground). The mask is upsampled to ``full_shape`` for registration.
    """
    if user_fg_small is not None:
        sky_small, info = segment_sky(user_fg_small, bias=bias)
        best = None
    else:
        sky_small, info = segment_sky(small_cube, bias=bias)
        best, _ = pick_foreground([small_cube[j] for j in range(len(small_cube))], 1.0 - sky_small)
    return best, _resize_mask(sky_small, full_shape), info


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
