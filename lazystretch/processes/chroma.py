"""Chromatic-aberration correction — align the R and G channels to B.

Lateral chromatic aberration (and OSC channel misregistration) leaves the R/G/B images of
each star slightly offset — often ~1-2 px, growing radially toward the frame edge. On its
own it's a faint colour fringe, but sharp per-channel processing (deconvolution, an
aggressive stretch) tightens each channel into a hard star and turns the offset into a
visible **dark crescent** between the channels.

This module measures the offset from stars themselves and removes it:

1. Detect bright, isolated, non-saturated stars across the frame.
2. Per star, measure the flux-weighted centroid of R, G, B in a small window and take the
   R→B and G→B offset vectors.
3. Fit a smooth, low-order 2-D polynomial offset field per channel (robust to outliers) —
   lateral CA is dominated by a linear/quadratic radial term, which the polynomial captures.
4. Warp R and G by the modelled per-pixel offset (× ``strength``) so their stars land on B's.

Reference channel is B: red is the most-refracted (largest lateral CA) and blue the least in
most fast optics, so aligning R and G *to* B is the stable choice. Mono images and images
with too few measurable stars are returned unchanged.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter


def _noop(_m: str) -> None:
    pass


def _lum(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=2) if a.ndim == 3 else a


def _star_centroids(a: np.ndarray, *, max_stars: int, win: int,
                    sat: float) -> list:
    """Find bright, isolated, non-saturated stars and return their (y, x) integer peaks."""
    L = _lum(a)
    H, W = L.shape
    thr = float(np.percentile(L, 99.5))
    peaks = (L == maximum_filter(L, size=2 * win + 1)) & (L > thr)
    ys, xs = np.where(peaks)
    order = np.argsort(-L[ys, xs])
    out = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if not (win < y < H - win and win < x < W - win):
            continue
        patch = a[y - win:y + win + 1, x - win:x + win + 1]
        pl = _lum(patch)
        if pl.max() >= sat:                       # skip saturated cores (centroid unreliable)
            continue
        # require all three channels to carry signal (a coloured star still has some in each)
        if a.ndim == 3 and min(patch[..., c].max() for c in range(3)) < thr * 0.15:
            continue
        out.append((y, x))
        if len(out) >= max_stars:
            break
    return out


def _channel_offset(patch_c: np.ndarray, patch_ref: np.ndarray) -> Optional[Tuple[float, float]]:
    """Flux-weighted centroid offset of one channel vs the reference, within a patch."""
    n = patch_c.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    def _com(p):
        p = p - p.min()
        s = p.sum()
        if s < 1e-6:
            return None
        return (float((p * yy).sum() / s), float((p * xx).sum() / s))
    cc, cr = _com(patch_c), _com(patch_ref)
    if cc is None or cr is None:
        return None
    return (cc[0] - cr[0], cc[1] - cr[1])          # channel − reference (dy, dx)


def _channel_sigma(patch_c: np.ndarray) -> Optional[float]:
    """Flux-weighted RMS radius (PSF sigma proxy) of a star in one channel patch."""
    n = patch_c.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    p = patch_c - patch_c.min()
    s = p.sum()
    if s < 1e-6:
        return None
    cy, cx = (p * yy).sum() / s, (p * xx).sum() / s
    var = (p * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / s
    return float(np.sqrt(max(var, 1e-6) / 2.0))            # 2-D var -> per-axis sigma


def _star_mask(a: np.ndarray, stars: list, win: int) -> np.ndarray:
    """Feathered mask over the bright stars (where size-matching is applied)."""
    H, W = a.shape[:2]
    m = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[-win:win + 1, -win:win + 1]
    disk = (yy ** 2 + xx ** 2) <= win ** 2
    for (y, x) in stars:
        m[y - win:y + win + 1, x - win:x + win + 1][disk] = 1.0
    return gaussian_filter(m, sigma=win / 2.0)


def _fit_poly(pts: np.ndarray, vals: np.ndarray, shape) -> np.ndarray:
    """Robust 2-D quadratic fit of an offset component; evaluate on the full pixel grid.

    ``pts`` are (y, x) star locations normalised to [-1, 1]; a 6-term quadratic basis
    (1, x, y, x², xy, y²) captures the smooth radial/tilt CA field. One reweighted pass
    down-weights outliers (mis-measured stars).
    """
    H, W = shape
    yn = pts[:, 0] / (H / 2) - 1.0
    xn = pts[:, 1] / (W / 2) - 1.0

    def basis(y, x):
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)

    A = basis(yn, xn)
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    resid = vals - A @ coef
    scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
    w = 1.0 / (1.0 + (resid / (3.0 * scale)) ** 2)               # soft outlier down-weight
    Aw = A * w[:, None]
    coef, *_ = np.linalg.lstsq(Aw, vals * w, rcond=None)

    yy, xx = np.mgrid[0:H, 0:W]
    yn2 = yy / (H / 2) - 1.0
    xn2 = xx / (W / 2) - 1.0
    return basis(yn2, xn2) @ coef                                # (H, W) modelled offset


def measure_offsets(img: np.ndarray, *, max_stars: int = 400, win: int = 7,
                    sat: float = 0.98) -> dict:
    """Measure per-star R→B and G→B centroid offsets. Returns a summary dict (no warping)."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] < 3:
        return {"n": 0, "median_px": 0.0}
    stars = _star_centroids(a, max_stars=max_stars, win=win, sat=sat)
    rows = []
    for (y, x) in stars:
        pr = a[y - win:y + win + 1, x - win:x + win + 1, 0]
        pg = a[y - win:y + win + 1, x - win:x + win + 1, 1]
        pb = a[y - win:y + win + 1, x - win:x + win + 1, 2]
        oR = _channel_offset(pr, pb)
        oG = _channel_offset(pg, pb)
        if oR is None or oG is None:
            continue
        rows.append((y, x, oR[0], oR[1], oG[0], oG[1]))
    if not rows:
        return {"n": 0, "median_px": 0.0}
    arr = np.array(rows)
    magR = np.hypot(arr[:, 2], arr[:, 3])
    magG = np.hypot(arr[:, 4], arr[:, 5])
    return {"n": len(rows), "rows": arr,
            "median_px": float(np.median(np.maximum(magR, magG))),
            "median_R_px": float(np.median(magR)), "median_G_px": float(np.median(magG))}


def correct_chromatic_aberration(img: np.ndarray, *, strength: float = 1.0,
                                 max_stars: int = 400, win: int = 7, sat: float = 0.98,
                                 min_stars: int = 12, match_size: bool = True,
                                 log: Callable[[str], None] = _noop) -> np.ndarray:
    """Correct lateral chromatic aberration: align R/G to B AND match per-channel star sizes.

    Two stages, both star-measured and both scaled by ``strength`` (0..1):
      1. **Position** — warp R/G by a smooth polynomial offset field so their stars sit on B's.
      2. **Size** (``match_size``) — the channels also differ in PSF size (a blue star is a big
         blue blob with tighter R/G), which leaves a coloured edge (the crescent) even when
         centred. Blur the tighter channels up to the softest channel's size, applied only
         through a feathered star mask so nebula detail is untouched.

    Mono images, or frames with fewer than ``min_stars`` measurable stars, are returned as-is.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] < 3 or strength <= 0.0:
        return a
    m = measure_offsets(a, max_stars=max_stars, win=win, sat=sat)
    if m["n"] < min_stars:
        log(f"  chromatic aberration: only {m['n']} usable stars (<{min_stars}) — skipped")
        return a
    arr = m["rows"]
    pts = arr[:, 0:2]
    H, W = a.shape[:2]
    s = float(np.clip(strength, 0.0, 1.0))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    out = a.copy()
    for ci, (dy_col, dx_col) in ((0, (2, 3)), (1, (4, 5))):     # R uses cols 2,3; G uses 4,5
        dy = _fit_poly(pts, arr[:, dy_col], (H, W)) * s
        dx = _fit_poly(pts, arr[:, dx_col], (H, W)) * s
        # sample the channel at (y+dy, x+dx): its star sits at ref+offset, so pulling from
        # +offset moves that star back onto B's position.
        out[..., ci] = map_coordinates(a[..., ci], [yy + dy, xx + dx],
                                       order=1, mode="nearest")
    msg = (f"  chromatic aberration: aligned R,G to B from {m['n']} stars "
           f"(median offset {m['median_px']:.2f}px, strength {s:.2f})")

    if match_size:
        stars = [(int(y), int(x)) for y, x in pts]
        sig = []
        for c in range(3):
            vals = [_channel_sigma(out[y - win:y + win + 1, x - win:x + win + 1, c])
                    for (y, x) in stars]
            vals = [v for v in vals if v is not None]
            sig.append(float(np.median(vals)) if vals else 0.0)
        target = max(sig)
        smask = _star_mask(out, stars, win)[..., None]
        for c in range(3):
            extra = np.sqrt(max(target ** 2 - sig[c] ** 2, 0.0))
            if extra > 0.3:                                    # blur tighter channels up to target
                blurred = gaussian_filter(out[..., c], sigma=extra * s)
                out[..., c] = out[..., c] * (1 - smask[..., 0]) + blurred * smask[..., 0]
        msg += f"; matched star sizes R/G/B {sig[0]:.2f}/{sig[1]:.2f}/{sig[2]:.2f}px → {target:.2f}px"

    log(msg)
    return np.clip(out, 0.0, 1.0)
