"""Halo Tamer — reflection filter-ring / ghost suppression (v1.5.1).

Faithful port of the *automatic* scan path of ``LS.haloScan`` (LazyStretch.js:3792-3962) +
the ring subtraction of ``Pipe.haloTamer`` (:6064-6215). Dual-band / LP filters throw a
faint concentric ring ("ghost") around bright stars; it is red-dominant on OSC/Ha, so the
subtraction is per-channel. Detection is a monotone-envelope + descend-back test on radial
median profiles around the brightest stars; the ring is then subtracted as a feathered
annulus, with a luminance guard so the stars themselves ride above the subtraction.

The port implements the automatic scan (no user-marked ghosts — the port has no click UI);
the disc/glow marked modes are omitted. Star detection uses photutils (already a dependency);
if it is absent the tamer is a graceful no-op.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

R0 = 12
DR = 6
_SCREEN_AMP = 0.009
_FINE_AMP = 0.012
_MAX_RINGS = 24
_bin_cache: dict = {}


def _noop(_m: str) -> None:
    pass


def _lum(img: np.ndarray) -> np.ndarray:
    return img[..., :3].mean(axis=2) if img.ndim == 3 else img


def _median(a: np.ndarray) -> float:
    return float(np.median(a)) if a.size else 0.0


def _bin_layout(rmax: int, stride: int, nbins: int):
    """Per-bin flat index lists over a strided (2·rmax+1)² patch (cached)."""
    key = (rmax, stride, nbins)
    hit = _bin_cache.get(key)
    if hit is not None:
        return hit
    coords = np.arange(-rmax, rmax + 1, stride)
    ys, xs = np.meshgrid(coords, coords, indexing="ij")
    r = np.hypot(xs, ys).ravel()
    b = np.floor((r - R0) / DR).astype(int)
    members = [np.nonzero(b == bb)[0] for bb in range(nbins)]
    _bin_cache[key] = (members, coords.size)
    return members, coords.size


def _radius_of(b: int) -> float:
    return R0 + b * DR + DR / 2.0


def _detect(prof: np.ndarray, nbins: int) -> Optional[dict]:
    """Monotone envelope + contained band + descend-back (LS.haloScan detectOn)."""
    env = np.minimum.accumulate(prof)
    bg = _median(prof[int(nbins * 0.75):])
    thr = max(0.006, 0.03 * bg)
    excess = prof - env
    best = None
    i = 0
    while i < nbins:
        if excess[i] > thr and _radius_of(i) > 32:
            j = i
            while j < nbins and excess[j] > thr:
                j += 1
            band = excess[i:j]
            pk = i + int(np.argmax(band))
            amp = float(band.max())
            descends = bool(np.any(prof[j:min(nbins, j + 10)] < prof[pk] - 0.5 * amp))
            if (descends and _radius_of(j - 1) - _radius_of(i) >= 15
                    and (best is None or amp > best["amp"])):
                best = {"i1": i, "i2": j, "r1": _radius_of(i), "r2": _radius_of(j - 1),
                        "rp": _radius_of(pk), "amp": amp, "bg": bg}
            i = j
        else:
            i += 1
    return best


def _profile(chans: List[np.ndarray], cx: int, cy: int, rmax: int, stride: int,
             nbins: int, with_channels: bool):
    """Radial median profile (mean-of-channels) around (cx,cy); optional per-channel bins."""
    members, _ = _bin_layout(rmax, stride, nbins)
    C = len(chans)
    patches = [c[cy - rmax:cy + rmax + 1, cx - rmax:cx + rmax + 1][::stride, ::stride].ravel()
               for c in chans]
    lum_flat = np.mean(patches, axis=0)
    prof = np.array([_median(lum_flat[m]) for m in members])
    cbins = None
    if with_channels:
        cbins = [np.array([_median(pf[m]) for m in members]) for pf in patches]
    return prof, cbins


def halo_scan(img: np.ndarray, log: Callable[[str], None] = _noop) -> List[dict]:
    """Detect reflection rings around the brightest stars. Returns ring dicts."""
    a = np.asarray(img, dtype=np.float64)
    H, W = a.shape[0], a.shape[1]
    rmax = min(300, (min(W, H) // 8))
    if rmax < 90:
        return []
    nbins = (rmax - R0) // DR
    if nbins < 4:
        return []

    try:
        from astropy.stats import sigma_clipped_stats
        from photutils.detection import DAOStarFinder
    except Exception:
        log("Halo Tamer: photutils not available — skipped.")
        return []

    lum = _lum(a)
    chans = [a[..., c] for c in range(3)] if a.ndim == 3 else [a]
    _mean, med, std = sigma_clipped_stats(lum, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        return []
    try:
        sources = DAOStarFinder(fwhm=3.0, threshold=max(5.0 * std, 1e-4))(lum - med)
    except Exception:
        sources = None
    if sources is None or len(sources) == 0:
        return []
    order = np.argsort(np.asarray(sources["flux"]))[::-1]
    xs = np.asarray(sources["xcentroid"])[order]
    ys = np.asarray(sources["ycentroid"])[order]

    margin = rmax + 8
    picked = []
    for x, y in zip(xs, ys):
        if x < margin or x > W - margin or y < margin or y > H - margin:
            continue
        xi, yi = int(round(x)), int(round(y))
        if any(abs(px - xi) < 40 and abs(py - yi) < 40 for px, py in picked):
            continue
        picked.append((xi, yi))
        if len(picked) >= 300:
            break

    rings = []
    for cx, cy in picked:
        screen, _ = _profile(chans, cx, cy, rmax, 4, nbins, False)
        sdet = _detect(screen, nbins)
        if sdet is None or sdet["amp"] < _SCREEN_AMP:
            continue
        fine, cbins = _profile(chans, cx, cy, rmax, 2, nbins, True)
        best = _detect(fine, nbins)
        if best is None or best["amp"] < _FINE_AMP:
            continue
        amps = []
        for pc in cbins:
            run_min = np.minimum.accumulate(pc)
            band = (pc - run_min)[best["i1"]:best["i2"]]
            amps.append(float(band.max()) if band.size else 0.0)
        rings.append({"x": cx, "y": cy, "r1": best["r1"], "r2": best["r2"],
                      "rp": best["rp"], "amp": best["amp"], "bg": best["bg"], "amps": amps})
    rings.sort(key=lambda r: r["amp"], reverse=True)
    return rings[:_MAX_RINGS]


def apply_halo_rings(img: np.ndarray, rings: List[dict], strength: float = 1.0) -> np.ndarray:
    """Subtract each ring as a feathered annulus per channel, guarding the stars/cores."""
    out = np.array(img, dtype=np.float64, copy=True)
    color = out.ndim == 3
    H, W = out.shape[0], out.shape[1]
    for g in rings:
        r1, r2 = g["r1"], g["r2"]
        feather = max(6.0, 0.3 * (r2 - r1))
        guard = g["bg"] + max(0.008, 3.0 * g["amp"])
        cx, cy = g["x"], g["y"]
        pad = int(np.ceil(r2 + feather)) + 1
        x0, x1 = max(0, cx - pad), min(W, cx + pad + 1)
        y0, y1 = max(0, cy - pad), min(H, cy + pad + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - cx, yy - cy)
        w = np.clip((d - r1) / feather, 0, 1) * np.clip((r2 - d) / feather, 0, 1)
        nch = 3 if color else 1
        amps = g["amps"]
        for c in range(nch):
            amp = amps[c] if c < len(amps) else amps[0]
            sub = strength * amp * w
            region = out[y0:y1, x0:x1, c] if color else out[y0:y1, x0:x1]
            hit = (w > 0) & (region < guard)
            region[hit] = np.maximum(0.0, region[hit] - sub[hit])
    return out


def halo_tamer(img: np.ndarray, strength: float = 1.0,
               log: Callable[[str], None] = _noop) -> np.ndarray:
    """Scan for reflection rings and suppress them (the automatic finish-stage pass)."""
    rings = halo_scan(img, log)
    if not rings:
        log("Halo Tamer: no reflection ghosts detected.")
        return np.asarray(img, dtype=np.float64)
    out = apply_halo_rings(img, rings, strength)
    log(f"Halo Tamer: {len(rings)} reflection ring(s) suppressed (per-channel).")
    return out
