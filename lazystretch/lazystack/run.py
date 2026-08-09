"""LazyStack orchestration — dataset folder → calibrated, registered, integrated master.

Scans ``lights/`` ``darks/`` ``flats/`` ``biases/`` subfolders (case-insensitive; if there is
no ``lights/`` the folder's own frames are treated as uncalibrated lights), builds calibration
masters, calibrates + cosmetic-corrects + measures + culls the lights, registers the survivors
to the ranked reference, integrates them, measures the junk edges, and writes the master as
FITS carrying the ``LZS*`` contract keywords the LazyStretch tab consumes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ..io.cache import cached_load
from ..io.image_io import RAW_EXT, load_image, save_image
from . import calibrate as cal, contract, integrate as integ, measure as meas, register as reg

FRAME_EXTS = {".xisf", ".fits", ".fit", ".fts", ".tif", ".tiff", ".png"}
_SUBSETS = ("lights", "darks", "flats", "biases")
MASTER_NAME = "lazystack_master.fits"
_RAWPY_OK = None


def _rawpy_available() -> bool:
    """Camera raws (demosaiced on load) are only usable when rawpy is installed."""
    global _RAWPY_OK
    if _RAWPY_OK is None:
        try:
            import rawpy  # noqa: F401
            _RAWPY_OK = True
        except Exception:
            _RAWPY_OK = False
    return _RAWPY_OK


def _loadable_exts() -> set:
    return FRAME_EXTS | RAW_EXT if _rawpy_available() else FRAME_EXTS


def _noop(_m: str) -> None:
    pass


def _list(folder: Path) -> List[str]:
    if not folder.is_dir():
        return []
    exts = _loadable_exts()
    return [str(f) for f in sorted(folder.iterdir()) if f.suffix.lower() in exts]


def find_sets(folder: str) -> Dict[str, List[str]]:
    """Locate the four calibration subsets (case-insensitive); lights-only if none."""
    root = Path(folder)
    sub = {name: [] for name in _SUBSETS}
    if root.is_dir():
        by_lower = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
        for name in _SUBSETS:
            if name in by_lower:
                sub[name] = _list(by_lower[name])
            elif name[:-1] in by_lower:                 # 'light' etc.
                sub[name] = _list(by_lower[name[:-1]])
    if not sub["lights"]:
        sub["lights"] = _list(root)                     # lights-only mode
    return sub


def _make_loader(folder: str, enabled: bool,
                 log: Callable[[str], None] = _noop) -> Callable[[str], Optional[np.ndarray]]:
    """A frame loader that caches decoded raws under <folder>/lazystack/cache (and logs them)."""
    cache_dir = Path(folder) / "lazystack" / "cache"
    return lambda path: cached_load(path, cache_dir, enabled=enabled, log=log)


def _master(paths: List[str], load: Callable, log, label: str, *,
            bias: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    frames = [f for f in (load(p) for p in paths) if f is not None]
    if not frames:
        return None
    if bias is not None:                                 # flats are bias-calibrated first
        frames = [cal.bias_calibrate(f, bias) for f in frames]
    log(f"  master {label}: combining {len(frames)} frame(s)")
    return integ.combine_master(frames)


def measure_only(folder: str, params, *, log: Callable[[str], None] = _noop) -> Optional[dict]:
    """The Phase-1 advisor: measure + cull the lights, stack nothing."""
    sets = find_sets(folder)
    lights = sets["lights"]
    if len(lights) < 2:
        log(f"Found {len(lights)} light(s) — need at least 2.")
        return None
    log(f"Measuring {len(lights)} lights…")
    load = _make_loader(folder, params.reuse_cache, log)
    measures = []
    for i, p in enumerate(lights):
        log(f"  [{i + 1}/{len(lights)}] {Path(p).name}: measuring…")
        measures.append(meas.measure_frame(load(p)))
    culled = meas.cull(measures, params, log=log)
    return {"measures": measures, "cull": culled, "lights": lights}


def stack(folder: str, params, *, log: Callable[[str], None] = _noop) -> Optional[dict]:
    """Full chain: calibrate → cosmetic → measure/cull → register → integrate → stamp."""
    sets = find_sets(folder)
    lights = sets["lights"]
    if len(lights) < 2:
        log(f"Found {len(lights)} light(s) — need at least 2.")
        return None
    log(f"LazyStack: {folder} — {len(lights)} lights, {len(sets['darks'])} darks, "
        f"{len(sets['flats'])} flats, {len(sets['biases'])} biases.")

    load = _make_loader(folder, params.reuse_cache, log)
    master_bias = _master(sets["biases"], load, log, "bias") if params.do_calibrate else None
    master_dark = _master(sets["darks"], load, log, "dark") if params.do_calibrate else None
    master_flat = (_master(sets["flats"], load, log, "flat", bias=master_bias)
                   if params.do_calibrate else None)

    log(f"Calibrating {len(lights)} lights…")
    cframes, exposure = [], 0.0
    for i, p in enumerate(lights):
        log(f"  [{i + 1}/{len(lights)}] {Path(p).name}…")
        img = load(p)
        if img is None:
            continue
        if params.do_calibrate:
            img = cal.calibrate_light(img, bias=master_bias, dark=master_dark, flat=master_flat)
        if params.do_cosmetic:
            img = cal.cosmetic_correct(img, master_dark)
        cframes.append(img)
        try:
            exp = load_image(p).keyword("EXPTIME")
            exposure += float(exp) if exp is not None else 0.0
        except Exception:
            pass
    if len(cframes) < 2:
        log("Fewer than 2 calibrated lights — nothing to stack.")
        return None

    log("Measuring + culling…")
    measures = [meas.measure_frame(f) for f in cframes]
    culled = meas.cull(measures, params, log=log)
    keep = culled["keep"]
    ref_global = culled["reference"]
    kept_frames = [cframes[i] for i in keep]
    ref_local = keep.index(ref_global) if ref_global in keep else 0

    aligned, kept_idx = reg.register(kept_frames, ref_local, log=log)
    log(f"Integrating {len(aligned)} frame(s)…")
    weights = [max(1e-3, measures[keep[i]]["snr"]) if measures[keep[i]] else 1.0
               for i in kept_idx]
    master = integ.integrate(aligned, weights=weights,
                             sigma_low=params.sigma_low, sigma_high=params.sigma_high)

    edges = contract.measure_edges(master)
    header = contract.contract_header(len(aligned), edges, exposure)
    out_dir = Path(folder) / "lazystack"
    out_dir.mkdir(parents=True, exist_ok=True)
    master_path = out_dir / MASTER_NAME
    save_image(str(master_path), master, bit_depth=16, header=header)
    log(f"Master: {master_path}  (LZSNSUB={len(aligned)}, crop L{edges['L']} R{edges['R']} "
        f"T{edges['T']} B{edges['B']})")
    return {"master": master, "master_path": str(master_path), "n_stacked": len(aligned),
            "n_lights": len(lights), "cull": culled, "edges": edges,
            "registered_with": "astroalign" if reg.astroalign_available() else "fft-translation"}
