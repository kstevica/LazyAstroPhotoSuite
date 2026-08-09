"""LazyStack orchestration — dataset folder → calibrated, registered, integrated master.

Scans ``lights/`` ``darks/`` ``flats/`` ``biases/`` subfolders (case-insensitive; if there is
no ``lights/`` the folder's own frames are treated as uncalibrated lights), builds calibration
masters, calibrates + cosmetic-corrects + measures + culls the lights, registers the survivors
to the ranked reference, integrates them, measures the junk edges, and writes the master as
FITS carrying the ``LZS*`` contract keywords the LazyStretch tab consumes.
"""
from __future__ import annotations

import shutil
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


def _master(paths: List[str], load: Callable, work: Path, log, label: str, *,
            bias: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Build a calibration master, streaming each frame to disk (bounded memory)."""
    tmp: List[Path] = []
    for i, p in enumerate(paths):
        f = load(p)
        if f is None:
            continue
        if bias is not None:                             # flats are bias-calibrated first
            f = cal.bias_calibrate(f, bias)
        tp = work / f"m_{label}_{i:04d}.npy"
        np.save(str(tp), np.asarray(f, dtype=np.float32))
        tmp.append(tp)
        del f
    if not tmp:
        return None
    log(f"  master {label}: combining {len(tmp)} frame(s)")
    m = integ.combine_files(tmp, sigma_low=5.0, sigma_high=5.0)
    for tp in tmp:
        try:
            tp.unlink()
        except OSError:
            pass
    return m


def _cleanup(work: Path) -> None:
    shutil.rmtree(work, ignore_errors=True)


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
    """Full chain: calibrate → cosmetic → measure/cull → register → integrate → stamp.

    Memory-bounded: calibrated and registered frames are staged to disk as float32 ``.npy``
    and streamed, so a large burst never holds the whole stack in RAM (that OOM'd/bus-errored
    on real data). Peak RAM is a handful of frames regardless of burst size.
    """
    sets = find_sets(folder)
    lights = sets["lights"]
    if len(lights) < 2:
        log(f"Found {len(lights)} light(s) — need at least 2.")
        return None
    log(f"LazyStack: {folder} — {len(lights)} lights, {len(sets['darks'])} darks, "
        f"{len(sets['flats'])} flats, {len(sets['biases'])} biases.")

    out_dir = Path(folder) / "lazystack"
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    load = _make_loader(folder, params.reuse_cache, log)

    master_bias = _master(sets["biases"], load, work, log, "bias") if params.do_calibrate else None
    master_dark = _master(sets["darks"], load, work, log, "dark") if params.do_calibrate else None
    master_flat = (_master(sets["flats"], load, work, log, "flat", bias=master_bias)
                   if params.do_calibrate else None)

    # --- stage 1: calibrate + cosmetic + measure, staging each light to disk ---
    log(f"Calibrating + measuring {len(lights)} lights…")
    cal_paths: List[Path] = []
    names: List[str] = []
    measures: List[Optional[dict]] = []
    exposure = 0.0
    for i, p in enumerate(lights):
        log(f"  [{i + 1}/{len(lights)}] {Path(p).name}…")
        img = load(p)
        if img is None:
            continue
        if params.do_calibrate:
            img = cal.calibrate_light(img, bias=master_bias, dark=master_dark, flat=master_flat)
        if params.do_cosmetic:
            img = cal.cosmetic_correct(img, master_dark)
        img = np.asarray(img, dtype=np.float32)
        measures.append(meas.measure_frame(img))
        cp = work / f"cal_{i:04d}.npy"
        np.save(str(cp), img)
        cal_paths.append(cp)
        names.append(Path(p).name)
        try:
            exp = load_image(p).keyword("EXPTIME")
            exposure += float(exp) if exp is not None else 0.0
        except Exception:
            pass
        del img
    if len(cal_paths) < 2:
        _cleanup(work)
        log("Fewer than 2 calibrated lights — nothing to stack.")
        return None

    culled = meas.cull(measures, params, log=log)
    keep = culled["keep"]
    ref_global = culled["reference"] if culled["reference"] in keep else keep[0]

    # --- stage 2: register the kept frames to the reference, streaming to disk ---
    log(f"Registering {len(keep)} frames to reference {names[ref_global]}…")
    ref = np.load(str(cal_paths[ref_global]))
    aligner = reg.Aligner(ref, log=log)
    reg_paths: List[Path] = []
    weights: List[float] = []
    for j, idx in enumerate(keep):
        log(f"  [{j + 1}/{len(keep)}] {names[idx]}: registering…")
        if idx == ref_global:
            aligned = ref
        else:
            frame = np.load(str(cal_paths[idx]))
            try:
                aligned = aligner.align(frame)
            except Exception as e:
                log(f"    dropped ({e})")
                del frame
                continue
            del frame
        rp = work / f"reg_{idx:04d}.npy"
        np.save(str(rp), np.asarray(aligned, dtype=np.float32))
        reg_paths.append(rp)
        m = measures[idx]
        weights.append(max(1e-3, m["snr"]) if m else 1.0)
        if idx != ref_global:
            del aligned
    del ref
    for cp in cal_paths:                                 # calibrated staging no longer needed
        try:
            cp.unlink()
        except OSError:
            pass
    if not reg_paths:
        _cleanup(work)
        log("No frames registered — nothing to integrate.")
        return None

    # --- stage 3: integrate the registered stack (streamed in row bands) ---
    log(f"Integrating {len(reg_paths)} frame(s)…")
    master = integ.combine_files(reg_paths, weights=weights,
                                 sigma_low=params.sigma_low, sigma_high=params.sigma_high)

    edges = contract.measure_edges(master)
    header = contract.contract_header(len(reg_paths), edges, exposure)
    master_path = out_dir / MASTER_NAME
    save_image(str(master_path), master, bit_depth=16, header=header)
    log(f"Master: {master_path}  (LZSNSUB={len(reg_paths)}, crop L{edges['L']} R{edges['R']} "
        f"T{edges['T']} B{edges['B']})")
    _cleanup(work)
    return {"master": master, "master_path": str(master_path), "n_stacked": len(reg_paths),
            "n_lights": len(lights), "cull": culled, "edges": edges,
            "registered_with": "astroalign" if reg.astroalign_available() else "fft-translation"}
