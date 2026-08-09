"""LazyStack orchestration — dataset folder → calibrated, registered, integrated master.

Scans ``lights/`` ``darks/`` ``flats/`` ``biases/`` subfolders (case-insensitive; if there is
no ``lights/`` the folder's own frames are treated as uncalibrated lights), builds calibration
masters, calibrates + cosmetic-corrects + measures + culls the lights, registers the survivors
to the ranked reference, integrates them, measures the junk edges, and writes the master as
FITS carrying the ``LZS*`` contract keywords the LazyStretch tab consumes.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ..io.cache import _cache_key, cached_load
from ..io.image_io import RAW_EXT, load_image, save_image
from . import (
    calibrate as cal,
    contract,
    integrate as integ,
    measure as meas,
    normalize as nrm,
    register as reg,
)

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


def _master(paths: List[str], load: Callable, work: Optional[Path], log, label: str, *,
            bias: Optional[np.ndarray] = None, reuse: bool = False) -> Optional[np.ndarray]:
    """Build a calibration master. Streams to ``work`` (bounded RAM) unless work is None.

    When ``reuse`` (staged only), a previously-built master is loaded from disk instead of
    being recombined.
    """
    if work is not None and reuse:
        mp = work / f"master_{label}.npy"
        if mp.exists():
            try:
                m = np.load(str(mp))
                log(f"  master {label}: reused from cache")
                return m
            except Exception:
                pass
    tmp: List[Path] = []
    loaded: List[np.ndarray] = []
    for i, p in enumerate(paths):
        f = load(p)
        if f is None:
            continue
        if bias is not None:                             # flats are bias-calibrated first
            f = cal.bias_calibrate(f, bias)
        if work is None:
            loaded.append(np.asarray(f, dtype=np.float32))
        else:
            tp = work / f"m_{label}_{i:04d}.npy"
            np.save(str(tp), np.asarray(f, dtype=np.float32))
            tmp.append(tp)
        del f
    if work is None:
        if not loaded:
            return None
        log(f"  master {label}: combining {len(loaded)} frame(s)")
        return integ.combine_master(loaded)
    if not tmp:
        return None
    log(f"  master {label}: combining {len(tmp)} frame(s)")
    m = integ.combine_files(tmp, sigma_low=5.0, sigma_high=5.0)
    for tp in tmp:
        try:
            tp.unlink()
        except OSError:
            pass
    if reuse:
        try:
            np.save(str(work / f"master_{label}.npy"), np.asarray(m, dtype=np.float32))
        except OSError:
            pass
    return m


def _frame_key(path: str, index: int) -> str:
    """Cache key for a frame's staged files: source identity, index fallback."""
    try:
        return _cache_key(path)
    except OSError:
        return f"idx{index:04d}"


def _load_measures_cache(work: Path) -> Dict[str, object]:
    p = work / "measures.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def _save_measures_cache(work: Path, cache: Dict[str, object]) -> None:
    try:
        (work / "measures.json").write_text(json.dumps(cache), encoding="utf-8")
    except OSError:
        pass


def _cleanup(work: Path) -> None:
    shutil.rmtree(work, ignore_errors=True)


def _prune_work(work: Path, keep: set, log) -> None:
    """Delete work files not produced by the current run (orphaned reg/cal from earlier
    runs with a different reference, normalization setting, or frame set)."""
    try:
        entries = [f for f in work.iterdir() if f.is_file()]
    except OSError:
        return
    removed = 0
    for f in entries:
        if f.name not in keep:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        log(f"Pruned {removed} stale work file(s).")


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

    Two modes (``params.stage_to_disk``): staged (default) writes calibrated/registered
    frames to ``lazystack/work`` as float32 and streams them, so a large burst never holds
    the whole stack in RAM (that OOM'd/bus-errored on real data); in-memory keeps everything
    in RAM (faster, no work files) for bursts that comfortably fit.
    """
    sets = find_sets(folder)
    lights = sets["lights"]
    if len(lights) < 2:
        log(f"Found {len(lights)} light(s) — need at least 2.")
        return None
    log(f"LazyStack: {folder} — {len(lights)} lights, {len(sets['darks'])} darks, "
        f"{len(sets['flats'])} flats, {len(sets['biases'])} biases.")

    out_dir = Path(folder) / "lazystack"
    out_dir.mkdir(parents=True, exist_ok=True)
    staged = bool(params.stage_to_disk)
    work = out_dir / "work" if staged else None
    if work is not None:
        work.mkdir(parents=True, exist_ok=True)
    log("Mode: staged (low memory, uses lazystack/work)" if staged
        else "Mode: in-memory (no work files; holds the burst in RAM)")
    load = _make_loader(folder, params.reuse_cache, log)
    reuse = staged and bool(params.reuse_cache)          # reuse existing work files
    if reuse:
        log("Reusing any existing work files (uncheck 'Reuse cached intermediates' to "
            "force a fresh run).")

    master_bias = (_master(sets["biases"], load, work, log, "bias", reuse=reuse)
                   if params.do_calibrate else None)
    master_dark = (_master(sets["darks"], load, work, log, "dark", reuse=reuse)
                   if params.do_calibrate else None)
    master_flat = (_master(sets["flats"], load, work, log, "flat", bias=master_bias, reuse=reuse)
                   if params.do_calibrate else None)

    def _get(handle):
        return np.load(str(handle)) if isinstance(handle, Path) else handle

    # --- stage 1: calibrate + cosmetic + measure (stage to disk or keep in RAM) ---
    log(f"Calibrating + measuring {len(lights)} lights…")
    frames: list = []                    # Path (staged) or ndarray (in-memory)
    names: List[str] = []
    keys: List[Optional[str]] = []
    measures: List[Optional[dict]] = []
    meas_cache = _load_measures_cache(work) if reuse else {}
    exposure = 0.0
    for i, p in enumerate(lights):
        name = Path(p).name
        log(f"  [{i + 1}/{len(lights)}] {name}…")
        key = _frame_key(p, i) if staged else None
        cal_path = (work / f"cal_{key}.npy") if staged else None
        img = None
        if reuse and cal_path is not None and cal_path.exists():
            try:
                img = np.load(str(cal_path))
                log("    reusing cached calibration")
            except Exception:
                img = None
        if img is None:
            raw = load(p)
            if raw is None:
                continue
            if params.do_calibrate:
                raw = cal.calibrate_light(raw, bias=master_bias, dark=master_dark, flat=master_flat)
            if params.do_cosmetic:
                raw = cal.cosmetic_correct(raw, master_dark)
            img = np.asarray(raw, dtype=np.float32)
            if staged:
                np.save(str(cal_path), img)
        if reuse and key in meas_cache:
            m = meas_cache[key]                          # measurement cached by source identity
        else:
            m = meas.measure_frame(img)
            if reuse and key is not None:
                meas_cache[key] = m
        measures.append(m)
        names.append(name)
        keys.append(key)
        if staged:
            frames.append(cal_path)
            del img
        else:
            frames.append(img)
        try:
            exp = load_image(p).keyword("EXPTIME")
            exposure += float(exp) if exp is not None else 0.0
        except Exception:
            pass
    if reuse:
        _save_measures_cache(work, meas_cache)
    if len(frames) < 2:
        if work is not None and not reuse:
            _cleanup(work)
        log("Fewer than 2 calibrated lights — nothing to stack.")
        return None

    culled = meas.cull(measures, params, log=log)
    keep = culled["keep"]
    ref_global = culled["reference"] if culled["reference"] in keep else keep[0]

    # --- stage 2: register the kept frames to the reference (reg cache keyed src+ref) ---
    log(f"Registering {len(keep)} frames to reference {names[ref_global]}…")
    ref_key = keys[ref_global] if staged else None
    ref = _get(frames[ref_global])
    aligner = reg.Aligner(ref, log=log)
    local_norm = bool(params.local_normalize)
    normalize = bool(params.normalize) or local_norm     # LN includes the global step
    ref_med, ref_sig = nrm.frame_stats(ref) if normalize else (0.0, 1.0)
    nsuf = "_ln" if local_norm else ("_n" if normalize else "")   # changes reg output
    if local_norm:
        log("Local normalization: matching each frame's gradient to the reference.")
    elif normalize:
        log("Normalizing each frame to the reference (background + scale).")
    aligned_handles: list = []
    weights: List[float] = []
    for j, idx in enumerate(keep):
        log(f"  [{j + 1}/{len(keep)}] {names[idx]}: registering…")
        m = measures[idx]
        w = max(1e-3, m["snr"]) if m else 1.0
        reg_path = (work / f"reg_{keys[idx]}_{ref_key}{nsuf}.npy") if staged else None
        if reuse and reg_path is not None and reg_path.exists():
            log("    reusing cached registration")
            aligned_handles.append(reg_path)
            weights.append(w)
            continue
        if idx == ref_global:
            aligned = ref
        else:
            frame = _get(frames[idx])
            try:
                aligned = aligner.align(frame)
            except Exception as e:
                log(f"    dropped ({e})")
                continue
        if local_norm:
            aligned = nrm.local_normalize_to_ref(aligned, ref, ref_med, ref_sig)
        elif normalize:
            aligned = nrm.normalize_to_ref(aligned, ref_med, ref_sig)
        weights.append(w)
        if staged:
            np.save(str(reg_path), np.asarray(aligned, dtype=np.float32))
            aligned_handles.append(reg_path)
        else:
            aligned_handles.append(np.asarray(aligned, dtype=np.float32))
    del ref
    if staged and not reuse:
        for h in frames:                                 # calibrated staging no longer needed
            try:
                h.unlink()
            except OSError:
                pass
    if not aligned_handles:
        if work is not None and not reuse:
            _cleanup(work)
        log("No frames registered — nothing to integrate.")
        return None

    # --- stage 3: integrate ---
    n_stacked = len(aligned_handles)
    log(f"Integrating {n_stacked} frame(s)…")
    if staged:
        master = integ.combine_files(aligned_handles, weights=weights,
                                     sigma_low=params.sigma_low, sigma_high=params.sigma_high,
                                     log=log)
    else:
        master = integ.integrate(aligned_handles, weights=weights,
                                 sigma_low=params.sigma_low, sigma_high=params.sigma_high)

    edges = contract.measure_edges(master)
    header = contract.contract_header(n_stacked, edges, exposure)
    master_path = out_dir / MASTER_NAME
    save_image(str(master_path), master, bit_depth=16, header=header)
    log(f"Master: {master_path}  (LZSNSUB={n_stacked}, crop L{edges['L']} R{edges['R']} "
        f"T{edges['T']} B{edges['B']})")
    if work is not None and not reuse:
        _cleanup(work)                                   # fresh run — drop all work files
    elif work is not None and reuse:                     # keep this run's files, prune orphans
        keep_names = {h.name for h in frames if isinstance(h, Path)}
        keep_names |= {h.name for h in aligned_handles if isinstance(h, Path)}
        keep_names |= {f"master_{lbl}.npy" for lbl in ("bias", "dark", "flat")}
        keep_names.add("measures.json")
        _prune_work(work, keep_names, log)
    return {"master": master, "master_path": str(master_path), "n_stacked": n_stacked,
            "n_lights": len(lights), "cull": culled, "edges": edges,
            "registered_with": "astroalign" if reg.astroalign_available() else "fft-translation"}
