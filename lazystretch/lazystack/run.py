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
from ..io.image_io import RAW_EXT, capture_time, load_image, save_image
from . import (
    calibrate as cal,
    contract,
    integrate as integ,
    measure as meas,
    meteors as met,
    normalize as nrm,
    register as reg,
)

FRAME_EXTS = {".xisf", ".fits", ".fit", ".fts", ".tif", ".tiff", ".png"}
_SUBSETS = ("lights", "darks", "flats", "biases")
MASTER_NAME = "lazystack_master.fits"
SNR_MAP_NAME = "lazystack_master_noise.npy"        # per-pixel σ companion (feeds the stretch SNR mask)
COVERAGE_MAP_NAME = "lazystack_master_coverage.npy"  # per-pixel frame-support companion
METEOR_MAP_NAME = "lazystack_master_meteors.npy"   # feathered linear-RGB meteor layer (composite in stretch)
METEOR_LABELS_NAME = "lazystack_master_meteorlabels.npy"  # per-meteor id map (for selection)
METEOR_META_NAME = "lazystack_master_meteors.json"  # detected-trail metadata
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
    paths: List[str] = []                # full source path parallel to names (for meteor timestamps)
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
        paths.append(p)
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

    # --- stage 1b: static hot/cold pixel repair in SENSOR space (walking-noise fix) ---
    # Must run before registration: undithered hot pixels sit at a fixed sensor coordinate and,
    # once dragged along the drift by registration, become sigma-clip-proof diagonal dashes.
    if getattr(params, "fix_walking_noise", True) and len(keep) >= 3:
        log("Scanning for static hot/cold pixels (walking-noise fix)…")
        bad = cal.static_hot_pixel_map((_get(frames[i]) for i in keep), log=log)
        nbad = int(bad.sum())
        if nbad:
            log(f"  repairing {nbad} static bad pixel(s) across {len(keep)} frame(s)…")
            for i in keep:
                fixed = cal.repair_bad_pixels(_get(frames[i]), bad)
                if staged:
                    np.save(str(frames[i]), np.asarray(fixed, dtype=np.float32))
                else:
                    frames[i] = np.asarray(fixed, dtype=np.float32)
        else:
            log("  no static bad pixels found.")

    # --- stage 2: register the kept frames to the reference (reg cache keyed src+ref) ---
    log(f"Registering {len(keep)} frames to reference {names[ref_global]}…")
    ref_key = keys[ref_global] if staged else None
    ref = _get(frames[ref_global])
    aligner = reg.Aligner(ref, log=log)
    local_norm = bool(params.local_normalize)
    normalize = bool(params.normalize) or local_norm     # LN includes the global step
    ref_med, ref_sig = nrm.frame_stats(ref) if normalize else (0.0, 1.0)
    # "_c" marks the coverage/NaN-aware reg format; "w" marks walking-noise repair applied — both
    # keep the reg cache from being reused across a format/repair change (else the repair, done on
    # the cal frames in stage 1b, is silently bypassed when a stale reg file is reused).
    wtag = "w" if getattr(params, "fix_walking_noise", True) else ""
    nsuf = ("_ln" if local_norm else ("_n" if normalize else "")) + "_c" + wtag   # changes reg output
    if local_norm:
        log("Local normalization: matching each frame's gradient to the reference.")
    elif normalize:
        log("Normalizing each frame to the reference (background + scale).")
    aligned_handles: list = []
    aligned_names: List[str] = []                        # parallel to aligned_handles == transient frame index
    aligned_paths: List[str] = []                        # full source path per aligned frame (meteor timestamp)
    weights: List[float] = []
    for j, idx in enumerate(keep):
        log(f"  [{j + 1}/{len(keep)}] {names[idx]}: registering…")
        m = measures[idx]
        w = max(1e-3, m["snr"]) if m else 1.0
        reg_path = (work / f"reg_{keys[idx]}_{ref_key}{nsuf}.npy") if staged else None
        if reuse and reg_path is not None and reg_path.exists():
            log("    reusing cached registration")
            aligned_handles.append(reg_path)
            aligned_names.append(names[idx])
            aligned_paths.append(paths[idx])
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
        nodata = ~np.isfinite(aligned)                   # registration no-overlap border
        if local_norm:
            aligned = nrm.local_normalize_to_ref(aligned, ref, ref_med, ref_sig)
        elif normalize:
            aligned = nrm.normalize_to_ref(aligned, ref_med, ref_sig)
        if nodata.any():                                 # keep no-data as no-data (NaN) for coverage
            aligned = np.asarray(aligned, dtype=np.float64)
            aligned[nodata] = np.nan
        weights.append(w)
        aligned_names.append(names[idx])
        aligned_paths.append(paths[idx])
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
    emit_snr = getattr(params, "emit_snr_map", True)
    emit_met = getattr(params, "preserve_meteors", True)
    ikw = dict(return_coverage=True, return_noise=emit_snr, return_transient=emit_met)
    if staged:
        res = integ.combine_files(aligned_handles, weights=weights, sigma_low=params.sigma_low,
                                  sigma_high=params.sigma_high, log=log, **ikw)
    else:
        res = integ.integrate(aligned_handles, weights=weights, sigma_low=params.sigma_low,
                              sigma_high=params.sigma_high, **ikw)
    res = list(res)                                      # order: master, coverage, [noise], [transient, tframe]
    master = res.pop(0)
    coverage = res.pop(0)
    noise = res.pop(0) if emit_snr else None
    transient, tframe = (res.pop(0), res.pop(0)) if emit_met else (None, None)

    edge_crop = getattr(params, "edge_crop", True)
    if edge_crop:                                        # crop to the common fully-covered overlap
        edges = contract.coverage_edges(coverage, n_stacked)
    else:
        edges = contract.measure_edges(np.nan_to_num(master, nan=0.0))
    master = np.nan_to_num(master, nan=0.0)              # no NaN (no-data) reaches the 16-bit save
    if edge_crop:
        master = contract.crop_to_contract(master, edges)
    if getattr(params, "fix_banding", True):             # remove column/row fixed-pattern banding
        master = cal.suppress_banding(master)
        log("Suppressed column/row banding (fixed-pattern residual).")
    header = contract.contract_header(n_stacked, edges, exposure)
    master_path = out_dir / MASTER_NAME
    save_image(str(master_path), master, bit_depth=16, header=header)
    log(f"Master: {master_path}  (LZSNSUB={n_stacked}, crop L{edges['L']} R{edges['R']} "
        f"T{edges['T']} B{edges['B']})")

    # Companion per-pixel noise map (standard error), cropped identically to the master. The
    # stretch reads it to build an SNR-protect mask that a luminance mask can't (it separates
    # faint real signal from pure noise). Best-effort — never fails the stack.
    snr_path = None
    if emit_snr and noise is not None:
        try:
            noise_c = contract.crop_to_contract(noise, edges) if edge_crop else noise
            fill = float(np.nanmedian(noise_c)) if np.any(np.isfinite(noise_c)) else 0.0
            noise_c = np.nan_to_num(noise_c, nan=fill).astype(np.float32)
            snr_path = out_dir / SNR_MAP_NAME
            np.save(str(snr_path), noise_c)
            # Frame-support (coverage) companion — feeds the same SNR-protect confidence: fewer
            # frames covering a pixel == less reliable == protect more.
            cov_c = contract.crop_to_contract(coverage, edges) if edge_crop else coverage
            np.save(str(out_dir / COVERAGE_MAP_NAME), np.asarray(cov_c, dtype=np.int32))
            log(f"Noise + coverage maps: {out_dir}  (feed the stretch SNR-protect mask)")
        except Exception as e:                           # optional companions, never fatal
            log(f"  (noise/coverage map skipped: {e})")
            snr_path = None

    # Meteor preservation: the transient map holds the high-side-rejected (meteor/plane) light the
    # clip discarded; detect trails, save the feathered linear-RGB layer for the stretch to composite.
    meteor_path = None
    meteor_list: List[dict] = []
    if emit_met and transient is not None:
        try:
            meteor_list, soft, labels = met.detect_meteors(np.nan_to_num(transient, nan=0.0),
                                                           tframe, log=log)
            if meteor_list:
                for mm in meteor_list:                    # attach the source light filename + capture time
                    fi = int(mm.get("frame", -1))
                    if 0 <= fi < len(aligned_names):
                        mm["source"] = aligned_names[fi]
                        mm["timestamp"] = capture_time(aligned_paths[fi])
                    else:
                        mm["source"] = mm["timestamp"] = None
                layer = met.meteor_layer(transient, soft)
                lab = labels
                if edge_crop:
                    layer = contract.crop_to_contract(layer, edges)
                    lab = contract.crop_to_contract(labels, edges)
                layer = np.nan_to_num(layer, nan=0.0).astype(np.float32)
                meteor_path = out_dir / METEOR_MAP_NAME
                np.save(str(meteor_path), layer)
                np.save(str(out_dir / METEOR_LABELS_NAME), np.asarray(lab, dtype=np.int16))
                (out_dir / METEOR_META_NAME).write_text(
                    json.dumps({"meteors": meteor_list}, indent=2), encoding="utf-8")
                srcs = ", ".join(str(m.get("source")) for m in meteor_list)
                log(f"Meteor layer: {meteor_path}  ({len(meteor_list)} trail(s): {srcs})")
        except Exception as e:                           # optional, never fatal
            log(f"  (meteor layer skipped: {e})")
            meteor_path = None
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
            "noise_map_path": str(snr_path) if snr_path else None,
            "meteor_layer_path": str(meteor_path) if meteor_path else None,
            "meteors": meteor_list,
            "registered_with": "astroalign" if reg.astroalign_available() else "fft-translation"}
