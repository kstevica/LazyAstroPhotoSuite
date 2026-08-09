"""LazyMoonSun orchestration — folder → stacked master → finished image.

Ties the engine (:mod:`.stack`) and finish (:mod:`.finish`) to disk I/O: scans a burst
folder, loads frames through the shared image reader, saves the master as FITS carrying the
``LSN*`` provenance keywords the rest of the suite reads, and appends ``mp_report.txt`` for
multi-point runs. Camera raws need ``rawpy`` (optional); without it they are reported and
skipped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ..io.cache import cached_load
from ..io.image_io import RAW_EXT as RAW_EXTS, load_image, save_image
from . import finish as finish_mod, stack

FRAME_EXTS = {".xisf", ".fits", ".fit", ".fts", ".tif", ".tiff", ".png"}
MASTER_NAME = "burst_master.fits"
MP_MASTER_NAME = "burst_master_mp.fits"
_RAWPY_OK = None


def _rawpy_available() -> bool:
    """rawpy is optional; camera raws are only loadable when it's installed."""
    global _RAWPY_OK
    if _RAWPY_OK is None:
        try:
            import rawpy  # noqa: F401
            _RAWPY_OK = True
        except Exception:
            _RAWPY_OK = False
    return _RAWPY_OK


def _noop(_m: str) -> None:
    pass


def find_frames(folder: str) -> dict:
    """List readable burst frames in ``folder`` (camera raws included when rawpy is present)."""
    p = Path(folder)
    frames, raws = [], []
    for f in sorted(p.iterdir()) if p.is_dir() else []:
        ext = f.suffix.lower()
        if ext in FRAME_EXTS:
            frames.append(str(f))
        elif ext in RAW_EXTS:
            (frames if _rawpy_available() else raws).append(str(f))
    return {"frames": frames, "raws": raws}


def _out_dir(folder: str) -> Path:
    d = Path(folder) / "lazysun"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_loader(folder: str, log: Callable[[str], None] = _noop) -> Callable[[str], Optional[np.ndarray]]:
    """A frame loader that caches decoded raws under <folder>/lazysun/cache (and logs them)."""
    cache_dir = Path(folder) / "lazysun" / "cache"
    return lambda path: cached_load(path, cache_dir, log=log)


def stack_burst_folder(folder: str, params, *, log: Callable[[str], None] = _noop) -> Optional[dict]:
    """Global engine over a folder → saves lazysun/burst_master.fits. Returns the result."""
    scan = find_frames(folder)
    if scan["raws"]:
        log(f"Note: {len(scan['raws'])} camera raw(s) skipped — install 'rawpy' to decode them.")
    frames = scan["frames"]
    if len(frames) < 2:
        log(f"Found {len(frames)} readable frame(s) — a burst needs at least 2.")
        return None
    log(f"Burst: {folder} ({len(frames)} frames).")
    res = stack.stack_burst(frames, _make_loader(folder, log), keep=params.keep, log=log)
    if res is None:
        return None
    master_path = _out_dir(folder) / MASTER_NAME
    save_image(str(master_path), res["master"], bit_depth=16,
               header={"LSNFRAME": res["n_stacked"]})
    res["master_path"] = str(master_path)
    log(f"Master: {master_path}")
    return res


def stack_multipoint_folder(folder: str, params, *,
                            log: Callable[[str], None] = _noop) -> Optional[dict]:
    """Multi-point engine → lazysun/burst_master_mp.fits + appended mp_report.txt."""
    master_path = _out_dir(folder) / MASTER_NAME
    if not master_path.exists():
        log("Multi-point needs the global master — run Stack burst (global) first.")
        return None
    master = load_image(str(master_path)).data
    scan = find_frames(folder)
    frames = scan["frames"]
    if len(frames) < 2:
        log(f"Found {len(frames)} frame(s) — a burst needs at least 2.")
        return None
    log(f"Multi-point: {folder} ({len(frames)} frames). Reference: {MASTER_NAME}")
    res = stack.stack_multipoint(frames, _make_loader(folder, log), master, params, log=log)
    if res is None:
        return None
    mp_path = _out_dir(folder) / MP_MASTER_NAME
    save_image(str(mp_path), res["master"], bit_depth=16,
               header={"LSNFRAME": int(round(res["mean_kept"])), "LSNMPAP": res["n_aps"]})
    res["master_path"] = str(mp_path)
    _append_report(_out_dir(folder) / "mp_report.txt", res["report"], log)
    log(f"Multi-point master: {mp_path}")
    return res


def _append_report(path: Path, block: str, log: Callable[[str], None]) -> None:
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = f"==== LazyMoonSun MP run {stamp} ====\n"
    prev = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(prev + header + block + "\n", encoding="utf-8")
    log(f"Report appended: {path}")


def finish_array(img: np.ndarray, params, *, log: Callable[[str], None] = _noop) -> Optional[np.ndarray]:
    """Run the finish on an in-memory image; returns the finished array or None."""
    res = finish_mod.finish(img, params, log=log)
    return res["image"] if res is not None else None
