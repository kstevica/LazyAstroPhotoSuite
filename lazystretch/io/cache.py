"""Decode cache for slow-to-read frames (camera raws).

Camera raws cost seconds each to demosaic and every frame is opened more than once (grade +
register, or calibrate + measure). On first read a raw is decoded and written to a 16-bit
TIFF beside the dataset (``<cache_dir>/<stem>_<key>.tif``); later reads load that in a blink.
The 16-bit TIFF is a LOSSLESS round-trip of rawpy's 16-bit output. Fast formats (FITS / TIFF
/ PNG / XISF) skip the cache. The key includes the source mtime + size, so editing or
replacing a frame invalidates its cache automatically.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .image_io import RAW_EXT, load_image, save_image


def _noop(_m: str) -> None:
    pass


def _cache_key(path: str) -> str:
    st = os.stat(path)
    raw = f"{Path(path).resolve()}|{st.st_mtime_ns}|{st.st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def cache_path(path: str, cache_dir: "str | Path") -> Path:
    return Path(cache_dir) / f"{Path(path).stem}_{_cache_key(path)}.tif"


def cached_load(path: str, cache_dir: "str | Path", *, enabled: bool = True,
                log: Callable[[str], None] = _noop) -> Optional[np.ndarray]:
    """Load ``path`` as float64 [0,1]; cache raws under ``cache_dir``. None on failure.

    Raw decodes are slow, so each one is logged (cache hit vs. the expensive decode) — a
    silent 30-raw pass otherwise looks stuck.
    """
    ext = Path(path).suffix.lower()
    name = Path(path).name
    if not enabled or ext not in RAW_EXT:
        try:
            return load_image(path).data
        except Exception:
            return None
    try:
        cp = cache_path(path, cache_dir)
    except OSError:
        cp = None
    if cp is not None and cp.exists():
        try:
            data = load_image(str(cp)).data
            log(f"   {name}: loaded from cache")
            return data
        except Exception:
            pass                                    # corrupt cache -> re-decode
    log(f"   {name}: decoding raw…")
    try:
        data = load_image(path).data                # the expensive rawpy decode
    except Exception:
        return None
    if cp is not None:
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            save_image(str(cp), data, bit_depth=16)
            log(f"   {name}: decoded + cached")
        except Exception:
            pass                                    # caching is best-effort
    return data
