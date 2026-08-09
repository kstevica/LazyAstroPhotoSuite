"""Image I/O — the layer PixInsight gave LazyStretch for free (PLAN §8).

Reads FITS / TIFF / PNG (and XISF where the ``xisf`` package is present) and writes
FITS / TIFF / PNG. Everything is normalised to float64 in ``[0, 1]``, alpha-stripped,
mono ``(H, W)`` or RGB ``(H, W, 3)`` — the contract the rest of the package expects
(PLAN §2). FITS RGB is stored planes-first ``(3, H, W)`` by convention and is
transposed on load / restored on save.

Linear masters (WBPP float FITS) arrive already in ``[0, 1]``; integer formats are
divided by their type max. Non-linear inputs (finished JPEG/TIFF for the
"input already stretched" polish mode, PLAN §3) load the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

FITS_EXT = {".fit", ".fits", ".fts"}
TIFF_EXT = {".tif", ".tiff"}
PNG_EXT = {".png"}
XISF_EXT = {".xisf"}
RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".dng", ".orf", ".rw2",
           ".pef", ".srw", ".raw"}


@dataclass
class LoadedImage:
    """A loaded image plus the metadata the pipeline may need."""

    data: np.ndarray                      # float64 [0,1], (H,W) or (H,W,3)
    header: Dict[str, object] = field(default_factory=dict)  # FITS keywords (upper-cased)
    path: Optional[str] = None
    source_dtype: str = "float64"
    source_format: str = ""

    @property
    def is_color(self) -> bool:
        return self.data.ndim == 3 and self.data.shape[2] == 3

    def keyword(self, name: str):
        """FITS keyword lookup (case-insensitive), or ``None``."""
        return self.header.get(name.upper())


# --------------------------------------------------------------------------- load


def _norm_dtype(a: np.ndarray) -> np.ndarray:
    """Return float64 in [0,1]: integers scaled by type max, floats clipped."""
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        out = a.astype(np.float64) / float(info.max)
    else:
        out = a.astype(np.float64)
    return np.clip(out, 0.0, 1.0)


def _canonicalize(a: np.ndarray, planes_first: bool) -> np.ndarray:
    """Coerce raw array to mono (H,W) or RGB (H,W,3), alpha-stripped."""
    a = np.asarray(a)
    if a.ndim == 3 and planes_first and a.shape[0] in (1, 3, 4):
        a = np.moveaxis(a, 0, -1)          # (C,H,W) -> (H,W,C)  (FITS convention)
    if a.ndim == 3:
        if a.shape[2] == 1:
            a = a[..., 0]                  # mono stored as (H,W,1)
        elif a.shape[2] >= 4:
            a = a[..., :3]                 # drop alpha / extra planes
    return a


def _load_fits(path: Path) -> LoadedImage:
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        hdu = next((h for h in hdul if getattr(h, "data", None) is not None), hdul[0])
        raw = hdu.data
        header = {str(k).upper(): v for k, v in hdu.header.items() if k}
    if raw is None:
        raise ValueError(f"{path}: FITS has no image data")
    data = _norm_dtype(_canonicalize(raw, planes_first=True))
    return LoadedImage(data=data, header=header, path=str(path),
                       source_dtype=str(raw.dtype), source_format="fits")


def _load_tiff(path: Path) -> LoadedImage:
    import tifffile

    raw = tifffile.imread(str(path))
    data = _norm_dtype(_canonicalize(raw, planes_first=False))
    return LoadedImage(data=data, path=str(path),
                       source_dtype=str(raw.dtype), source_format="tiff")


def _load_png(path: Path) -> LoadedImage:
    import imageio.v3 as iio

    raw = iio.imread(str(path))
    data = _norm_dtype(_canonicalize(raw, planes_first=False))
    return LoadedImage(data=data, path=str(path),
                       source_dtype=str(raw.dtype), source_format="png")


def _load_xisf(path: Path) -> LoadedImage:
    try:
        from xisf import XISF
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "reading XISF needs the 'xisf' package (pip install xisf)"
        ) from exc
    x = XISF(str(path))
    raw = x.read_image(0)  # (H, W, C) or (H, W)
    header = {}
    try:
        meta = x.get_images_metadata()[0].get("FITSKeywords", {})
        header = {str(k).upper(): (v[0].get("value") if isinstance(v, list) and v else v)
                  for k, v in meta.items()}
    except Exception:
        pass
    data = _norm_dtype(_canonicalize(raw, planes_first=False))
    return LoadedImage(data=data, header=header, path=str(path),
                       source_dtype=str(raw.dtype), source_format="xisf")


def _load_raw(path: Path) -> LoadedImage:
    """Decode a camera raw to linear RGB via rawpy — VNG demosaic, no white balance.

    Mirrors the .js ``convertRaws`` hint ('vng no-white-balance'): a linear (gamma 1),
    auto-bright-off, WB-neutral 16-bit RGB so calibration/stretch see the sensor's linear
    signal. Bursts of raws therefore behave like linear FITS/XISF frames.
    """
    try:
        import rawpy
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "reading camera raws needs the 'rawpy' package (pip install rawpy)"
        ) from exc
    with rawpy.imread(str(path)) as raw:
        try:
            algo = rawpy.DemosaicAlgorithm.VNG
        except Exception:  # pragma: no cover - older rawpy
            algo = None
        kwargs = dict(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False,
                      use_auto_wb=False, user_wb=[1.0, 1.0, 1.0, 1.0],
                      output_bps=16, four_color_rgb=False)
        if algo is not None:
            kwargs["demosaic_algorithm"] = algo
        rgb = raw.postprocess(**kwargs)                 # (H, W, 3) uint16, linear
    data = _norm_dtype(_canonicalize(rgb, planes_first=False))
    return LoadedImage(data=data, path=str(path),
                       source_dtype=str(rgb.dtype), source_format="raw")


def load_image(path: "str | Path") -> LoadedImage:
    """Load an image from FITS / TIFF / PNG / XISF / camera-raw into the normalised contract."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in FITS_EXT:
        return _load_fits(p)
    if ext in TIFF_EXT:
        return _load_tiff(p)
    if ext in PNG_EXT:
        return _load_png(p)
    if ext in XISF_EXT:
        return _load_xisf(p)
    if ext in RAW_EXT:
        return _load_raw(p)
    raise ValueError(f"unsupported image extension: {ext!r} ({p.name})")


# --------------------------------------------------------------------------- save


def _to_planes_first(a: np.ndarray) -> np.ndarray:
    return np.moveaxis(a, -1, 0) if a.ndim == 3 else a


def save_image(path: "str | Path", data: np.ndarray, *, bit_depth: int = 16,
               header: Optional[Dict[str, object]] = None) -> str:
    """Write ``data`` (float [0,1]) to FITS / TIFF / PNG chosen by extension.

    ``bit_depth`` applies to TIFF/PNG (8 or 16). FITS is always written float32.
    XISF write is intentionally unsupported (PLAN §11) — use FITS/TIFF instead.
    """
    p = Path(path)
    ext = p.suffix.lower()
    a = np.clip(np.asarray(data, dtype=np.float64), 0.0, 1.0)

    if ext in FITS_EXT:
        from astropy.io import fits

        hdu = fits.PrimaryHDU(_to_planes_first(a).astype(np.float32))
        if header:
            for k, v in header.items():
                try:
                    hdu.header[k[:8]] = v
                except Exception:
                    pass
        hdu.writeto(str(p), overwrite=True)
    elif ext in TIFF_EXT:
        import tifffile

        if bit_depth == 8:
            arr = np.rint(a * 255.0).astype(np.uint8)
        else:
            arr = np.rint(a * 65535.0).astype(np.uint16)
        tifffile.imwrite(str(p), arr)
    elif ext in PNG_EXT:
        import imageio.v3 as iio

        if bit_depth == 16:
            arr = np.rint(a * 65535.0).astype(np.uint16)
        else:
            arr = np.rint(a * 255.0).astype(np.uint8)
        iio.imwrite(str(p), arr)
    elif ext in XISF_EXT:
        raise ValueError(
            f"XISF write is not supported ({p.name}); XISF is read-only (PLAN §11). "
            "Save as .fits or .tif instead.")
    else:
        raise ValueError(f"unsupported output extension: {ext!r} ({p.name})")
    return str(p)
