"""Plate solving — the hard non-portable seam, behind a stable result contract.

The output :class:`SolveResult` mirrors PI's ``AstrometryEngine.makeResult`` (PLAN §2),
so everything downstream (catalog search, classification) is solver-agnostic.

Two backends:
  * **fast path** — read an existing WCS from the FITS header via ``astropy.wcs`` (the
    common case; no external tool).
  * **ASTAP** — shell out to the ``astap`` CLI when installed (offline, self-contained
    star DBs). Feature-detected; absent -> returns an unsolved result gracefully.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..io.image_io import LoadedImage, load_image


@dataclass(frozen=True)
class SolveResult:
    solved: bool
    fromExisting: bool = False        # solved from an existing header WCS (vs a fresh solve)
    ra: float = float("nan")          # field-centre RA, degrees
    dec: float = float("nan")         # field-centre Dec, degrees
    resolution: float = float("nan")  # degrees / pixel
    pixScaleAsec: float = float("nan")  # arcsec / pixel
    focal: float = float("nan")       # mm (needs pixel size); NaN if unknown
    width: int = 0
    height: int = 0
    fovDeg: float = float("nan")      # diagonal field of view, degrees

    @property
    def fov_radius_deg(self) -> float:
        return self.fovDeg / 2.0 if math.isfinite(self.fovDeg) else float("nan")


def _dims(img: LoadedImage):
    h, w = img.data.shape[0], img.data.shape[1]
    return int(w), int(h)


def _focal_from_pixel_size(header, pix_scale_asec: float) -> float:
    """focal_mm = 206.265 * pixel_size_um / pixel_scale_arcsec, if the header has it."""
    for key in ("XPIXSZ", "PIXSIZE", "PIXSIZE1"):
        v = header.get(key)
        if v is not None:
            try:
                px_um = float(v)
                if px_um > 0 and pix_scale_asec > 0:
                    return 206.265 * px_um / pix_scale_asec
            except (TypeError, ValueError):
                pass
    return float("nan")


def solve_from_header(img: LoadedImage) -> Optional[SolveResult]:
    """Fast path: build a result from an existing celestial WCS, or None if absent."""
    try:
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
    except ImportError:  # pragma: no cover
        return None
    if not img.header:
        return None
    try:
        wcs = WCS(img.header)
    except Exception:
        return None
    if not (wcs.has_celestial and wcs.naxis >= 2):
        return None

    w, h = _dims(img)
    wcs2 = wcs.celestial
    try:
        scales = proj_plane_pixel_scales(wcs2)      # deg/px per axis
        resolution = float(sum(scales[:2]) / 2.0)
        sky = wcs2.pixel_to_world(w / 2.0, h / 2.0)  # field centre
        ra = float(sky.icrs.ra.deg)
        dec = float(sky.icrs.dec.deg)
    except Exception:
        return None
    if not (math.isfinite(ra) and math.isfinite(dec) and resolution > 0):
        return None

    pix_asec = resolution * 3600.0
    fov = math.sqrt(w * w + h * h) * resolution
    return SolveResult(
        solved=True, fromExisting=True, ra=ra, dec=dec, resolution=resolution,
        pixScaleAsec=pix_asec, focal=_focal_from_pixel_size(img.header, pix_asec),
        width=w, height=h, fovDeg=fov,
    )


def astap_available(astap_path: Optional[str] = None) -> bool:
    return bool(astap_path and Path(astap_path).exists()) or shutil.which("astap") is not None


def solve_with_astap(path: "str | Path", *, astap_path: Optional[str] = None,
                     fov_guess_deg: Optional[float] = None,
                     timeout: int = 120) -> Optional[SolveResult]:
    """Shell out to ASTAP and read the emitted WCS. None if ASTAP is unavailable/fails."""
    exe = astap_path if (astap_path and Path(astap_path).exists()) else shutil.which("astap")
    if not exe:
        return None
    src = Path(path)
    with tempfile.TemporaryDirectory() as td:
        stem = Path(td) / src.stem
        cmd = [exe, "-f", str(src), "-o", str(stem), "-wcs"]
        if fov_guess_deg:
            cmd += ["-fov", f"{fov_guess_deg:.4f}"]
        try:
            subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        except (subprocess.TimeoutExpired, OSError):
            return None
        wcs_file = Path(str(stem) + ".wcs")
        if not wcs_file.exists():
            return None
        # ASTAP writes a FITS-header-style .wcs; parse it with astropy.
        try:
            from astropy.io import fits
            hdr = fits.Header.fromtextfile(str(wcs_file))
        except Exception:
            return None
        img = load_image(path)
        merged = LoadedImage(data=img.data,
                             header={**img.header, **{str(k).upper(): v for k, v in hdr.items()}})
        res = solve_from_header(merged)
        if res is None:
            return None
        return SolveResult(**{**res.__dict__, "fromExisting": False})


def solve(img: LoadedImage, *, astap_path: Optional[str] = None,
          allow_astap: bool = True) -> SolveResult:
    """Solve ``img``: existing WCS first, then ASTAP if available, else unsolved."""
    res = solve_from_header(img)
    if res is not None:
        return res
    if allow_astap and img.path and astap_available(astap_path):
        astap_res = solve_with_astap(img.path, astap_path=astap_path)
        if astap_res is not None:
            return astap_res
    w, h = _dims(img)
    return SolveResult(solved=False, width=w, height=h)
