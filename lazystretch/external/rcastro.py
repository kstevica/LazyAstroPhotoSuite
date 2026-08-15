"""RC-Astro standalone CLI wrappers — BlurX / StarX / NoiseX (the *real* products).

RC-Astro shipped a stand-alone command-line tool (``rc-astro``, v1.0.0, 2025) that
runs BlurXTerminator, StarXTerminator and NoiseXTerminator outside PixInsight. When
that binary is installed **and licensed**, these are the genuine article and beat the
open-source substitutes the pipeline otherwise falls back to (classical Richardson-Lucy
for BlurX, StarNet for StarX, DeepSNR/GraXpert/MMT for NoiseX). Absence is invisible:
detection returns ``None`` and the pipeline uses the same fallbacks as before.

CLI shape (verified against the real ``rc-astro`` v1.1.3 ``--help``, Aug 2026)::

    rc-astro --no-banner bxt <in> [--sharpen-stars s] [--sharpen-nonstellar s] \
             --output <out.tif> --overwrite
    rc-astro --no-banner sxt <in> --output <out.tif> --overwrite   # starless only
    rc-astro --no-banner nxt <in> [--denoise a] --output <out.tif> --overwrite

Notes from the real help: ``--output`` is a filename *or* directory and the format is
taken from the extension, so we hand it an explicit ``.tif`` path (a bare dir has no
extension); the tool refuses to overwrite by default, hence ``--overwrite``; ``sxt``
writes the starless by default (``--stars`` would add a stars-only image beside it — we
don't need it, the stars layer is ``orig - starless`` in numpy). Every tunable is left
at the tool's own default (bxt ss=0.5/sn=1.0, nxt dn=0.9) unless the caller passes one,
so no flag *value* is guessed.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base import ExternalTool
from .shellout import run_image_roundtrip
from .starx import StarX

_ENV_VAR = "LAZYSTRETCH_RCASTRO"
_EXES = ["rc-astro", "RC-Astro", "rc-astro.exe"]
# The installer usually puts rc-astro on PATH; cover the common fixed locations too.
_APP_PATHS = [
    "/usr/local/bin/rc-astro",
    "/opt/homebrew/bin/rc-astro",
    "/Applications/RC-Astro/rc-astro",
    str(Path.home() / "Applications/RC-Astro/rc-astro"),
    str(Path.home() / ".local/bin/rc-astro"),
    r"C:\Program Files\RC-Astro\rc-astro.exe",
]


def resolve_rcastro(path: Optional[str] = None) -> Optional[str]:
    """Locate the ``rc-astro`` binary: explicit path -> env var -> PATH -> app dirs."""
    if path and Path(path).exists():
        return path
    env = os.environ.get(_ENV_VAR)
    if env and Path(env).exists():
        return env
    for exe in _EXES:
        found = shutil.which(exe)
        if found:
            return found
    for p in _APP_PATHS:
        if Path(p).exists():
            return p
    return None


def _run_rcastro(subcmd: str, img: np.ndarray, extra: Optional[List[str]] = None, *,
                 path: Optional[str] = None, in_ext: str = ".tif",
                 bit_depth: int = 16) -> np.ndarray:
    """Invoke ``rc-astro <subcmd> <in> [extra] --output <scratch>`` and read the result."""
    exe = resolve_rcastro(path)
    if exe is None:
        raise RuntimeError("rc-astro binary not found")

    def builder(in_path: str, tmp: str) -> Tuple[list, str]:
        out = str(Path(tmp) / f"rcout{in_ext}")          # explicit .tif; ext picks format
        argv = ([exe, "--no-banner", subcmd, in_path]
                + list(extra or []) + ["--output", out, "--overwrite"])
        return argv, out

    return run_image_roundtrip(builder, img, in_ext=in_ext, bit_depth=bit_depth)


class _RCTool(ExternalTool):
    """Shared detection for the three ``rc-astro`` sub-commands (one binary)."""

    executables = list(_EXES)
    env_var = _ENV_VAR

    def resolve(self) -> Optional[str]:
        return resolve_rcastro(self._path)


class BlurX(_RCTool):
    """BlurXTerminator — ML deconvolution / sharpening (the real BlurX)."""

    name = "BlurXTerminator"
    label = "BlurXTerminator"

    def deconvolve(self, img: np.ndarray, sharpen_stars: Optional[float] = None,
                   sharpen_nonstellar: Optional[float] = None) -> np.ndarray:
        """Deconvolve a LINEAR image. Amounts default to BlurX's own settings."""
        extra: List[str] = []
        if sharpen_stars is not None:
            extra += ["--sharpen-stars", f"{sharpen_stars:.3f}"]
        if sharpen_nonstellar is not None:
            extra += ["--sharpen-nonstellar", f"{sharpen_nonstellar:.3f}"]
        return _run_rcastro("bxt", img, extra, path=self._path)

    def run(self, img, **kw) -> Tuple[np.ndarray, bool]:
        if not self.is_available():
            return np.asarray(img, dtype=np.float64).copy(), False
        return self.deconvolve(img, **kw), True


class NoiseX(_RCTool):
    """NoiseXTerminator — ML noise reduction (the real NoiseX)."""

    name = "NoiseXTerminator"
    label = "NoiseXTerminator"

    def denoise(self, img: np.ndarray, amount: Optional[float] = None) -> np.ndarray:
        """Denoise. ``amount`` defaults to NoiseX's own setting when None."""
        extra = ["--denoise", f"{amount:.3f}"] if amount is not None else []
        return _run_rcastro("nxt", img, extra, path=self._path)

    def run(self, img, **kw) -> Tuple[np.ndarray, bool]:
        if not self.is_available():
            return np.asarray(img, dtype=np.float64).copy(), False
        return self.denoise(img, **kw), True


class RCStarX(StarX):
    """StarXTerminator via the rc-astro CLI. Reuses StarX's tool-agnostic numpy star
    logic (``reduce_stars`` / ``star_mask`` / ``remove_stars`` all sit on ``starless``);
    only detection + the starless call itself route through ``rc-astro sxt``."""

    name = "StarXTerminator"
    label = "StarXTerminator"
    executables = list(_EXES)
    env_var = _ENV_VAR

    def resolve(self) -> Optional[str]:
        return resolve_rcastro(self._path)

    def starless(self, img: np.ndarray) -> np.ndarray:
        return _run_rcastro("sxt", img, path=self._path, in_ext=".tif", bit_depth=16)
