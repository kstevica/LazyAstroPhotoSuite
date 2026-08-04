"""GraXpert wrapper — background extraction + denoising (PLAN §7).

Substitutes GradientCorrection (background) and NoiseXTerminator (denoise) via the
GraXpert CLI, feature-detected. GraXpert has **no** deconvolution CLI, so BlurX has no
GraXpert substitute (deconvolution falls to a local classical method — see
``processes.deconv``).

CLI (GraXpert 3.x): ``graxpert <in.fits> -cli -cmd background-extraction|denoising
-output <name> [-smoothing s] [-strength s] [-gpu true]`` — output written as
``<name>.fits``. Verified against the GraXpert README (2025).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .base import ExternalTool
from .shellout import run_image_roundtrip


class GraXpert(ExternalTool):
    name = "GraXpert"
    executables = ["graxpert", "GraXpert", "GraXpert-win64.exe"]
    env_var = "LAZYSTRETCH_GRAXPERT"
    # GraXpert ships as an app bundle / installer, not usually on PATH — check the
    # standard macOS/Windows/Linux install locations too.
    _APP_PATHS = [
        "/Applications/GraXpert.app/Contents/MacOS/GraXpert",
        str(Path.home() / "Applications/GraXpert.app/Contents/MacOS/GraXpert"),
        "/opt/GraXpert/GraXpert",
    ]

    def __init__(self, path=None, gpu: bool = True):
        super().__init__(path)
        self.gpu = gpu

    def resolve(self):
        found = super().resolve()
        if found:
            return found
        for p in self._APP_PATHS:
            if Path(p).exists():
                return p
        return None

    def _cmd(self, cmd: str, extra):
        exe = self.resolve()

        def builder(in_path: str, tmp: str) -> Tuple[list, str]:
            out = str(Path(tmp) / "gx_out")
            argv = [exe, in_path, "-cli", "-cmd", cmd, "-output", out]
            if self.gpu:
                argv += ["-gpu", "true"]
            argv += extra
            return argv, out + ".*"

        return builder

    def background_extraction(self, img: np.ndarray, smoothing: float = 0.1) -> np.ndarray:
        """Model + subtract the background/gradient (GradientCorrection substitute)."""
        return run_image_roundtrip(
            self._cmd("background-extraction",
                      ["-correction", "Subtraction", "-smoothing", f"{smoothing:.3f}"]),
            img, in_ext=".fits")

    def denoise(self, img: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Denoise (NoiseXTerminator substitute)."""
        return run_image_roundtrip(
            self._cmd("denoising", ["-strength", f"{strength:.3f}"]),
            img, in_ext=".fits")

    def run(self, img, **kwargs) -> Tuple[np.ndarray, bool]:
        if not self.is_available():
            return np.asarray(img, dtype=np.float64).copy(), False
        return self.denoise(img, **kwargs), True
