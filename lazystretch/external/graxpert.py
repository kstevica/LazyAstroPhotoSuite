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

    def background_extraction(self, img: np.ndarray, smoothing: float = 0.1,
                              max_dim: int = 1600) -> np.ndarray:
        """Model + subtract the background/gradient (GradientCorrection substitute).

        The background is low-frequency, so on a big master (20-40 MP the GraXpert AI takes
        minutes and looks hung) we run GraXpert on a downsampled copy, recover its background
        *model* (input − corrected), upsample it, and subtract at full resolution — the same
        bounded-cost trick as the ABE path, seconds instead of minutes.
        """
        a = np.asarray(img, dtype=np.float64)
        H, W = a.shape[0], a.shape[1]
        run = lambda x: run_image_roundtrip(
            self._cmd("background-extraction",
                      ["-correction", "Subtraction", "-smoothing", f"{smoothing:.3f}"]),
            x, in_ext=".fits")
        if max(H, W) <= max_dim:
            return run(a)
        from scipy.ndimage import zoom
        f = max_dim / float(max(H, W))
        zin = (f, f, 1) if a.ndim == 3 else (f, f)
        small = zoom(a, zin, order=1)
        model_small = small - np.asarray(run(small), dtype=np.float64)      # the background model
        zf = (H / model_small.shape[0], W / model_small.shape[1]) + ((1,) if a.ndim == 3 else ())
        return np.clip(a - zoom(model_small, zf, order=1), 0.0, 1.0)

    def denoise(self, img: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Denoise (NoiseXTerminator substitute)."""
        return run_image_roundtrip(
            self._cmd("denoising", ["-strength", f"{strength:.3f}"]),
            img, in_ext=".fits")

    def run(self, img, **kwargs) -> Tuple[np.ndarray, bool]:
        if not self.is_available():
            return np.asarray(img, dtype=np.float64).copy(), False
        return self.denoise(img, **kwargs), True
