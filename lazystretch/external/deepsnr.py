"""DeepSNR wrapper — deep-learning noise reduction (PLAN §7).

DeepSNR (Nikita Misiura, the StarNet author) is a neural-net denoiser for **stretched**
images — a strong NoiseXTerminator substitute. Same CLI family as StarNet2::

    deepsnr -i input.tif -o output.tif -s <stride> [-m <model 1|2>]

Model 2 (default) handles mono + RGB; model 1 needs RGB. 16-bit TIFF in/out.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .base import ExternalTool
from .shellout import run_image_roundtrip


class DeepSNR(ExternalTool):
    name = "DeepSNR"
    executables = ["deepsnr", "DeepSNR"]
    env_var = "LAZYSTRETCH_DEEPSNR"

    def __init__(self, path=None, stride: int = 256, model: int = 2):
        super().__init__(path)
        self.stride = stride
        self.model = model

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """Denoise a (stretched) image via DeepSNR."""
        exe = self.resolve()

        def builder(in_path: str, tmp: str) -> Tuple[list, str]:
            out = str(Path(tmp) / "denoised.tif")
            return [exe, "-i", in_path, "-o", out, "-s", str(self.stride),
                    "-m", str(self.model), "-q"], out

        return run_image_roundtrip(builder, img, in_ext=".tif", bit_depth=16)

    def run(self, img, **_) -> Tuple[np.ndarray, bool]:
        if not self.is_available():
            return np.asarray(img, dtype=np.float64).copy(), False
        return self.denoise(img), True
