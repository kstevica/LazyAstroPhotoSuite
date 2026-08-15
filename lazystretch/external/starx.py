"""StarNet wrapper — star removal, reduction, and star mask (PLAN §7).

Substitutes StarXTerminator. StarNet operates on **stretched 16-bit** images, so it
runs in the nonlinear stage. The LazyStretch star LOGIC is tool-agnostic numpy:
``stars = orig - starless``, then recombine with a small-star gamma (js:3121 reduceStars),
or use the stars layer as a protect-the-stars mask (js:2658 buildStarMask).

CLI (StarNet++ v2): ``starnet++ <in.tif> <out.tif> [stride]`` — 16-bit TIFF in/out.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .base import ExternalTool
from .shellout import run_image_roundtrip


class StarX(ExternalTool):
    name = "StarXTerminator"
    label = "StarNet"
    executables = ["starnet++", "starnet2", "StarNet", "starnet"]
    env_var = "LAZYSTRETCH_STARNET"

    def __init__(self, path=None, stride: int = 256):
        super().__init__(path)
        self.stride = stride

    def starless(self, img: np.ndarray) -> np.ndarray:
        """Return the starless version of a (stretched) image via StarNet."""
        exe = self.resolve()

        def builder(in_path: str, tmp: str) -> Tuple[list, str]:
            out = str(Path(tmp) / "starless.tif")
            if "++" in Path(exe).name:
                # StarNet++ v1: positional  `starnet++ INPUT OUTPUT [STRIDE]`
                return [exe, in_path, out, str(self.stride)], out
            # StarNet2 (and Misiura-style): flagged  `starnet2 -i INPUT -o OUTPUT -s STRIDE`
            return [exe, "-i", in_path, "-o", out, "-s", str(self.stride)], out

        return run_image_roundtrip(builder, img, in_ext=".tif", bit_depth=16)

    def reduce_stars(self, img: np.ndarray, star_level: float,
                     small_stars: float = 0.0) -> np.ndarray:
        """Shrink stars: starless + star_level * shaped(orig - starless) (js:3121)."""
        return star_recombine(img, self.starless(img), star_level, small_stars)

    def star_mask(self, img: np.ndarray) -> np.ndarray:
        """A 2-D star mask (white on stars) = lightness of (orig - starless)."""
        s = self.starless(img)
        stars = np.clip(np.asarray(img, dtype=np.float64) - s, 0.0, 1.0)
        m = stars if stars.ndim == 2 else stars.mean(axis=2)
        peak = float(m.max())
        return m / peak if peak > 1e-8 else m

    def remove_stars(self, img: np.ndarray):
        """Full star removal (js:4147-4214): return ``(starless, stars)``.

        ``stars = max(0, orig - starless)`` — the additive layer that recombines back
        to the original (``starless + stars == orig``). The result goes starless; the
        caller keeps the stars layer as a separate output (``<name>_stars``).
        """
        a = np.asarray(img, dtype=np.float64)
        starless = self.starless(a)
        stars = np.clip(a - starless, 0.0, 1.0)
        return starless, stars

    def run(self, img, star_level: float = 0.5, small_stars: float = 0.0, **_) -> Tuple[np.ndarray, bool]:
        if not self.is_available():
            return np.asarray(img, dtype=np.float64).copy(), False
        return self.reduce_stars(img, star_level, small_stars), True


def star_recombine(original: np.ndarray, starless: np.ndarray,
                   star_level: float, small_stars: float = 0.0) -> np.ndarray:
    """Recombine a reduced-star image from an original + a starless layer (pure numpy).

    ``stars = original - starless``; ``result = starless + star_level * shaped(stars)``
    where ``shaped`` applies the small-star gamma ``sign(d)*|d|^(1+small_stars)``.
    """
    o = np.asarray(original, dtype=np.float64)
    s = np.asarray(starless, dtype=np.float64)
    d = o - s
    if small_stars > 0.0:
        d = np.sign(d) * np.abs(d) ** (1.0 + small_stars)
    return np.clip(s + star_level * d, 0.0, 1.0)
