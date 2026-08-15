"""XTerminator tools — BlurX / NoiseX / StarX inside LazyDevelop.

Each tool prefers the *real* RC-Astro product (the ``rc-astro`` CLI, when installed and
licensed) and falls back to the open-source equivalent otherwise, so the same tool works
on any machine:

* **BlurXTerminator (BX)** — deconvolution → RC-Astro BlurX, else classical Richardson-Lucy.
* **NoiseXTerminator (NX)** — noise reduction → RC-Astro NoiseX, else multiscale-median (MMT).
* **StarXTerminator (SX)** — star removal/reduction → RC-Astro StarX, else StarNet, else the
  morphological star reducer.

The external wrappers feature-detect themselves, so these ops need no external state; in a
hermetic environment (no tools installed) they transparently use the open-source path.
"""
from __future__ import annotations

import numpy as np

from ...external import BlurX, NoiseX, RCStarX, StarX
from ...processes.deconv import richardson_lucy
from ...processes.multiscale import noise_reduction_mmt
from ...processes.starreduce import shrink_stars
from . import Op, ParamSpec, register


def _blurx(img: np.ndarray, p: dict) -> np.ndarray:
    ss = float(p.get("sharpen_stars", 0.5))
    sn = float(p.get("sharpen_nonstellar", 0.9))
    bx = BlurX()
    if bx.is_available():
        try:
            return np.clip(bx.deconvolve(img, sharpen_stars=ss, sharpen_nonstellar=sn),
                           0.0, 1.0)
        except Exception:
            pass                                        # licence/run failure → open-source
    iterations = max(int(round(4 + 26 * sn)), 1)        # RL strength from nonstellar sharpen
    return np.clip(richardson_lucy(img, sigma=1.2, iterations=iterations), 0.0, 1.0)


def _noisex(img: np.ndarray, p: dict) -> np.ndarray:
    strength = float(p.get("denoise", 0.9))
    nx = NoiseX()
    if nx.is_available():
        try:
            return np.clip(nx.denoise(img, amount=strength), 0.0, 1.0)
        except Exception:
            pass
    nr = np.asarray(noise_reduction_mmt(img), dtype=np.float64)   # MMT blended by strength
    a = np.asarray(img, dtype=np.float64)
    return np.clip(a + strength * (nr - a), 0.0, 1.0)


def _best_star_tool():
    """Preferred available star backend: RC-Astro StarX → StarNet → None."""
    for tool in (RCStarX(), StarX()):
        if tool.is_available():
            return tool
    return None


def _starx(img: np.ndarray, p: dict) -> np.ndarray:
    remove = bool(int(p.get("mode", 0)))                # 0 = Reduce, 1 = Remove (starless)
    amount = float(p.get("amount", 0.5))                # reduction strength (1 = full removal)
    small = float(p.get("small", 0.5))
    tool = _best_star_tool()
    if tool is not None:
        try:
            if remove:
                starless, _ = tool.remove_stars(img)
                return np.clip(starless, 0.0, 1.0)
            return np.clip(tool.reduce_stars(img, 1.0 - amount, small), 0.0, 1.0)
        except Exception:
            pass
    amt = 1.0 if remove else amount                     # morphological can't truly go starless
    return np.clip(shrink_stars(img, amt, small=small), 0.0, 1.0)


register(Op(
    "blurx", "BlurXTerminator (BX)", "XTerminator", _blurx, heavy=True,
    params=[
        ParamSpec("sharpen_stars", "Sharpen stars", "float", 0.0, 0.7, 0.5, 2),
        ParamSpec("sharpen_nonstellar", "Sharpen nonstellar", "float", 0.0, 1.0, 0.9, 2),
    ],
    tooltip="Deconvolution / sharpening. Uses RC-Astro BlurXTerminator if the rc-astro CLI "
            "is installed, otherwise classical Richardson-Lucy. Note: BlurX is designed for "
            "LINEAR data, but LazyDevelop works on the already-stretched image, so this "
            "sharpens in the nonlinear domain (its ideal slot is the pipeline's linear "
            "deconvolution step).",
))

register(Op(
    "noisex", "NoiseXTerminator (NX)", "XTerminator", _noisex, heavy=True,
    params=[ParamSpec("denoise", "Denoise", "float", 0.0, 1.0, 0.9, 2)],
    tooltip="Noise reduction. Uses RC-Astro NoiseXTerminator if installed, otherwise the "
            "multiscale-median (MMT) denoiser blended by the same amount.",
))

register(Op(
    "starx", "StarXTerminator (SX)", "XTerminator", _starx, heavy=True,
    params=[
        ParamSpec("mode", "Mode", "choice", 0, 1, 0, 0,
                  choices=["Reduce", "Remove (starless)"]),
        ParamSpec("amount", "Reduction", "float", 0.0, 1.0, 0.5, 2),
        ParamSpec("small", "Bias to small stars", "float", 0.0, 1.0, 0.5, 2),
    ],
    tooltip="Star removal / reduction. Uses RC-Astro StarXTerminator if installed, else "
            "StarNet, else the morphological star reducer. 'Remove' needs an AI tool for a "
            "true starless result.",
))
