"""Registration (StarAlignment equivalent).

Uses ``astroalign`` when installed — asterism-matched affine alignment (rotation + shift +
scale), the faithful stand-in for StarAlignment. Without it, falls back to the LazyMoonSun
FFT phase-correlation engine, which is TRANSLATION ONLY: fine for short/untracked sets but it
cannot correct field rotation, so it warns. (astroalign has no polynomial distortion model
either — wide-lens distortion correction is the one thing neither path reproduces.)
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from ..moonsun import register as fftreg


def _noop(_m: str) -> None:
    pass


def astroalign_available() -> bool:
    try:
        import astroalign  # noqa: F401
        return True
    except Exception:
        return False


def _align_astroalign(frame: np.ndarray, ref: np.ndarray):
    import astroalign
    color = frame.ndim == 3
    src = frame.mean(axis=2) if color else frame
    tgt = ref.mean(axis=2) if color else ref
    transform, _ = astroalign.find_transform(src, tgt)
    if color:
        out = np.stack([astroalign.apply_transform(transform, frame[..., c], ref[..., c])[0]
                        for c in range(frame.shape[2])], axis=-1)
    else:
        out, _ = astroalign.apply_transform(transform, frame, ref)
    return np.clip(out, 0.0, 1.0)


def register(frames: List[np.ndarray], reference: int, *,
             log: Callable[[str], None] = _noop) -> Tuple[List[np.ndarray], List[int]]:
    """Align every frame to ``frames[reference]``. Returns (aligned, kept-indices)."""
    ref = np.asarray(frames[reference], dtype=np.float64)
    use_aa = astroalign_available()
    if use_aa:
        log("Registration: astroalign (asterism-matched affine).")
    else:
        log("Registration: astroalign not installed — FFT translation-only fallback "
            "(no rotation correction; install astroalign for tracked/rotated sets).")
        ref_wk, _ = fftreg.working(ref)
        ref_fft = fftreg.make_ref(fftreg.gradient(ref_wk), ref_wk.shape[1], ref_wk.shape[0])

    aligned, kept = [], []
    for i, f in enumerate(frames):
        f = np.asarray(f, dtype=np.float64)
        if i == reference:
            aligned.append(f)
            kept.append(i)
            continue
        try:
            if use_aa:
                aligned.append(_align_astroalign(f, ref))
            else:
                wk, k = fftreg.working(f)
                dx, dy = fftreg.measure_against(ref_fft, fftreg.gradient(wk),
                                                wk.shape[1], wk.shape[0])
                aligned.append(fftreg.apply_shift(f, -dx / k, -dy / k))
            kept.append(i)
        except Exception as e:
            log(f"  frame {i} failed to register ({e}) — dropped")
    return aligned, kept
