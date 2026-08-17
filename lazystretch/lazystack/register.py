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
from scipy.ndimage import shift as _nd_shift

from ..moonsun import register as fftreg


def _noop(_m: str) -> None:
    pass


def astroalign_available() -> bool:
    try:
        import astroalign  # noqa: F401
        return True
    except Exception:
        return False


def _warp_fine(frame: np.ndarray, transform, out_shape, *, scale: float,
               order: int = 3) -> np.ndarray:
    """Warp ``frame`` ONCE onto a scaled grid (drizzle-lite): compose the astroalign
    similarity transform with the grid scale, so no second interpolation ever runs.
    ``scale`` is the REFERENCE-grid factor (the transform maps source→reference coords,
    so the scale that follows it must be the reference's, not the frame's — a cropped
    frame still lands correctly on the common canvas). The non-covered border is marked
    NaN (no-data), like the astroalign path."""
    from skimage.transform import AffineTransform, warp
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1.0]])
    fine = AffineTransform(matrix=S @ np.asarray(transform.params, dtype=np.float64))
    color = frame.ndim == 3
    ones = np.ones(frame.shape[:2], dtype=np.float64)
    # order=1 + a strict threshold flags the border fringe whose cubic support straddles
    # the source edge (blended toward cval=0, i.e. darker than sky) as no-data too.
    fp = warp(ones, inverse_map=fine.inverse, output_shape=out_shape[:2],
              order=1, cval=0.0, preserve_range=True) < 0.999    # True == no source data
    chans = []
    for c in range(frame.shape[2]) if color else [None]:
        img = frame[..., c] if color else frame
        w = warp(np.asarray(img, dtype=np.float64), inverse_map=fine.inverse,
                 output_shape=out_shape[:2], order=order, cval=0.0, preserve_range=True)
        chans.append(np.clip(w, 0.0, 1.0))
    out = np.stack(chans, axis=-1) if color else chans[0]
    if color:
        out[np.broadcast_to(fp[..., None], out.shape)] = np.nan
    else:
        out[fp] = np.nan
    return out


def _align_astroalign(frame: np.ndarray, ref: np.ndarray,
                      src: Optional[np.ndarray] = None, tgt: Optional[np.ndarray] = None,
                      fine_shape=None):
    import astroalign
    color = frame.ndim == 3
    if src is None:
        src = frame.mean(axis=2) if color else frame
    if tgt is None:
        tgt = ref.mean(axis=2) if color else ref
    transform, _ = astroalign.find_transform(src, tgt)
    if fine_shape is not None:                     # single interpolation, straight to 2×
        return _warp_fine(frame, transform, fine_shape,
                          scale=fine_shape[0] / ref.shape[0])
    if color:
        chans, footprint = [], None
        for c in range(frame.shape[2]):
            reg_c, footprint = astroalign.apply_transform(transform, frame[..., c], ref[..., c])
            chans.append(reg_c)
        out = np.clip(np.stack(chans, axis=-1), 0.0, 1.0)
        out[np.broadcast_to(footprint[..., None], out.shape)] = np.nan
    else:
        out, footprint = astroalign.apply_transform(transform, frame, ref)
        out = np.clip(out, 0.0, 1.0)
        out[footprint] = np.nan                    # footprint True == no source data
    return out


def _shift_with_nodata(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """FFT-fallback shift, then mark the shifted-in border as NaN (no-data), not 0."""
    shifted = fftreg.apply_shift(img, dx, dy)
    ones = np.ones(img.shape[:2], dtype=np.float64)
    covered = _nd_shift(ones, (dy, dx), order=0, mode="constant", cval=0.0) >= 0.5
    if shifted.ndim == 3:
        shifted[~covered] = np.nan
    else:
        shifted[~covered] = np.nan
    return shifted


class Aligner:
    """Aligns single frames to a fixed reference (astroalign, else FFT translation).

    Prepared once from the reference so the FFT fallback doesn't re-transform it per frame;
    lets the caller register + stage frames one at a time (bounded memory) instead of holding
    the whole aligned stack in RAM.
    """

    def __init__(self, ref: np.ndarray, *, log: Callable[[str], None] = _noop,
                 sky_mask: Optional[np.ndarray] = None):
        self.ref = np.asarray(ref, dtype=np.float64)
        # sky_mask (1=sky): blank the foreground before source detection so astroalign matches sky
        # asterisms only — a nightscape's bright static land otherwise breaks the whole-frame match.
        self.sky_mask = None if sky_mask is None else (np.asarray(sky_mask) < 0.5)   # True == foreground
        self._ref_src = self._sky_source(self.ref) if self.sky_mask is not None else None
        self.use_aa = astroalign_available()
        self._ref_fft = None
        if self.use_aa:
            log("Registration: astroalign (asterism-matched affine)."
                + (" Sky-masked (nightscape)." if self.sky_mask is not None else ""))
        else:
            log("Registration: astroalign not installed — FFT translation-only fallback "
                "(no rotation correction; install astroalign for tracked/rotated sets).")
            ref_wk, _ = fftreg.working(self.ref)
            self._ref_fft = fftreg.make_ref(fftreg.gradient(ref_wk),
                                            ref_wk.shape[1], ref_wk.shape[0])

    def _sky_source(self, frame: np.ndarray) -> np.ndarray:
        """Luminance with the foreground blanked to the sky-median (for sky-only source detection)."""
        L = (frame.mean(axis=2) if frame.ndim == 3 else frame).astype(np.float64).copy()
        fg = self.sky_mask
        L[fg] = float(np.median(L[~fg])) if np.any(~fg) else 0.0
        return L

    def set_fine(self, factor: int) -> bool:
        """Enable the fine output grid (amplified drizzle-lite). astroalign only —
        the FFT fallback has no transform to compose. Returns whether it's active."""
        self._fine = int(factor) if (self.use_aa and factor > 1) else None
        return self._fine is not None

    def _fine_shape(self, frame: np.ndarray):
        if getattr(self, "_fine", None) is None:
            return None
        f = self._fine                                     # canvas from the REFERENCE, so a
        return (self.ref.shape[0] * f, self.ref.shape[1] * f)   # cropped odd frame still
                                                                # lands on the common grid

    def align_ref(self) -> np.ndarray:
        """The reference itself on the output grid (identity transform, scaled)."""
        if getattr(self, "_fine", None) is None:
            return self.ref
        from skimage.transform import AffineTransform
        return _warp_fine(self.ref, AffineTransform(matrix=np.eye(3)),
                          self._fine_shape(self.ref), scale=float(self._fine))

    def align(self, frame: np.ndarray) -> Optional[np.ndarray]:
        f = np.asarray(frame, dtype=np.float64)
        if self.use_aa:
            src = self._sky_source(f) if self.sky_mask is not None else None
            return _align_astroalign(f, self.ref, src=src, tgt=self._ref_src,
                                     fine_shape=self._fine_shape(f))
        wk, k = fftreg.working(f)
        dx, dy = fftreg.measure_against(self._ref_fft, fftreg.gradient(wk),
                                        wk.shape[1], wk.shape[0])
        return _shift_with_nodata(f, -dx / k, -dy / k)


def register(frames: List[np.ndarray], reference: int, *,
             log: Callable[[str], None] = _noop) -> Tuple[List[np.ndarray], List[int]]:
    """Align every frame to ``frames[reference]``. Returns (aligned, kept-indices)."""
    aligner = Aligner(np.asarray(frames[reference], dtype=np.float64), log=log)
    aligned, kept = [], []
    for i, f in enumerate(frames):
        if i == reference:
            aligned.append(np.asarray(f, dtype=np.float64))
            kept.append(i)
            continue
        try:
            aligned.append(aligner.align(f))
            kept.append(i)
        except Exception as e:
            log(f"  frame {i} failed to register ({e}) — dropped")
    return aligned, kept
