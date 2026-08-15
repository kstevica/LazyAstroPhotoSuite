"""Depth-image renderer — the engine that makes a still look 3D.

The starless nebula is warped every frame by a *continuous per-pixel* amount
that grows with each pixel's synthesised nearness: near gas scales and pans
faster than the faint outskirts, so a dolly-in makes the core billow toward you
while the background barely moves. That differential motion is the parallax the
eye reads as depth. Because the warp is continuous (one smooth displacement
field, not a stack of discrete depth planes) it has no banding and no radial
plane-replication. The stars are a separate point cloud warped on their own —
each with a depth from its brightness — so they sweep outward past the frame as
you fly in, the single strongest 3D cue. A screen-blended bloom gives the gas
volume.

It invents no structure: the sheet is the real (starless) pixels resampled, and
the stars are the real detected stars.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .depth import _as_rgb, depth_field, detect_stars, starless


@dataclass
class Cam:
    """A single camera pose. ``zoom`` ≥ 1 dollies in; ``pan_*`` are fractions of
    the frame; ``roll`` is degrees; ``twinkle`` phase animates star shimmer."""
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    roll: float = 0.0
    twinkle: float = 0.0


# depth-parallax gains: how much scale / pan a pixel or star gets vs its nearness.
# The nebula sheet moves as a gentle smooth relief (small near/far spread → no
# melty shear); the stars, given the same gains but sharper depths, carry the pop.
_FAR_SCALE, _NEAR_SCALE = 0.55, 0.95
_FAR_PAN, _NEAR_PAN = 0.45, 1.05


def _depth_scale(zoom, depth, overscan: float):
    """Per-pixel/-star scale about frame centre; nearer parallaxes more."""
    return overscan * (1.0 + (zoom - 1.0) * (_FAR_SCALE + _NEAR_SCALE * depth))


class Flythrough3D:
    """Build the starless sheet + star cloud once, then render any camera pose."""

    def __init__(self, img: np.ndarray, *,
                 masks: Optional[Dict[str, np.ndarray]] = None,
                 overscan: float = 1.16, max_stars: int = 1200,
                 star_depth: Tuple[float, float] = (0.62, 1.0),
                 bloom: float = 0.45) -> None:
        rgb = _as_rgb(img)
        self.h, self.w = rgb.shape[:2]
        self.overscan = float(overscan)
        self.bloom_strength = float(bloom)

        self.depth = depth_field(rgb, masks=masks).astype(np.float64)
        self.base = starless(rgb)                       # nebula without stars
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float64)
        self._yy, self._xx = yy, xx                     # output coordinate grid

        # --- star point cloud (parallaxed independently) ----------------------
        st = detect_stars(rgb, max_stars=max_stars)
        self.star_yx = st[:, :2].astype(np.float64)
        self.star_flux = st[:, 2].astype(np.float64)
        self.star_rgb = st[:, 3:6].astype(np.float64)
        if len(self.star_flux):
            rank = np.argsort(np.argsort(self.star_flux)) / max(len(self.star_flux) - 1, 1)
            self.star_d = star_depth[0] + (star_depth[1] - star_depth[0]) * rank
        else:
            self.star_d = np.zeros(0)

    # ------------------------------------------------------------------ warp
    def _warp_nebula(self, cam: Cam) -> np.ndarray:
        """Continuous per-pixel depth warp of the starless sheet (no banding)."""
        from scipy.ndimage import map_coordinates

        h, w = self.h, self.w
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        z = self.depth                                  # (h, w) nearness in [0,1]
        s = self.overscan * (1.0 + (cam.zoom - 1.0) * (_FAR_SCALE + _NEAR_SCALE * z))
        pan_gain = _FAR_PAN + (_NEAR_PAN - _FAR_PAN) * z
        ty = cam.pan_y * h * pan_gain
        tx = cam.pan_x * w * pan_gain
        th = np.deg2rad(cam.roll)                        # roll is uniform
        cos, sin = np.cos(th), np.sin(th)

        # inverse map: source = c + (1/s) R(-th) (out - c - t)
        dy = self._yy - cy - ty
        dx = self._xx - cx - tx
        inv = 1.0 / s
        py = cy + inv * (cos * dy + sin * dx)
        px = cx + inv * (-sin * dy + cos * dx)
        coords = np.stack([py, px])
        out = np.empty((h, w, 3), np.float32)
        for ch in range(3):
            out[..., ch] = map_coordinates(self.base[..., ch], coords, order=1,
                                           mode="nearest", prefilter=False)
        return np.clip(out, 0.0, 1.0)

    # ---------------------------------------------------------------- stars
    def _render_stars(self, cam: Cam) -> np.ndarray:
        h, w = self.h, self.w
        buf = np.zeros((h, w, 3), np.float32)
        if not len(self.star_d):
            return buf
        from scipy.ndimage import gaussian_filter

        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        d = self.star_d
        # stars parallax harder than the nebula sheet (they are the foreground)
        s = self.overscan * (1.0 + (cam.zoom - 1.0) * (0.6 + 1.9 * d))
        th = np.deg2rad(cam.roll)                        # roll is uniform
        cos, sin = np.cos(th), np.sin(th)
        y = self.star_yx[:, 0] - cy
        x = self.star_yx[:, 1] - cx
        xr = cos * x - sin * y
        yr = sin * x + cos * y
        pan_gain = 0.5 + 1.5 * d
        yy = cy + s * yr + cam.pan_y * h * pan_gain
        xx = cx + s * xr + cam.pan_x * w * pan_gain

        inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        yi = yy[inside].astype(np.intp)
        xi = xx[inside].astype(np.intp)
        flux = self.star_flux[inside] * (0.75 + 0.25 * s[inside])
        if cam.twinkle:
            idx = np.nonzero(inside)[0]
            flux = flux * (1.0 + 0.18 * np.sin(cam.twinkle + idx * 1.7))
        col = self.star_rgb[inside]
        for ch in range(3):
            np.add.at(buf[..., ch], (yi, xi), flux * col[:, ch])
        for ch in range(3):
            buf[..., ch] = gaussian_filter(buf[..., ch], 0.8)
        return np.clip(buf * 2.7, 0.0, 1.0)

    # ---------------------------------------------------------------- bloom
    def _bloom(self, out: np.ndarray) -> np.ndarray:
        if self.bloom_strength <= 0:
            return out
        from scipy.ndimage import gaussian_filter

        hi = np.clip(out - 0.72, 0.0, 1.0)
        r = max(self.w, self.h) / 180.0
        glow = np.stack([gaussian_filter(hi[..., c], r) for c in range(3)], axis=-1)
        return out + self.bloom_strength * glow

    # --------------------------------------------------------------- render
    def render_frame(self, cam: Cam) -> np.ndarray:
        """Render one frame → float32 ``(H, W, 3)`` in ``[0, 1]``."""
        neb = self._warp_nebula(cam)
        stars = self._render_stars(cam)
        out = 1.0 - (1.0 - neb) * (1.0 - stars)         # screen (additive light)
        return np.clip(self._bloom(out), 0.0, 1.0)
