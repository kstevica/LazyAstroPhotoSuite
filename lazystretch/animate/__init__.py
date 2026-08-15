"""LazyFlight — turn a finished still into a faithful 3D fly-through.

The input is an *already stretched / developed* master (a LazyStretch or
LazyDevelop result). The output is a short video that simulates flying through
or past the object. There is no invented structure: every frame re-projects the
real pixels you captured. The "3D" is synthesised from two honest cues —

  1. **Differential parallax** between depth layers derived from the image itself
     (bright emission and Hα come forward, dust recedes, stars sit nearest so
     they sweep past as the camera dollies in), and
  2. **Self-glow bloom** so the gas reads as volume, not a flat sheet.

Pipeline: :func:`depth.depth_field` (relief) + :func:`depth.starless`
(background) + :func:`depth.detect_stars` (parallax point layer) feed
:class:`render.Flythrough3D`, driven along an eased :mod:`camera` path and
encoded by :func:`encode.write_video`.
"""
from __future__ import annotations

from .render import Cam, Flythrough3D
from .camera import flythrough, flyby, orbit, pullback
from .encode import write_video, ffmpeg_available
from .clip import render_flythrough, PATHS
from .parallel import auto_workers, parallel_frames
from .volume3d import SpaceFly, VolCam, fly_volume, render_space, color_grade
from .flyv2 import V2Fly, V2Cam, fly_v2, render_v2

__all__ = [
    "Cam",
    "Flythrough3D",
    "flythrough",
    "flyby",
    "orbit",
    "pullback",
    "write_video",
    "ffmpeg_available",
    "render_flythrough",
    "PATHS",
    "auto_workers",
    "parallel_frames",
    "SpaceFly",
    "VolCam",
    "fly_volume",
    "render_space",
    "color_grade",
    "V2Fly",
    "V2Cam",
    "fly_v2",
    "render_v2",
]
