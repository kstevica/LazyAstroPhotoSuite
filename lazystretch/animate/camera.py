"""Eased camera paths. Each returns a list of :class:`~.render.Cam` poses.

Motion is deterministic (no RNG) so a render is reproducible. Progress is shaped
by smootherstep so moves accelerate and settle rather than starting/stopping
abruptly, and a low-frequency sway keeps the shot from feeling locked-off.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .render import Cam


def _ease(t: float) -> float:
    """Smootherstep (Ken Perlin): 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _twinkle(u: float) -> float:
    return u * 2.0 * np.pi * 6.0


def flythrough(n: int, *, zoom_end: float = 1.35, pan=(0.02, -0.016),
               roll_deg: float = 1.1, sway: float = 0.01) -> List[Cam]:
    """Dolly straight in — the core billows toward you, near stars sweep past."""
    out: List[Cam] = []
    for i in range(n):
        u = i / max(n - 1, 1)
        e = _ease(u)
        px = pan[0] * e + sway * np.sin(u * 2.0 * np.pi)
        py = pan[1] * e + sway * 0.7 * np.sin(u * 2.0 * np.pi + 1.1)
        roll = roll_deg * np.sin(u * np.pi) * 0.6
        out.append(Cam(zoom=1.0 + (zoom_end - 1.0) * e, pan_x=px, pan_y=py,
                       roll=roll, twinkle=_twinkle(u)))
    return out


def flyby(n: int, *, span: float = 0.16, zoom_end: float = 1.14,
          roll_deg: float = 1.6) -> List[Cam]:
    """Lateral drift across the field with a slow push and gentle roll."""
    out: List[Cam] = []
    for i in range(n):
        u = i / max(n - 1, 1)
        e = _ease(u)
        out.append(Cam(zoom=1.0 + (zoom_end - 1.0) * e,
                       pan_x=span * (e - 0.5) * 2.0, pan_y=0.0,
                       roll=roll_deg * np.sin(u * np.pi), twinkle=_twinkle(u)))
    return out


def orbit(n: int, *, radius: float = 0.05, zoom_end: float = 1.28,
          roll_deg: float = 1.2) -> List[Cam]:
    """Sweep the camera around a small circle while pushing in — a reveal."""
    out: List[Cam] = []
    for i in range(n):
        u = i / max(n - 1, 1)
        e = _ease(u)
        ang = u * np.pi                                  # half turn over the clip
        out.append(Cam(zoom=1.0 + (zoom_end - 1.0) * e,
                       pan_x=radius * np.sin(ang),
                       pan_y=radius * (1.0 - np.cos(ang)) * 0.6,
                       roll=roll_deg * np.sin(ang), twinkle=_twinkle(u)))
    return out
