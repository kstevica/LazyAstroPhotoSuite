"""Highlight roll-off — a soft top-end knee so highlights never hard-clip to pure white.

Not a port of a specific PI process: a **calibration** to match PixInsight's highlight
character. The golden survey (calibration/survey.py) showed PI keeps near-white ≈ 0% on
every target — stars and bright cores roll off just under white and keep colour — while
the port's finishing steps (esp. LocalHistogramEqualization) push bright pixels over 1.0
and hard-clip them to (1,1,1), producing blown grey-white cores/stars.

This applies a smooth per-channel compression above a knee: below the knee, identity;
above it, ``t/(1+t)`` maps ``[knee, 1] -> [knee, knee + 0.5*(1-knee)]`` asymptotically, so
no channel reaches 1.0 and relative channel order (hue) is preserved. Run last.
"""
from __future__ import annotations

import numpy as np

# Knee where the roll-off begins. Tuned against the golden set so the brightest pixels
# land near PI's peak (~0.90-0.95) and near-white % approaches PI's ~0-0.1%.
_HIGHLIGHT_KNEE: float = 0.82


def highlight_rolloff(img, knee: float = _HIGHLIGHT_KNEE) -> np.ndarray:
    """Soft-compress the top end per channel so highlights don't hard-clip to white.

    ``img`` float64 in [0, 1], mono or RGB. Below ``knee`` values pass through; above it
    they are smoothly compressed toward (but never reaching) 1.0. Returns a new array.
    """
    a = np.asarray(img, dtype=np.float64)
    if not (0.0 < knee < 1.0):
        return np.clip(a, 0.0, 1.0)
    span = 1.0 - knee
    hi = a > knee
    # t >= 0 above the knee; t/(1+t) in [0, 1) -> asymptotes below 1, monotone.
    t = np.where(hi, (a - knee) / span, 0.0)
    comp = knee + span * (t / (1.0 + t))
    out = np.where(hi, comp, a)
    return np.clip(out, 0.0, 1.0)
