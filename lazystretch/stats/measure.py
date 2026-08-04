"""Channel-averaged image statistics — PI's ``median()`` / ``avgDev()`` / ``mean()``.

Ported from PixInsight LazyStretch v1.2.1. The stretch is a **linked** stretch: a
statistic is computed per channel, then averaged over channels (LazyStretch.js:2086-2097).

``avgDev()`` is the **mean absolute deviation about the median** — NOT the standard
deviation and NOT ``1.4826 * MAD``. The entire auto-stretch is built on it (PLAN §4.2).

Channel-count contract (PLAN §2): inputs are expected **alpha-stripped** — mono
``(H, W)`` or RGB ``(H, W, 3)``. PI mixes ``numberOfChannels`` (includes alpha) with
``numberOfNominalChannels`` (excludes alpha); by stripping alpha at load, "all
channels" and "nominal channels" coincide here.
"""
from __future__ import annotations

import numpy as np


def _channels(img):
    """Yield each channel of ``img`` as a 2-D float64 array."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 2:
        return [a]
    if a.ndim == 3:
        return [a[..., c] for c in range(a.shape[2])]
    raise ValueError(f"expected a 2-D (mono) or 3-D (RGB) image, got shape {a.shape}")


def avg_dev(x) -> float:
    """Mean absolute deviation about the median of a single array (PI ``avgDev()``)."""
    a = np.asarray(x, dtype=np.float64)
    return float(np.mean(np.abs(a - np.median(a))))


def median_channel_avg(img) -> float:
    """Channel-averaged median (PI ``img.median()`` summed over channels / N)."""
    return float(np.mean([np.median(c) for c in _channels(img)]))


def avg_dev_channel_avg(img) -> float:
    """Channel-averaged mean-absolute-deviation-about-median (PI ``img.avgDev()``)."""
    return float(np.mean([avg_dev(c) for c in _channels(img)]))


def mean_channel_avg(img) -> float:
    """Channel-averaged mean (PI ``img.mean()``)."""
    return float(np.mean([np.mean(c) for c in _channels(img)]))


def region_median_channel_avg(img, x0: int, y0: int, x1: int, y1: int) -> float:
    """Channel-averaged median inside the half-open rect ``[x0:x1, y0:y1]``.

    Matches PI's ``img.selectedRect`` median averaged over channels
    (LazyStretch.js:2181-2184, 2592-2596). ``x``/``y`` are column/row pixel indices.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 2:
        return float(np.median(a[y0:y1, x0:x1]))
    return float(np.mean([np.median(a[y0:y1, x0:x1, c]) for c in range(a.shape[2])]))
