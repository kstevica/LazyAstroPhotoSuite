"""Software star reduction — thin the star carpet without StarNet (port-only finishing dial).

For widefield / Milky Way frames the dense star field buries the dust lanes and nebulosity.
StarNet-based reduction needs the external tool and struggles on a million-star field; this
is a fast morphological alternative: a grayscale opening removes small bright peaks (stars)
while leaving extended structure (nebulosity, large stars) intact, and the isolated star
excess is subtracted by ``amount``. Applied on luminance and scaled per channel so colour is
preserved. Cheap enough to run live in Preview.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

_LUM = np.array([0.2126, 0.7152, 0.0722])


def _luminance(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return img[..., :3] @ _LUM
    return img


def shrink_stars(img: np.ndarray, amount: float, small: float = 0.5) -> np.ndarray:
    """Reduce stars by ``amount`` (0..1). ``small`` (0..1) sets the star scale targeted.

    A morphological opening (min- then max-filter) of the luminance removes peaks smaller
    than the structuring element — i.e. stars — giving a star-free-ish floor; the excess
    above it is the star light, which is subtracted by ``amount``.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return np.asarray(img, dtype=np.float64)
    a = np.asarray(img, dtype=np.float64)
    lum = _luminance(a)
    size = int(3 + round(float(np.clip(small, 0.0, 1.0)) * 6))    # 3..9 px star scale
    opened = maximum_filter(minimum_filter(lum, size=size, mode="nearest"),
                            size=size, mode="nearest")
    star = np.clip(lum - opened, 0.0, None)                       # small bright peaks = stars
    reduced = lum - amount * star
    if a.ndim == 3:
        scale = np.where(lum > 1e-6, reduced / np.where(lum > 1e-6, lum, 1.0), 1.0)
        out = a * scale[..., None]
    else:
        out = reduced
    return np.clip(out, 0.0, 1.0)
