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


def _opening(x: np.ndarray, size: int) -> np.ndarray:
    """Grayscale morphological opening (min- then max-filter) — removes bright peaks < size."""
    return maximum_filter(minimum_filter(x, size=size, mode="nearest"),
                          size=size, mode="nearest")


def shrink_stars(img: np.ndarray, amount: float, small: float = 0.5, *,
                 med_keep: float = 0.35, anchor_lo: float = 0.18,
                 anchor_hi: float = 0.45) -> np.ndarray:
    """Reduce stars by ``amount`` (0..1), size- and brightness-TIERED (P4).

    A single morphological opening treats every star the same — it dims the bright anchor
    stars along with the faint carpet, and flattens the medium-size stars that give a field
    its depth. Instead this separates the star light into tiers and reduces each differently:

    * **Tiny carpet** (peaks < ``s_tiny`` px): crushed at full ``amount`` — the dense speckle
      that buries nebulosity.
    * **Medium stars** (``s_tiny``–``s_med`` px): kept mostly, reduced only by ``amount·med_keep``,
      so real stellar profiles survive.
    * **Bright anchors**: preserved regardless of size — a smoothstep on each star's PEAK excess
      (``anchor_lo``→``anchor_hi``) rolls the reduction to zero on the brightest stars (only their
      halos, not the cores, would ever be touched).

    ``small`` (0..1) sets the tiny-star scale. Applied on luminance and rescaled per channel so
    colour is preserved. Input never mutated; output clipped to [0, 1].
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return np.asarray(img, dtype=np.float64)
    a = np.asarray(img, dtype=np.float64)
    lum = _luminance(a)

    s_tiny = int(3 + round(float(np.clip(small, 0.0, 1.0)) * 2))   # 3..5 px (tiny carpet)
    s_med = s_tiny + 4                                             # 7..9 px (medium stars)
    open_t = _opening(lum, s_tiny)                                # floor without tiny stars
    open_m = _opening(open_t, s_med)                             # floor without tiny + medium
    tiny = np.clip(lum - open_t, 0.0, None)                      # tiny-star excess
    med = np.clip(open_t - open_m, 0.0, None)                    # medium-star band

    # Preserve bright anchors: gate on each star's PEAK excess (max over the medium footprint,
    # so a whole bright star — not just its centre — is spared, avoiding a reduced-edge ring).
    peak = maximum_filter(tiny + med, size=s_med, mode="nearest")
    t = np.clip((peak - anchor_lo) / max(anchor_hi - anchor_lo, 1e-6), 0.0, 1.0)
    keep = 1.0 - t * t * (3.0 - 2.0 * t)                        # 1 on carpet, →0 on bright anchors

    reduced = lum - amount * keep * (tiny + med_keep * med)
    if a.ndim == 3:
        scale = np.where(lum > 1e-6, reduced / np.where(lum > 1e-6, lum, 1.0), 1.0)
        out = a * scale[..., None]
    else:
        out = reduced
    return np.clip(out, 0.0, 1.0)
