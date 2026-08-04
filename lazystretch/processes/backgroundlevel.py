"""Background-level pull — a midtones (MTF) pull toward a target sky level.

Ported from PixInsight LazyStretch v1.2.1:
  * ``Pipe.backgroundLevel`` — LazyStretch.js:2481-2502

Measure the sky background, then (only if it sits above target) solve for the
midtones balance ``m`` such that ``mtf(m, bg) == targetLevel`` and apply that MTF
to the whole image. See PLAN §11e (measure -> correct).

Source (LazyStretch.js:2483-2502)::

    let bg = medSum / n;                         // channel-averaged median
    if ( bg > targetLevel + 0.005 )
    {
       let m = LS.findMidtonesBalance( targetLevel, bg );  // mtf(m,bg)=targetLevel
       let HT = new HistogramTransformation;
       HT.H = [ [0,0.5,1,0,1], [0,0.5,1,0,1], [0,0.5,1,0,1],
                [0, m, 1,0,1], [0,0.5,1,0,1] ];              // RGB/K row = m
       HT.executeOn( view, false );
    }

The ``HT.H`` matrix leaves the per-channel R/G/B rows (0,1,2) at identity
(midtones 0.5, shadows 0, highlights 1) and puts the pull on the combined RGB/K
row (index 3) with midtones ``m``. In PixInsight the RGB/K row applies the *same*
transform to every nominal channel, so this is a **linked** MTF: one ``m`` for all
channels. Shadows=0, highlights=1 and dynamic range [0,1] mean the transform is a
pure ``mtf(m, x)`` mapping [0,1] -> [0,1] (no rescale, natural [0,1] output).
"""
from __future__ import annotations

import math

import numpy as np

from ..data.loader import get_data
from ..stats.mtf import find_midtones_balance, mtf
from ..stats.measure import avg_dev_channel_avg, median_channel_avg

# Guard hysteresis from the source: only pull when the background exceeds the
# target by more than this margin (LazyStretch.js:2493).
_BACKGROUND_MARGIN: float = 0.005

# P1 calibration (NOT a PI number — kept out of the verbatim-PI JSON, like
# highlights._HIGHLIGHT_KNEE). PI's HT.H uses shadows=0, so the linked MTF slides the
# sky band down but never clips its low tail; combined with the port's approximate linear
# upstream (ABE~DBE, MMT/DeepSNR~NoiseX) and the deliberately SOFT auto-stretch knee
# (js:2422-2445, never hard-clips), a milky floor survives (survey blk ~2x PI). When the
# background is pulled, trim the pedestal with a small shadows clip c = bg - k*avgDev and
# re-anchor: this deepens the floor AND — because the re-solved MTF is gentler — lifts the
# dim cores (clum). Self-limiting: c<=0 (band already reaches black) -> faithful pure MTF.
# Calibrated k=6.0 against the golden set (execute mode): lifts the dim-core targets while
# landing the brightest emission core (M42) at PI (clum -1%, NOT overshooting into the
# washed/overblown look) — a smaller k (4.0) closed the aggregate clum gap harder but pushed
# M42 to +22% above PI. Larger k is also strictly safer for the must-not-regress targets
# (m8/ic1396 fall to the c<=0 faithful fallback sooner). See [[lazystretch-p1-core-brightness-gap]].
_BLACKPOINT_KNEE_AVGDEV: float = 6.0


def background_level(img, target_level: float) -> np.ndarray:
    """Lower the sky background toward ``target_level`` via a linked MTF pull.

    Parameters
    ----------
    img : numpy.ndarray
        Float64 image in [0, 1], shape (H, W) mono or (H, W, 3) RGB (R,G,B).
    target_level : float
        Desired background level. The image is only modified when the measured
        background sits more than 0.005 above this (LazyStretch.js:2493).

    Returns
    -------
    numpy.ndarray
        A new array. When no pull is applied, an unmodified copy of ``img``.

    Notes
    -----
    ``bg`` is the channel-averaged median (PI ``img.median()`` summed over
    channels / N). The pull uses a single midtones balance ``m`` solved so that
    ``mtf(m, bg) == target_level`` and applies ``mtf(m, .)`` to every channel
    (a *linked* stretch, matching the RGB/K row of ``HT.H``). Mono and RGB take
    the identical code path — the only difference is how many channels the median
    is averaged over and how many channels the MTF maps.
    """
    a = np.asarray(img, dtype=np.float64)

    # bg = channel-averaged median (LazyStretch.js:2488-2491).
    bg = median_channel_avg(a)

    # Gentle: only move if the background is above target (LazyStretch.js:2493).
    if bg > target_level + _BACKGROUND_MARGIN:
        # Shadows-clip re-anchor (P1 calibration): trim the milky pedestal at c = bg -
        # k*avgDev with PI's OWN soft knee (softmax0, applyAutoStretch js:2426-2444), then
        # solve the midtones from the SOFTENED background — a gentler MTF that deepens the
        # floor without crushing the cores it re-anchors. Self-limiting: c<=0 (the band
        # already reaches black) falls back to the bit-identical faithful pure-MTF pull.
        avgdev = avg_dev_channel_avg(a)
        c = bg - _BLACKPOINT_KNEE_AVGDEV * avgdev
        if c > 0.0:
            st = get_data().stretch
            e = max(st["kneeWidthFactor"] * avgdev, st["kneeEpsilonFloor"])
            d = a - c
            xk = np.clip((d + np.sqrt(d * d + e * e)) / (2.0 * (1.0 - c)), 0.0, 1.0)
            dm = bg - c
            soft_bg = (dm + math.sqrt(dm * dm + e * e)) / (2.0 * (1.0 - c))
            if not math.isfinite(soft_bg):
                soft_bg = bg
            m = find_midtones_balance(target_level, soft_bg)
            return mtf(m, xk)

        # c <= 0: no pedestal to trim -> the faithful pure-MTF pull (LazyStretch.js:2495).
        # Solve m so mtf(m, bg) == target_level; linked MTF over all channels (HT.H RGB/K
        # row). mtf maps [0,1] -> [0,1] and returns a NEW array (never mutates input).
        m = find_midtones_balance(target_level, bg)
        return mtf(m, a)

    # No pull: return an unmodified copy (never mutate / alias the input).
    return a.copy()
