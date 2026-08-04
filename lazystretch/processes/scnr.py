"""SCNR average-neutral green removal.

Ported from PixInsight LazyStretch v1.2.1:
  * ``Pipe.scnr`` — LazyStretch.js:2434-2442

Source (LazyStretch.js:2434-2442)::

    Pipe.scnr = function( view )
    {
       let P = new SCNR;
       P.amount           = 1.0;
       P.protectionMethod = SCNR.AverageNeutral;
       P.colorToRemove    = SCNR.Green;
       P.preserveLightness = true;
       P.executeOn( view );
    };

Average-neutral green protection removes the residual green cast that most colour
data shows after calibration/stretch.  For the *Green* colour and the
*AverageNeutral* protection method the per-pixel rule is::

    m       = (R + B) / 2
    G_new   = (1 - amount) * G + amount * min(G, m)

With ``amount = 1.0`` this collapses to ``G_new = min(G, (R + B) / 2)`` — green is
only ever pulled *down*, never raised, and only where it exceeds the red/blue
average.  This part is a faithful, exact port of the SCNR maths (see the PCL
``SCNRInstance`` average-neutral kernel).

``preserveLightness = true`` is where PixInsight uses its own colour engine
(``RGBColorSystem`` / the image's RGB working space, CIE L\\*).  That colour space
and its default coefficients are not reproduced here, so **lightness preservation
is an APPROXIMATION** (see :func:`scnr` notes):

  * PixInsight's *default* RGB working space uses **equal** luminance weights
    (Y = (R+G+B)/3) — no channel is favoured — so we preserve that equal-weight
    luminance.  CIE L\\* is a strictly monotonic function of luminance Y, so
    holding Y fixed holds L\\* fixed; we therefore rescale the whole pixel by
    ``k = Y_before / Y_after`` (>= 1, since removing green only lowers Y).  This
    restores brightness while keeping the *reduced* green-to-red/blue ratio, i.e.
    the cast stays removed.  We do NOT model the working-space gamma / cube-root
    transfer, hence "approximate".

Colour-only op: a **no-op on mono** input (PI's SCNR requires an RGB view), and
truncated to [0, 1] on output.
"""
from __future__ import annotations

import numpy as np


def scnr(img, amount: float = 1.0, preserve_lightness: bool = True) -> np.ndarray:
    """SCNR average-neutral green removal (``Pipe.scnr``, LazyStretch.js:2434-2442).

    Parameters
    ----------
    img
        float64 array in [0, 1]; mono ``(H, W)`` or RGB ``(H, W, 3)`` with channel
        order R, G, B == index 0, 1, 2.
    amount
        Blend between the original green and the average-neutral-clipped green.
        PI hard-codes ``1.0`` (full removal); exposed for testing/calibration.
    preserve_lightness
        PI hard-codes ``True``.  Rescales each pixel to hold its equal-weight
        luminance (an approximation of PI's CIE-L\\* preservation — see module
        docstring).

    Returns
    -------
    np.ndarray
        A new array (input never mutated).  Mono input is returned as an
        unchanged copy (colour-only no-op).  RGB output is clipped to [0, 1].
    """
    a = np.asarray(img, dtype=np.float64)

    # Colour-only op: no-op on mono / single-channel (PI SCNR needs an RGB view).
    if a.ndim != 3 or a.shape[2] < 3:
        return a.copy()

    out = a.copy()
    R = a[..., 0]
    G = a[..., 1]
    B = a[..., 2]

    # Average-neutral protection for green: pull green down toward (R+B)/2 only
    # where it exceeds it.  G_new = (1-amount)*G + amount*min(G, m).
    m = 0.5 * (R + B)
    g_clipped = np.minimum(G, m)
    g_new = (1.0 - amount) * G + amount * g_clipped

    out[..., 1] = g_new

    if preserve_lightness:
        # Preserve equal-weight luminance Y = (R+G+B)/3 (PI default RGBWS weights;
        # CIE L* is monotonic in Y, so this holds L* fixed).  Scale the whole
        # pixel by k = Y_before / Y_after.  Removing green lowers Y, so k >= 1.
        y_before = R + G + B                     # (drop the common /3 factor)
        y_after = R + g_new + B
        # Guard degenerate pixels (Y_after == 0): leave the post-SCNR value as-is.
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(y_after > 0.0, y_before / y_after, 1.0)
        out = out * k[..., np.newaxis]

    # SCNR keeps samples in range; truncate to [0, 1].
    np.clip(out, 0.0, 1.0, out=out)
    return out
