"""Deepen — the deterministic second-pass stretch and its source-highlight restore.

Ported from PixInsight LazyStretch v1.4.1:
  * ``Deepen`` step (the inline pipeline stage) — LazyStretch.js:4554-4608
  * ``Pipe.restoreSourceHighlights``            — LazyStretch.js:4224-4327

Both functions are **statistics-free** — they MEASURE NOTHING (the six-round IC 2118
lesson: re-measured statistics break second passes). Every number is a fixed MTF or a
fixed smoothstep threshold driven only by the Deepen dial.

Deepen (js:4554-4608)
---------------------
The second pass is a fixed midtones-transfer-function lift split in two:
  * 30% of the lift is ridden by EVERYONE — ``mtf(mAll, .)`` applied globally
    (``pmA`` in the source), so protected zones do not fall below the risen field
    and carve dark moats around bright stars at a high dial;
  * the remaining 70% is the strong part — ``mtf(mRest, .)`` applied through an
    INVERTED bright-range mask (``pmB`` under ``win.maskInverted = true``), so the
    lift runs on the DIM side while bright highlights and their grown halo zones
    are protected from compounding.

restoreSourceHighlights (js:4224-4327)
--------------------------------------
Puts back the pre-Deepen highlights (and, for RGB, the blue reflection halo) that the
whole polish pipeline pulled down, keyed off the ORIGINAL source rather than any
re-measured statistic. The source is first ridden by 30% of the Deepen (matching the
ride the protected zones got) so the revert can never undershoot the risen field; a
bright-range mask + an inline RGB blue-excess halo key select where to blend the source
back; finally a hue-of-the-gain term damps blue-dominant gains (lifted haze) back toward
the source without any threshold on the image itself.

Contract: float64 arrays in [0, 1]; ``(H, W)`` mono or ``(H, W, 3)`` RGB with channel
order R,G,B == index 0,1,2. Inputs are never mutated — new arrays are returned. Every
PixelMath step in the source sets ``rescale = false`` / ``truncate = true``, so results
are clipped to [0, 1] exactly where PI truncates. The two colour terms (blue-halo
extension and hue-of-gain damp) are skipped on mono, matching the ``nCh >= 3`` guards.

Fidelity note: ``build_range_mask`` approximates PI ``RangeSelection`` (see
``masks.py``); the surrounding arithmetic here is bit-faithful to the source, so both
functions are ``near-exact`` — as faithful as the shared mask primitive allows.
"""
from __future__ import annotations

import numpy as np

from ..stats.mtf import mtf
from .crop import crop_edges
from .masks import apply_masked, build_range_mask


def deepen_stretch(img, amount):
    """Deterministic Deepen second-pass lift (LazyStretch.js:4554-4608).

    Parameters
    ----------
    img : numpy.ndarray
        ``(H, W)`` mono or ``(H, W, 3)`` RGB float64 image in [0, 1].
    amount : float
        The Deepen dial ``d`` in [0, 1] (0 = none, 1 = a full additional stretch).

    Returns
    -------
    numpy.ndarray
        A NEW array; the input is never mutated. Clipped to [0, 1].

    Notes
    -----
    ``d <= 0`` returns an unchanged copy. Otherwise everyone rides 30% of the lift
    globally (``mAll``) and the strong 70% (``mRest``) is applied through the
    bright-range mask INVERTED, so the lift lands on the dim side while bright
    highlights and their grown surroundings are protected. If the mask cannot be
    built, ``mtf(mRest, .)`` is applied globally (matches the source's MASK FAILED
    fallback path).
    """
    a = np.asarray(img, dtype=np.float64)
    d = float(np.clip(amount, 0.0, 1.0))
    if d <= 0.0:
        return a.copy()

    # js:4571-4572 — everyone rides 30%; the strong 70% is masked away from highlights.
    m_all = 0.5 - 0.35 * (0.30 * d)
    m_rest = 0.5 - 0.35 * (0.70 * d)

    # pmA (js:4573-4579): the 30% ride applied globally. mtf maps [0,1] -> [0,1], and
    # PI truncates; the value is already in range so no separate clip is needed.
    out = mtf(m_all, a)

    # pmB (js:4591-4597): the strong 70% computed over the whole image, then composited
    # through the INVERTED bright mask (js:4587 win.maskInverted = true).
    lifted = mtf(m_rest, out)

    # Mask is built from the 30%-ridden image (js:4582 buildRangeMask( target, ... )).
    mask = None
    try:
        mask = build_range_mask(out, 0.35, 1.0, 0.40, 60)
    except Exception:
        mask = None

    if mask is None:
        # MASK FAILED — strong part applied globally (js:4608).
        result = lifted
    else:
        # invert=True -> process (lift) where the mask is BLACK == the dim side;
        # bright + grown surroundings (mask white) keep the 30%-ridden `out`.
        result = apply_masked(out, lifted, mask, invert=True)

    return np.clip(result, 0.0, 1.0)


def restore_source_highlights(target, source, crop_fraction, deepen_amount):
    """Blend the pre-Deepen source highlights (and RGB blue halo) back into ``target``.

    Faithful port of ``Pipe.restoreSourceHighlights`` (LazyStretch.js:4224-4327).

    Parameters
    ----------
    target : numpy.ndarray
        The finished/polished image, ``(H, W)`` or ``(H, W, 3)`` float64 in [0, 1].
    source : numpy.ndarray
        The original (pre-Deepen) reference image to restore highlights from.
    crop_fraction : float
        Ratio in [0, 1) trimmed off each edge of ``source`` first (js:4231-4232),
        so its geometry matches ``target``. Pass 0 for no crop.
    deepen_amount : float
        The Deepen dial ``dd`` in [0, 1]; the source is ridden by 30% of it before
        restoring (js:4256-4267) so the revert can never undershoot the risen field.

    Returns
    -------
    numpy.ndarray
        A NEW array; neither input is mutated. Clipped to [0, 1]. If, after the
        optional crop, ``source`` and ``target`` shapes differ, ``target`` is returned
        unchanged (the geometry-mismatch skip, js:4233-4235).
    """
    tgt = np.asarray(target, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)

    # js:4231-4232 — crop the source's ragged edges to match the result geometry.
    if crop_fraction > 0:
        src = crop_edges(src, crop_fraction)

    # js:4233-4235 — geometry mismatch after crop -> skip, target unchanged.
    if src.shape != tgt.shape:
        return tgt.copy()

    # Own a copy so the source input is never mutated (mtf below also returns new).
    src = src.copy()

    # js:4237 — 2-D bright-range mask from the UN-RIDDEN source. PI builds this BEFORE
    # applying the 30% ride (comment js:4254: "Mask was built from the UNLIFTED source
    # above — the user's calibrated thresholds stay meaningful"). Build it first.
    mask = build_range_mask(src, 0.35, 1.0, 0.40, 60)

    # js:4256-4267 — ride the source by 30% of the Deepen so the restore matches the
    # ride the protected zones received and cannot carve dark moats at a high dial.
    dd = float(np.clip(deepen_amount, 0.0, 1.0))
    if dd > 0.0:
        m_ride = 0.5 - 0.35 * (0.30 * dd)
        src = mtf(m_ride, src)  # PI truncate=true; mtf already lands in [0, 1].

    is_rgb = tgt.ndim == 3

    # js:4279-4287 — blue-halo extension (RGB only): a blue-excess key (B - R),
    # smooth-ramped 0.05 -> 0.12, gated to mid-luminance 0.16 -> 0.28, max'd into
    # the mask so the residual reflection glow is caught by COLOUR, not luminance.
    if is_rgb:
        L = src.mean(axis=2)
        t_b = np.clip(((src[..., 2] - src[..., 0]) - 0.05) / 0.07, 0.0, 1.0)
        t_l = np.clip((L - 0.16) / 0.12, 0.0, 1.0)
        halo = (t_b * t_b * (3.0 - 2.0 * t_b)) * t_l
        mask = np.maximum(mask, halo)

    # pm (js:4289-4295): blend the source's bright zones + halo back over the target,
    # out = target*(1-M) + src*M  (M broadcast over channels by apply_masked).
    out = apply_masked(tgt, src, mask, invert=False)

    # pm2 (js:4306-4318) — hue-of-the-gain damp (RGB only): wherever the second-pass
    # GAIN is blue-dominant (delta-B exceeds delta-R by 0.015 -> 0.065, smooth), pull
    # that gain back toward the source. Neutral gains keep 100% everywhere.
    if is_rgb:
        t_k = np.clip(
            (((out[..., 2] - src[..., 2]) - (out[..., 0] - src[..., 0])) - 0.015) / 0.050,
            0.0, 1.0,
        )
        k = t_k * t_k * (3.0 - 2.0 * t_k)
        out = out - k[..., None] * (out - src)

    return np.clip(out, 0.0, 1.0)
