"""Shadow anchor — a soft black-point pull so the finished sky floor lands near
PixInsight's deep black point instead of a milky elevated band.

Not a port of a specific PI process: a **P1 calibration**, the low-end mirror of
:func:`lazystretch.processes.highlights.highlight_rolloff` (the sanctioned non-PI
finishing-step precedent). The golden survey (``calibration/survey.py``) showed the
port's 1%ile luminance (``blk``) runs ~2x PI on emission / open fields even in execute
mode (m42 .093 vs .029, rosette .171 vs .031, horsehead .140 vs .056): after Background
Neutralization the sky is a tight near-black cluster, the *faithful* shadow clip lands
below it (clips ~nothing), and the MTF hauling median->bkg fans the sky into a milky band
the finishing chain never claws back. The post-stretch NN denoisers (DeepSNR/GraXpert)
fix the *noise* but not this *tone* floor.

This step measures the true sky level (``median_channel_avg`` — PI's own background
statistic) and the current floor (1%ile), then remaps the shadows through an Akima curve
that pulls the milky floor toward black while pinning the background median ``f`` and
leaving everything at/above it **identity** (so object cores / ``clum`` are untouched).
Three properties keep it safe (see the module design notes and the P1 calibration memo):

* **class-gated** — reflection + galaxy shadows are *signal* (Witch Head's outer wings,
  a galaxy's faint halo/dust lanes), so those classes are a hard no-op;
* **self-limiting** — the pull is a fraction (``strength``) of the *excess* over the
  background-relative target ``tau*f``, and no-ops when the floor is already near/below it
  (so already-dark fields and inputStretched polish results are left put);
* **identity at/above ``f``** — only the sub-background sky is remapped.

The single structural limit (flagged by the design review): within one class the finished
``blk`` value carries no information about how much crush PI wants — tulip (blk .140, wants
~none) and horsehead (blk .141, wants a large crush) look identical here. ``tau``/``strength``
bound the worst case to a cosmetic shadow over-crush (never a core/colour change); they are
calibrated against the golden set to the most aggressive point that keeps every target's
``blk`` from crossing darker-than-PI.
"""
from __future__ import annotations

import numpy as np

from ..stats.measure import median_channel_avg

# Classes whose dark regions carry signal, not empty sky -> hard no-op.
_GATE_CLASSES = frozenset({"reflection", "galaxy"})

# Calibrated against the golden set (calibration/survey.py, execute mode). This is the
# SAFE operating point: the most aggressive (tau, strength) at which NO golden target's
# 1%ile crosses darker-than-PI (tulip, the least-milky emission field, is the binding
# constraint — it lands at PI). It cuts the blk gap mean|Δ| ~105% -> ~74% and halves it on
# moderately-milky fields while leaving cores (clum) bit-identical. The very-milky targets
# (rosette PI blk 0.031, horsehead 0.056) are only PARTIALLY reachable: pulling them the
# rest of the way would over-crush tulip below PI (the emission blk-discriminator wall —
# within one class the finished floor value can't distinguish a milky band from a
# legitimately-bright one). See [[lazystretch-p1-core-brightness-gap]].
_TAU: float = 0.65       # black-point target as a fraction of the background median f
_STRENGTH: float = 0.70  # fraction of the floor's excess-over-target that is removed
_MARGIN: float = 1.0e-3  # near-target no-op (idempotency / already-dark guard)


def shadow_anchor_applies_to(object_class: str) -> bool:
    """False for classes whose shadows are signal (reflection, galaxy)."""
    return object_class not in _GATE_CLASSES


def shadow_anchor(img, object_class: str, *, tau: float = _TAU,
                  strength: float = _STRENGTH, log=None) -> np.ndarray:
    """Pull a milky sky floor toward true black without dimming cores.

    Parameters
    ----------
    img : numpy.ndarray
        Finished (non-linear) image in [0, 1], mono (H, W) or RGB (H, W, 3).
    object_class : str
        Object class; ``reflection``/``galaxy`` are a no-op (shadows are signal).
    tau, strength : float
        Black-point target ``tau*f`` (f = background median) and the fraction of the
        floor's excess over that target to remove. Module defaults are golden-calibrated.

    Returns
    -------
    numpy.ndarray
        A new array; an unmodified copy when the guards fire (no-op).
    """
    a = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)

    def _note(m):
        if log:
            log(m)

    if not shadow_anchor_applies_to(object_class):
        _note(f"   shadow anchor: {object_class} shadows are signal -> no-op")
        return a.copy()

    f = median_channel_avg(a)                    # PI's background statistic
    if not (f > 1.0e-3):
        return a.copy()

    L = a.mean(axis=2) if a.ndim == 3 else a
    blk = float(np.percentile(L, 1.0))
    target = tau * f
    if blk >= f or blk <= target + _MARGIN:      # degenerate, or already dark / idempotent
        _note(f"   shadow anchor: floor {blk:.3f} already near target {target:.3f} -> no-op")
        return a.copy()

    # Self-limiting proportional pull: land the *floor* exactly at ``desired`` (a fraction
    # ``strength`` of the way from the current floor toward the target). PREDICTABLE — the
    # 1%ile maps to ``desired`` by construction, so the safe operating point (no target
    # crossing darker-than-PI) is findable; the Akima variant over-pulled past ``desired``.
    desired = blk - strength * (blk - target)

    # Smooth shadow expansion pivoting at the background median f: rescale each pixel's
    # depth-below-f so a=blk -> desired and a=f -> f with UNIT slope at f (no kink, no
    # posterizing hard clip until the sub-floor tail). d2 is quadratic in depth, so the
    # darkest sub-floor pixels are pushed further down (deep blacks) while f is pinned.
    d0 = f - blk                                 # reference depth (>0)
    kk = (f - desired) / d0                       # expansion factor (>= 1)
    d = f - a                                     # depth below background (may be < 0 above f)
    d2 = d * (1.0 + (kk - 1.0) * d / d0)
    pulled = f - d2
    out = np.where(a < f, np.clip(pulled, 0.0, 1.0), a)   # identity at/above f -> cores untouched
    _note(f"   shadow anchor: floor {blk:.3f} -> {desired:.3f} (bg {f:.3f}, target {target:.3f}, "
          f"tau {tau:.2f} strength {strength:.2f})")
    return np.clip(out, 0.0, 1.0)
