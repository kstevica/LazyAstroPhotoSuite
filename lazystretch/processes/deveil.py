"""De-veil — a user-controlled, colour-preserving background-floor deepener.

The emission stretch parks the sky floor in a milky ~0.10-0.16 band (the P1
"black-point wall": after Background Neutralization the sky is a tight near-black
cluster, the MTF fans it into a milky band, and :func:`shadowanchor.shadow_anchor`
can only claw it back to a *safe global frontier* — pushing further would over-crush
other targets). On rich emission fields (M42, the Horsehead region) that leaves a
brown milky veil the main subject floats in.

Crucially the "brown" is partly REAL faint red nebulosity (integrated flux, faint Hα)
that shares the subject's colour — so *neutralising* the cast greys the actual nebula.
The fix is to attack BRIGHTNESS, not colour: deepen the luminance floor and flatten the
large-scale tilt while keeping R:G:B ratios, so the veil goes dark but the pink survives.

:func:`deepen_background` is the low-end mirror of :func:`shadowanchor.shadow_anchor`
made user-controllable and colour-preserving:

* the pull is applied on **luminance** and the channels are ratio-scaled (hue held);
* it can reach a **deeper target** than the safe global frontier — the user picks how
  far per image, which a global step can't (it would regress other targets);
* it **pins the background median** ``f`` and is identity at/above it, so the nebula
  (which sits above ``f``) is untouched — only the sub-background sky is deepened;
* it removes a gentle **deg-2 tilt** fitted to the low-signal background (a genuine
  lighting gradient, not the localised nebula glow);
* it is **self-limiting** — no-op when the floor is already at/below the target.
"""
from __future__ import annotations

import numpy as np

from ..stats.measure import median_channel_avg


def _fit_gradient(lum: np.ndarray, *, grid: int = 64, keep_frac: float = 0.40) -> np.ndarray:
    """Deg-2 polynomial surface fitted to the LOW-signal (background) samples of ``lum``.

    Samples on a coarse grid, keeps the lower ``keep_frac`` (rejecting stars/nebula), and
    least-squares fits a quadratic — a smooth lighting tilt, never the localised subject.
    """
    H, W = lum.shape
    ys = np.linspace(0, H - 1, grid).astype(int)
    xs = np.linspace(0, W - 1, grid).astype(int)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    V = lum[gy, gx].ravel()
    Xn = gx.ravel() / (W - 1) - 0.5
    Yn = gy.ravel() / (H - 1) - 0.5
    keep = V <= np.quantile(V, keep_frac)
    A = np.stack([np.ones(int(keep.sum())), Xn[keep], Yn[keep],
                  Xn[keep] ** 2, Xn[keep] * Yn[keep], Yn[keep] ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(A, V[keep], rcond=None)
    yy, xx = np.mgrid[0:H, 0:W]
    Xf = xx / (W - 1) - 0.5
    Yf = yy / (H - 1) - 0.5
    return (coef[0] + coef[1] * Xf + coef[2] * Yf
            + coef[3] * Xf ** 2 + coef[4] * Xf * Yf + coef[5] * Yf ** 2)


def deepen_background(img: np.ndarray, amount: float, *, target: float = 0.05,
                      flatten: float = 0.6, log=None) -> np.ndarray:
    """Colour-preservingly deepen a milky background floor toward ``target``.

    ``amount`` (0..1) is the strength: the background level ``f`` is pulled a fraction
    ``amount`` of the way toward the absolute ``target`` floor (adaptive — a milkier frame is
    pulled harder, and SELF-LIMITING — no-op when the sky is already at/below ``target``, so
    it can't over-darken a clean field). A deg-2 background tilt is flattened by
    ``flatten·amount``. Highlights stay pinned at 1 and channels are ratio-scaled from the
    luminance pull, so hue/real faint colour survive. Input never mutated; clipped to [0, 1].
    """
    a = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return a.copy()
    is_rgb = a.ndim == 3
    lum = a[..., :3].mean(axis=2) if is_rgb else a
    f = median_channel_avg(a)
    if not (f > 1.0e-3):
        return a.copy()

    # 1. flatten a gentle deg-2 lighting tilt (from low-signal samples), scaled by amount.
    surf = _fit_gradient(lum)
    grad = np.clip(surf - np.percentile(surf, 5.0), 0.0, None) * (flatten * amount)
    lum_f = lum - grad

    # 2. colour-preserving black-point pull: bring the milky background level f (which the
    #    veil sits AT, so pinning it — as shadow_anchor does — would leave the veil put) down
    #    toward target, rescaling so highlights stay pinned at 1 (subject brightness held).
    #    The pull is a fraction ``amount`` of the way from f to target — adaptive, so a milkier
    #    frame is pulled harder and both land near the same floor. Self-limiting: no-op if the
    #    background is already at/below target.
    c = amount * max(f - target, 0.0)
    if c > 1.0e-4:
        lum2 = np.clip((lum_f - c) / (1.0 - c), 0.0, 1.0)
        if log:
            log(f"   de-veil: bg {f:.3f} -> ~{max(f - c, 0.0) / (1.0 - c):.3f} "
                f"(target {target:.3f}, amount {amount:.2f})")
    else:
        lum2 = np.clip(lum_f, 0.0, 1.0)
        if log:
            log(f"   de-veil: bg {f:.3f} already near target {target:.3f} -> flatten only")

    # 3. colour-preserving reconstruction: ratio-scale channels by the luminance change.
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(lum > 1.0e-4, lum2 / np.where(lum > 1.0e-4, lum, 1.0), 1.0)
    out = a * scale[..., None] if is_rgb else a * scale
    return np.clip(out, 0.0, 1.0)
