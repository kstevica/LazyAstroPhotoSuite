"""PixInsight midtones-transfer-function primitives — the pinned numerical core.

Ported from PixInsight LazyStretch v1.2.1:
  * ``Math.mtf``            — PJSR built-in (standard PixInsight MTF; see note below)
  * ``LS.findMidtonesBalance`` — LazyStretch.js:2063-2079

Every stretch, background pull and auto-assessment score flows through :func:`mtf`,
so a mismatch here fails *all* golden tests at once. Keep it bit-faithful (PLAN
§4.1, §9.2).

.. note::
   ``Math.mtf`` is a PJSR built-in and is NOT defined in the source. The standard
   PixInsight Midtones Transfer Function is pinned here::

       mtf(m, x) = (m - 1) * x / ((2m - 1) * x - m)

   with ``mtf(m, 0) = 0``, ``mtf(m, 1) = 1`` and ``mtf(m, m) = 0.5``. This must be
   confirmed against a real PI install before P0 closes (PLAN §11).
"""
from __future__ import annotations

import numpy as np

# findMidtonesBalance bisection tolerance (LazyStretch.js:2068).
FIND_MIDTONES_EPS: float = 5.0e-5

# Paranoia cap on the bisection loop. The source uses an unbounded ``for(;;)`` and
# relies on guaranteed convergence (the target is always reachable within the chosen
# bracket). Real inputs converge in <60 iterations; this only guards against a bug.
_MAX_BISECT_ITERS: int = 200


def mtf(m: float, x):
    """Midtones Transfer Function, vectorized over ``x``.

    Parameters
    ----------
    m : float
        Midtones balance in ``(0, 1)``.
    x : float or numpy.ndarray
        Input value(s). Values ``<= 0`` map to 0 and ``>= 1`` map to 1, exactly as
        PI's HistogramTransformation (shadow=0, highlight=1) does.

    Returns
    -------
    numpy.ndarray
        ``mtf(m, x)`` with the same shape as ``x`` (0-d for scalar input).
    """
    x = np.asarray(x, dtype=np.float64)
    denom = (2.0 * m - 1.0) * x - m
    with np.errstate(divide="ignore", invalid="ignore"):
        y = ((m - 1.0) * x) / denom
    # The only point where denom == 0 is x = m / (2m - 1), which always lies outside
    # [0, 1] for m in (0, 1), so it is masked by the clamps below.
    y = np.where(x <= 0.0, 0.0, y)
    y = np.where(x >= 1.0, 1.0, y)
    return y


def mtf_scalar(m: float, x: float) -> float:
    """Scalar MTF (used by the :func:`find_midtones_balance` bisection loop)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)


def find_midtones_balance(v0: float, v1: float, eps: float = FIND_MIDTONES_EPS) -> float:
    """Solve for the midtones balance ``m`` such that ``mtf(m, v1) == v0``.

    Verbatim port of ``LS.findMidtonesBalance(v0, v1)`` — ``v0`` is the target
    background, ``v1`` the value being mapped. Bisection with the source's bracket
    selection and ``eps`` kept identical for parity (LazyStretch.js:2063-2079).
    """
    if v1 <= 0.0:
        return 0.0
    if v1 >= 1.0:
        return 1.0
    v0 = min(max(v0, 0.0), 1.0)
    if v1 < v0:
        m0, m1 = 0.0, 0.5
    else:
        m0, m1 = 0.5, 1.0
    m = (m0 + m1) / 2.0
    for _ in range(_MAX_BISECT_ITERS):
        m = (m0 + m1) / 2.0
        v = mtf_scalar(m, v1)
        if abs(v - v0) < eps:
            return m
        if v < v0:
            m1 = m
        else:
            m0 = m
    return m  # unreachable for valid inputs; best-effort safety return


def range_clip(v, lo: float, hi: float):
    """``Math.range(v, lo, hi)`` — clamp to ``[lo, hi]`` (== :func:`numpy.clip`)."""
    return np.clip(v, lo, hi)
