"""ABE-style background / gradient extraction — the polynomial-surface fallback.

Ported from PixInsight LazyStretch v1.2.1:
  * ``Pipe.backgroundExtract`` — LazyStretch.js:2303-2323
  * ``Pipe.backgroundCorrect`` — LazyStretch.js:2329-2349

Source (LazyStretch.js:2306-2321)::

    let P = new AutomaticBackgroundExtractor;
    P.polyDegree        = 4;
    P.tolerance         = 1.0;    // sample-selection tuning (AutoDBE-proven) so
    P.deviation         = 0.8;    // bright targets like M42 don't corrupt the fit —
    P.unbalance         = 1.8;    // reject samples deviating from the background model
    P.minBoxFraction    = 0.05;
    P.boxSize           = 5;
    P.boxSeparation     = 5;
    P.targetCorrection  = Subtract;
    P.normalize         = true;
    P.discardModel      = true;
    P.replaceTarget     = true;   // correct the target in place
    P.executeOn( view );

What ABE actually does — and what we reproduce
----------------------------------------------
``AutomaticBackgroundExtractor`` places a grid of small sample boxes over the
image, takes a robust statistic (the box median) as each sample, iteratively
**rejects** samples that deviate from the current background model (the
``tolerance`` / ``deviation`` / ``unbalance`` knobs bias rejection so that bright
structure — stars, nebula cores — is dropped), fits a **degree-4 polynomial
surface** to the survivors *per channel*, and — with ``targetCorrection=Subtract``
and ``normalize=true`` — subtracts the varying part of that surface while keeping
the original background pedestal. This is the portable model PI itself falls back
to; it is what this module implements in pure NumPy via a design-matrix
least-squares fit with iterative sigma rejection.

``Pipe.backgroundCorrect`` prefers the modern **GradientCorrection** process
(a proprietary *multiscale* gradient model whose scripting parameters are
undocumented and whose internal representation has no NumPy equivalent) and only
falls back to ABE if it is absent or errors (LazyStretch.js:2325-2348). There is
no faithful way to port GradientCorrection, so :func:`background_correct` takes
the **ABE polynomial path** — exactly PI's own fallback branch.

Approximation notes (why 'near-exact', not 'exact')
---------------------------------------------------
* ABE's box-sampling geometry (``boxSize``/``boxSeparation``/``minBoxFraction``)
  and its rejection constants (``tolerance``/``deviation``/``unbalance``) are
  internal, undocumented details. We sample the image on a regular grid of block
  medians and reject with an asymmetric MAD-based sigma clip (``kappa_low`` for
  dark samples, ``kappa_high`` for bright ones — the ``unbalance`` idea: reject
  bright deviations harder). The *fitted polynomial* and the subtract+renormalize
  are faithful; the sample-selection tuning is an approximation. Calibrate the
  ``kappa_*`` / ``grid`` defaults against a real PI ABE run if bit-parity matters.
* ``normalize=true`` is read as ABE's documented behaviour: after subtracting the
  model, restore the background pedestal using the **median of the model**, i.e.
  ``out = img - (B - median(B))``. This removes the gradient and leaves the sky at
  its original level (the correct pre-stretch behaviour), rather than crushing the
  background toward zero.
* ``targetCorrection=Subtract`` truncates the corrected image to ``[0, 1]``.
"""
from __future__ import annotations

import numpy as np

# A low-order gradient surface is smooth by construction, so its box-median samples can be
# gathered from a downsampled copy — mathematically the same fit, but ~100× faster than medianing
# full-resolution boxes on a 20-30 MP master (that full-res sampling was the multi-minute
# "background extraction looks stuck" on execute). The surface is still evaluated at full res.
_FIT_SAMPLE_MAXDIM = 1400


def _downsample_for_fit(chan: np.ndarray) -> np.ndarray:
    """Downsample ``chan`` to <= ``_FIT_SAMPLE_MAXDIM`` on its long edge (for sampling only)."""
    H, W = chan.shape
    m = max(H, W)
    if m <= _FIT_SAMPLE_MAXDIM:
        return chan
    from scipy.ndimage import zoom
    f = _FIT_SAMPLE_MAXDIM / float(m)
    return zoom(chan, f, order=1)          # bilinear: averages, diluting stars into the samples


def _nterms(degree: int) -> int:
    """Number of coefficients of a 2-D total-degree-``degree`` polynomial."""
    return (degree + 1) * (degree + 2) // 2


def _poly_terms(degree: int):
    """Exponent pairs ``(i, j)`` for the term ``x**i * y**j`` with ``i + j <= degree``.

    Total-degree ordering (grouped by ``i + j``), matching a standard ABE-style
    polynomial surface of the given degree.
    """
    return [(i, total - i) for total in range(degree + 1) for i in range(total + 1)]


def _design(xn: np.ndarray, yn: np.ndarray, terms) -> np.ndarray:
    """Least-squares design matrix ``(n_samples, n_terms)`` for sample coords.

    ``xn`` / ``yn`` are 1-D normalized ([-1, 1]) coordinate arrays of equal length.
    """
    return np.stack([xn ** i * yn ** j for (i, j) in terms], axis=-1)


def _grid_samples(chan: np.ndarray, grid: int):
    """Box-median samples on a regular ``<= grid x grid`` grid over ``chan``.

    Mirrors ABE's box sampling in spirit: each cell contributes its **median**
    (robust to stars/hot pixels inside the box). Returns pixel-space centers
    ``(xs, ys)`` and the sample values ``zs``.
    """
    H, W = chan.shape
    ny = int(min(grid, H))
    nx = int(min(grid, W))
    y_edges = np.linspace(0, H, ny + 1).astype(int)
    x_edges = np.linspace(0, W, nx + 1).astype(int)

    xs, ys, zs = [], [], []
    for iy in range(ny):
        y0, y1 = int(y_edges[iy]), int(y_edges[iy + 1])
        if y1 <= y0:
            continue
        for ix in range(nx):
            x0, x1 = int(x_edges[ix]), int(x_edges[ix + 1])
            if x1 <= x0:
                continue
            zs.append(float(np.median(chan[y0:y1, x0:x1])))
            xs.append((x0 + x1 - 1) / 2.0)
            ys.append((y0 + y1 - 1) / 2.0)
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def _eval_surface(coef, terms, xn_full: np.ndarray, yn_full: np.ndarray) -> np.ndarray:
    """Evaluate the fitted polynomial over the full pixel grid, memory-cheaply.

    Builds the ``(H, W)`` model by accumulating separable rank-1 terms
    ``c * y**j (col) * x**i (row)`` — never materializes the giant ``(H*W, n_terms)``
    design matrix.
    """
    H, W = yn_full.size, xn_full.size
    max_i = max(i for i, _ in terms)
    max_j = max(j for _, j in terms)
    xpow = [xn_full ** i for i in range(max_i + 1)]  # each (W,)
    ypow = [yn_full ** j for j in range(max_j + 1)]  # each (H,)
    B = np.zeros((H, W), dtype=np.float64)
    for c, (i, j) in zip(coef, terms):
        if c == 0.0:
            continue
        B += c * (ypow[j][:, None] * xpow[i][None, :])
    return B


def _fit_channel(
    chan: np.ndarray,
    degree: int,
    grid: int,
    kappa_low: float,
    kappa_high: float,
    iterations: int,
) -> np.ndarray:
    """Fit the ABE-style polynomial background surface of one channel.

    Returns the model ``B`` (shape of ``chan``). Degree is reduced automatically
    if the grid yields too few samples; a fully degenerate fit falls back to a
    constant (the sample median).
    """
    H, W = chan.shape
    samp = _downsample_for_fit(chan)            # sample the smooth surface cheaply…
    sh, sw = samp.shape
    xs, ys, zs = _grid_samples(samp, grid)
    n = zs.size

    # Drop degree until we have comfortably more samples than coefficients.
    d = int(degree)
    while d > 0 and n < 2 * _nterms(d):
        d -= 1
    terms = _poly_terms(d)

    # Normalize sample coords to [-1, 1] over the (downsampled) image extent for good
    # conditioning of the high powers. Because both sampling and evaluation normalize to the
    # same [-1, 1] domain over the full physical frame, the fit transfers to full resolution.
    scx, scy = (sw - 1) / 2.0, (sh - 1) / 2.0
    ssx = (sw - 1) / 2.0 or 1.0
    ssy = (sh - 1) / 2.0 or 1.0
    xn = (xs - scx) / ssx
    yn = (ys - scy) / ssy

    A_all = _design(xn, yn, terms)
    mask = np.ones(n, dtype=bool)
    coef = None
    # lstsq's internal SVD can set the FPU divide/overflow flags on near-singular
    # inputs; NumPy then reports them (spuriously) on the following matmul. We
    # validate coef finiteness explicitly and clip the output, so silence them.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for _ in range(max(1, iterations)):
            A = A_all[mask]
            if A.shape[0] < A.shape[1]:
                break
            fit, *_ = np.linalg.lstsq(A, zs[mask], rcond=None)
            if not np.all(np.isfinite(fit)):
                break  # ill-conditioned solve — keep the last good fit (or fall back)
            coef = fit

            res = zs - A_all @ coef
            rk = res[mask]
            med = float(np.median(rk))
            sigma = 1.4826 * float(np.median(np.abs(rk - med)))
            if sigma <= 0.0:
                sigma = float(np.std(rk))
            if sigma <= 0.0:
                break  # residuals already flat — nothing left to reject

            # Asymmetric clip: bright samples (positive residual) rejected harder
            # (kappa_high) than dark ones (kappa_low) — the ABE `unbalance` idea.
            newmask = (res <= med + kappa_high * sigma) & (res >= med - kappa_low * sigma)
            # Guard against over-rejection: keep enough to fit, and never drop below
            # ~30% of the samples.
            if newmask.sum() < A.shape[1] or newmask.sum() < 0.3 * n:
                break
            if np.array_equal(newmask, mask):
                break
            mask = newmask

    if coef is None:
        return np.full((H, W), float(np.median(zs)), dtype=np.float64)

    fcx, fcy = (W - 1) / 2.0, (H - 1) / 2.0     # evaluate at FULL resolution ([-1,1] over the frame)
    fsx = (W - 1) / 2.0 or 1.0
    fsy = (H - 1) / 2.0 or 1.0
    xn_full = (np.arange(W, dtype=np.float64) - fcx) / fsx
    yn_full = (np.arange(H, dtype=np.float64) - fcy) / fsy
    return _eval_surface(coef, terms, xn_full, yn_full)


def background_extract(
    img,
    degree: int = 4,
    *,
    grid: int = 64,
    kappa_low: float = 4.0,
    kappa_high: float = 2.0,
    iterations: int = 5,
    normalize: bool = True,
) -> np.ndarray:
    """Fit and subtract a degree-``N`` polynomial background surface (ABE fallback).

    Port of ``Pipe.backgroundExtract`` (LazyStretch.js:2303-2323): per-channel
    degree-4 polynomial surface, sample rejection so bright targets do not corrupt
    the fit, ``Subtract`` correction, ``normalize=true``.

    Parameters
    ----------
    img : numpy.ndarray
        Float64 linear image in ``[0, 1]``, shape ``(H, W)`` mono or ``(H, W, 3)``
        RGB (channels R, G, B == 0, 1, 2). Fit independently per channel.
    degree : int, default 4
        Polynomial total degree (PI ``polyDegree = 4``). Auto-reduced if the grid
        yields too few samples.
    grid : int, keyword, default 64
        Maximum number of sample boxes per axis; each box contributes its median.
    kappa_low, kappa_high : float, keyword
        Asymmetric MAD-sigma rejection thresholds for dark / bright samples. The
        approximation of ABE's ``tolerance``/``deviation``/``unbalance`` tuning:
        bright deviations (stars, nebula) are rejected more aggressively.
    iterations : int, keyword, default 5
        Maximum reject-and-refit passes.
    normalize : bool, keyword, default True
        Mirrors ABE ``normalize=true``: restore the background pedestal using the
        model median, i.e. subtract only the gradient (``img - (B - median(B))``).
        When False, subtract the full model (background pulled toward zero).

    Returns
    -------
    numpy.ndarray
        A NEW corrected array, truncated to ``[0, 1]`` (``targetCorrection=Subtract``).
        The input is never mutated. Applies to mono and RGB alike (this is not a
        colour-only op).
    """
    a = np.asarray(img, dtype=np.float64)

    if a.ndim == 2:
        channels = [a]
    elif a.ndim == 3:
        channels = [a[..., c] for c in range(a.shape[2])]
    else:
        raise ValueError(f"expected a 2-D (mono) or 3-D (RGB) image, got shape {a.shape}")

    corrected = []
    for chan in channels:
        B = _fit_channel(chan, degree, grid, kappa_low, kappa_high, iterations)
        pedestal = float(np.median(B)) if normalize else 0.0
        corrected.append(chan - B + pedestal)  # new array; `chan` view untouched

    out = corrected[0] if a.ndim == 2 else np.stack(corrected, axis=-1)
    return np.clip(out, 0.0, 1.0)


def background_correct(
    img,
    degree: int = 4,
    *,
    grid: int = 64,
    kappa_low: float = 4.0,
    kappa_high: float = 2.0,
    iterations: int = 5,
    normalize: bool = True,
) -> np.ndarray:
    """Background / gradient correction — the ABE polynomial path.

    Port of ``Pipe.backgroundCorrect`` (LazyStretch.js:2329-2349). PI prefers the
    proprietary, undocumented *multiscale* ``GradientCorrection`` process and only
    falls back to ``Pipe.backgroundExtract`` (ABE) when it is unavailable or
    errors. GradientCorrection has no NumPy equivalent, so this port always takes
    PI's own ABE fallback branch — see :func:`background_extract` for the model and
    the accepted parameters.

    Returns
    -------
    numpy.ndarray
        A NEW corrected array in ``[0, 1]`` (the input is never mutated). Unlike the
        JS, which corrects the view in place and returns a ``"GradientCorrection"``/
        ``"ABE"`` label string, this functional port returns the corrected image.
    """
    return background_extract(
        img,
        degree=degree,
        grid=grid,
        kappa_low=kappa_low,
        kappa_high=kappa_high,
        iterations=iterations,
        normalize=normalize,
    )
