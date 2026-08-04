"""BackgroundNeutralization + ColorCalibration (the portable colour-cal path).

Ported from PixInsight LazyStretch v1.2.1:
  * ``Pipe.colorCalibrate`` — LazyStretch.js:2351-2365

Source (LazyStretch.js:2351-2365)::

    Pipe.colorCalibrate = function( view )
    {
       let bn = new BackgroundNeutralization;
       bn.mode             = BackgroundNeutralization.RescaleAsNeeded;
       bn.targetBackground = 0.001;
       bn.executeOn( view );

       let cc = new ColorCalibration;
       cc.whiteLow = 0; cc.whiteHigh = 0.9; cc.whiteUseROI = false;
       cc.structureDetection = true; cc.structureLayers = 5; cc.noiseLayers = 1;
       cc.manualWhiteBalance = false;
       cc.backgroundLow = 0; cc.backgroundHigh = 0.3; cc.backgroundUseROI = false;
       cc.executeOn( view );
       return "BN + ColorCalibration";
    };

This is the *fallback* path used whenever SPCC (Gaia-based spectrophotometric
calibration) is unavailable — ``Pipe.spcc`` (LazyStretch.js:2398-2415) is a wall
tool (needs an astrometric solution + the Gaia DR3 database) and is **not** ported
here. Only the portable BackgroundNeutralization + ColorCalibration pair is.

Fidelity notes
--------------
Both PixInsight processes are closed-source and, for their reference selection,
operate through PI's own robust estimators (BN's sigma-clipped background sample)
and its multiscale structure-detection engine (CC's ``structureDetection`` with
5 structure layers / 1 noise layer). Those internals are not reproduced. What *is*
faithfully reproduced is the visible per-process contract:

  * **BackgroundNeutralization** — a per-channel *additive* offset that drives each
    channel background to ``targetBackground`` (equalising the RGB background so it
    is neutral grey), followed by PI's ``RescaleAsNeeded`` output mode: a single
    global (all-channels) rescale into [0, 1] applied *only* when the offset pushed
    samples out of range. A global rescale is an affine map applied identically to
    every channel, so it preserves the neutralisation (equal backgrounds stay
    equal). The background *estimate* here is the per-channel median (a robust
    background locator on a background-dominated linear frame) rather than BN's
    exact sigma-clipped mean — hence **approximate** on the reference, exact on the
    offset+rescale contract.

  * **ColorCalibration** — a per-channel *multiplicative* white-balance that makes
    the (background-subtracted) white reference neutral. PI's structure detection is
    approximated by taking the brightest structure pixels (a high luminance
    percentile within ``[whiteLow, whiteHigh]``, excluding saturated cores) and
    measuring each channel over that *shared* set of pixel locations — so the
    reference keeps its true colour. Factors are normalised so the strongest channel
    is left at 1.0 and the others are pulled down (``f[c] = min(W')/W'[c] <= 1``),
    which balances colour without brightening past range, matching CC's default
    no-clip behaviour.

Colour-only op: :func:`color_calibrate` (and both sub-steps) are a **no-op on
mono** input — PI's BN/CC require an RGB view — returning an unchanged copy.
"""
from __future__ import annotations

import numpy as np

# --- ColorCalibration structure-detection approximation -------------------------
# PI detects discrete structures (stars / bright nebula) via a 5-layer multiscale
# transform. We approximate "structure" by the brightest luminance pixels inside
# the valid white range. This fraction sets how large the white-reference sample is;
# ~top 1% isolates genuine bright structure while staying statistically stable.
_WHITE_STRUCTURE_FRACTION: float = 0.01
# Never trust a white reference built from fewer than this many pixels.
_MIN_STRUCTURE_PIXELS: int = 64


def _is_rgb(a: np.ndarray) -> bool:
    return a.ndim == 3 and a.shape[2] >= 3


def _rescale_as_needed(a: np.ndarray) -> np.ndarray:
    """PI ``RescaleAsNeeded``: global (all-channels) rescale into [0, 1] iff needed.

    Mirrors ``GenericImage::Rescale()`` — one min/max over the *whole* image, so the
    affine map is identical across channels and preserves colour balance / the
    background neutralisation. A no-op when the samples are already in range.
    """
    mn = float(a.min())
    mx = float(a.max())
    if mn < 0.0 or mx > 1.0:
        if mx > mn:
            return (a - mn) / (mx - mn)
        return np.zeros_like(a)
    return a


def background_neutralize(img, target: float = 0.001) -> np.ndarray:
    """BackgroundNeutralization: align each channel's background to ``target``.

    Faithful port of the visible BN contract (LazyStretch.js:2353-2356,
    ``mode = RescaleAsNeeded``, ``targetBackground = 0.001``): subtract a per-channel
    additive offset so every channel background lands on ``target`` (neutral grey),
    then apply the global ``RescaleAsNeeded`` rescale only if that pushed samples out
    of [0, 1]. The background estimate is the per-channel median (approximation of
    BN's sigma-clipped background sample — see module docstring).

    No-op on mono input (returns an unchanged copy).
    """
    a = np.asarray(img, dtype=np.float64)
    if not _is_rgb(a):
        return a.copy()

    out = a.copy()
    n_ch = a.shape[2]
    for c in range(n_ch):
        bg = float(np.median(a[..., c]))          # robust per-channel background
        out[..., c] = a[..., c] - bg + target      # additive offset -> target
    return _rescale_as_needed(out)


def color_calibration(
    img,
    white_low: float = 0.0,
    white_high: float = 0.9,
    background_low: float = 0.0,
    background_high: float = 0.3,
) -> np.ndarray:
    """ColorCalibration: per-channel white balance from image structure.

    Faithful port of the visible CC contract (LazyStretch.js:2358-2363). PI's
    ``structureDetection`` (5 structure layers, 1 noise layer) is approximated by a
    high-luminance-percentile structure sample; ``whiteLow``/``whiteHigh`` clip the
    white reference (excludes saturated star cores at > 0.9), ``backgroundLow``/
    ``backgroundHigh`` bound the background reference.

    The white reference colour is measured over a **shared** set of bright-structure
    pixel locations (selected on luminance) so channel ratios are meaningful; each
    channel's white value is background-subtracted, then channels are scaled by
    ``f[c] = min_c(W'[c]) / W'[c] <= 1`` (strongest channel unchanged, others pulled
    down) to neutralise the white reference without clipping.

    No-op on mono input, or when the reference is degenerate (no structure pixels,
    or a non-positive background-subtracted white value) — returns an unchanged copy
    so the caller never gets a blown-up frame (robustness for torture-testing).
    """
    a = np.asarray(img, dtype=np.float64)
    if not _is_rgb(a):
        return a.copy()

    n_ch = a.shape[2]
    # Luminance proxy (equal-weight, PI default RGBWS) used both to *locate* the
    # structure/background samples and to keep channel selection consistent.
    lum = a[..., :n_ch].mean(axis=2)

    # --- White reference: brightest structure inside [white_low, white_high] ------
    valid = (lum >= white_low) & (lum <= white_high)
    if not np.any(valid):
        return a.copy()
    lum_valid = lum[valid]
    n_valid = lum_valid.size
    # Keep the top fraction of valid-luminance pixels (>= a hard floor of pixels).
    k = max(_MIN_STRUCTURE_PIXELS, int(np.ceil(_WHITE_STRUCTURE_FRACTION * n_valid)))
    k = min(k, n_valid)
    # k-th largest luminance among valid pixels = threshold for the structure mask.
    thr = float(np.partition(lum_valid, n_valid - k)[n_valid - k])
    struct = valid & (lum >= thr)
    if int(np.count_nonzero(struct)) < _MIN_STRUCTURE_PIXELS:
        return a.copy()

    # --- Background reference: pixels inside [background_low, background_high] -----
    bg_mask = (lum >= background_low) & (lum <= background_high)

    # Per-channel white value W[c] and background value B[c] over the SAME masks.
    w = np.empty(n_ch, dtype=np.float64)
    b = np.empty(n_ch, dtype=np.float64)
    for c in range(n_ch):
        chan = a[..., c]
        w[c] = float(chan[struct].mean())
        b[c] = float(np.median(chan[bg_mask])) if np.any(bg_mask) else float(np.median(chan))

    # Background-subtracted white reference (true colour of the bright structure).
    w_prime = w - b
    if np.any(w_prime <= 0.0):
        return a.copy()  # degenerate reference -> skip rather than blow up

    # Normalise so the strongest channel stays at 1.0, others pulled down (no clip).
    f = w_prime.min() / w_prime
    # Apply the white balance to the SIGNAL ABOVE BACKGROUND only, and restore the
    # background: out = b + (a - b) * f. Scaling the whole channel (a * f) would
    # re-tint the just-neutralised background by the per-channel factor difference —
    # invisible near black, but the auto-stretch then amplifies it ~250x into a
    # strong colour cast (diagnosed on a neutral OSC Horsehead master). Anchoring on
    # the per-channel background keeps the sky neutral while balancing the structure.
    out = a.copy()
    for c in range(n_ch):
        out[..., c] = b[c] + (a[..., c] - b[c]) * f[c]
    np.clip(out, 0.0, 1.0, out=out)
    return out


def color_calibrate(img, bn_target: float = 0.001) -> np.ndarray:
    """BN + ColorCalibration (``Pipe.colorCalibrate``, LazyStretch.js:2351-2365).

    Runs :func:`background_neutralize` (aligning channel backgrounds to
    ``bn_target``) then :func:`color_calibration` (structure-based white balance),
    in that order — matching the source. This is the *portable* colour-cal path;
    SPCC is intentionally not implemented (wall tool).

    Parameters
    ----------
    img
        float64 array in [0, 1]; mono ``(H, W)`` or RGB ``(H, W, 3)`` with channel
        order R, G, B == index 0, 1, 2.
    bn_target
        BackgroundNeutralization target background. PI hard-codes ``0.001``
        (LazyStretch.js:2355); exposed for testing/calibration.

    Returns
    -------
    np.ndarray
        A new array (input never mutated). Mono input is returned as an unchanged
        copy (colour-only no-op). RGB output is in [0, 1].
    """
    a = np.asarray(img, dtype=np.float64)
    if not _is_rgb(a):
        return a.copy()
    out = background_neutralize(a, target=bn_target)
    out = color_calibration(out)
    # Re-neutralise after CC: its per-channel factors can leave a tiny residual
    # background tint, and the near-black linked auto-stretch amplifies even a ~5%
    # channel imbalance into a visible cast. A final BN guarantees an exactly-neutral
    # sky entering the stretch, which a linked stretch then preserves.
    out = background_neutralize(out, target=bn_target)
    return out
