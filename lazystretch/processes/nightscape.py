"""Nightscape composite — blend a sharp foreground over the stretched deep sky.

LazyStack emits, for a foreground-locked Milky Way stack, three companions beside the master: the
deep SKY master (the ordinary LazyStretch input, stretched aggressively as the Milky Way), a linear
FOREGROUND layer (the sharp, static, usually moonlit landscape kept from one frame), and a feathered
SKY MASK (1 = sky, 0 = foreground). The two subjects need different development — an aggressive sky
stretch would blow the moonlit foreground — so the foreground is developed GENTLY on its own and
composited into the sky-region-masked result along the feathered seam.
"""
from __future__ import annotations

import numpy as np


def develop_foreground(layer: np.ndarray, brightness: float = 0.5) -> np.ndarray:
    """Render the linear foreground layer as a NATURAL photo — NOT a deep-sky stretch.

    The foreground is a moonlit landscape; it should look like an ordinary (dark) photograph, so it
    gets only a display-gamma-style rendering, never the aggressive stretch the sky gets. ``brightness``
    (0..1) sets a modest target median luminance (~0.05–0.18, a dark-but-visible landscape); the
    luminance is gamma-mapped to reach it, but the gamma is CAPPED so it is never harsher than a
    normal sRGB display curve (≈0.45) — that cap is what stops the wash-out/over-stretch. Channels are
    ratio-scaled so hue survives; a gray-world white balance first neutralizes the raw colour cast.
    Input never mutated; output clipped to [0, 1].
    """
    a = np.clip(np.asarray(layer, dtype=np.float64), 0.0, 1.0)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    # Gray-world white balance: the foreground is a LINEAR raw (strong green cast on X-Trans/OSC);
    # equalize the channel means so a moonlit landscape reads neutral instead of neon-green.
    means = a[..., :3].reshape(-1, 3).mean(axis=0)
    if np.all(means > 1e-6):
        a = np.clip(a * (means.mean() / means), 0.0, 1.0)
    lum = a[..., :3].mean(axis=2)
    pos = lum[lum > 1e-4]
    if pos.size == 0:
        return a
    med = float(np.median(pos))
    target = 0.05 + 0.13 * float(np.clip(brightness, 0.0, 1.0))     # dark, photographic (was 0.12–0.62)
    g = float(np.clip(np.log(max(target, 1e-3)) / np.log(max(med, 1e-3)), 0.35, 0.9))  # never harsher than ~sRGB
    new_lum = np.power(lum, g)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(lum > 1e-4, new_lum / np.where(lum > 1e-4, lum, 1.0), 1.0)
    return np.clip(a * scale[..., None], 0.0, 1.0)


def composite(sky_img: np.ndarray, fg_layer: np.ndarray, sky_mask: np.ndarray,
              brightness: float = 0.5, *, develop: bool = True, feather_frac: float = 0.004
              ) -> np.ndarray:
    """Composite the foreground into the foreground region of the (stretched) sky image.

    ``sky_img`` is the finished sky (Milky Way); ``fg_layer`` the foreground; ``sky_mask`` the feathered
    1=sky/0=foreground weight. Returns ``mask·sky + (1−mask)·foreground``.

    ``develop`` (default True) renders the foreground with :func:`develop_foreground` (for a LINEAR
    raw layer); set it False when the supplied foreground is ALREADY developed (e.g. a finished JPEG in
    Mode 2) so it is used as-is. ``feather_frac`` softens the mask edge by a Gaussian of that fraction
    of the frame's long edge, so the horizon seam has a smooth falloff instead of a hard/stair-stepped
    boundary (0 disables).
    """
    sky = np.clip(np.asarray(sky_img, dtype=np.float64), 0.0, 1.0)
    if sky.ndim == 2:
        sky = np.stack([sky] * 3, axis=-1)
    if develop:
        fg = develop_foreground(fg_layer, brightness)
    else:                                            # already-developed foreground → use as-is
        fg = np.clip(np.asarray(fg_layer, dtype=np.float64), 0.0, 1.0)
        if fg.ndim == 2:
            fg = np.stack([fg] * 3, axis=-1)
    if fg.shape[:2] != sky.shape[:2]:
        return sky                                   # shape mismatch → leave sky untouched (safe no-op)
    m = np.clip(np.asarray(sky_mask, dtype=np.float64), 0.0, 1.0)
    if m.shape[:2] != sky.shape[:2]:
        return sky
    if feather_frac and feather_frac > 0:            # smooth the seam at the working resolution
        from scipy.ndimage import gaussian_filter
        m = np.clip(gaussian_filter(m, float(feather_frac) * max(m.shape[:2])), 0.0, 1.0)
    m = m[..., None]
    return np.clip(m * sky + (1.0 - m) * fg, 0.0, 1.0)
