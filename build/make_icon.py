"""Generate build/appicon.icns — the macOS app icon.

A dark squircle starfield with a hero star carrying 4 diffraction spikes, rendered with the
suite's own spikes engine (lazystretch.develop.spikes). Run with the build venv:

    .venv-build/bin/python build/make_icon.py

Requires macOS `iconutil` + `sips` (both ship with the OS). Produces build/appicon.icns,
which build/build_macos.sh picks up via --macos-app-icon.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from lazystretch.develop.spikes import render_spikes

HERE = os.path.dirname(os.path.abspath(__file__))
S = 1024


def _art() -> Image.Image:
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    r = np.hypot(xx - S / 2, yy - S / 2) / (S * 0.7)
    base = np.clip(0.16 - 0.13 * r, 0.02, 0.16)
    img = np.stack([base * 0.55, base * 0.62, base * 1.0], -1)     # deep blue ground
    rng = np.random.default_rng(7)
    for _ in range(120):                                           # faint star dust
        cx, cy = rng.integers(0, S), rng.integers(0, S)
        a, s = rng.uniform(0.05, 0.25), rng.uniform(1.2, 2.2)
        img += (a * np.exp(-(((xx - cx) / s) ** 2 + ((yy - cy) / s) ** 2)))[..., None]
    stars = [{"x": 0.5, "y": 0.5, "len": 0.34, "flux": 1.0, "col": [1, 1, 1]},
             {"x": 0.30, "y": 0.34, "len": 0.12, "flux": 0.6, "col": [1, 0.85, 0.7]},
             {"x": 0.70, "y": 0.66, "len": 0.14, "flux": 0.7, "col": [0.75, 0.85, 1.0]}]
    img = render_spikes(np.clip(img, 0, 1), stars, count=4, thickness=2.2, intensity=1.3,
                        colored=True, fringe=0.35)
    rgb = (np.clip(img, 0, 1) ** (1 / 1.5) * 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").convert("RGBA")


def main() -> None:
    art = _art()
    pad, rad = 100, 200                                            # macOS squircle w/ padding
    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([pad, pad, S - pad, S - pad], radius=rad, fill=255)
    canvas.paste(art, (0, 0), mask)

    iconset = os.path.join(HERE, "appicon.iconset")
    os.makedirs(iconset, exist_ok=True)
    for sz in (16, 32, 64, 128, 256, 512, 1024):
        canvas.resize((sz, sz), Image.LANCZOS).save(f"{iconset}/icon_{sz}x{sz}.png")
        if sz <= 512:
            canvas.resize((sz * 2, sz * 2), Image.LANCZOS).save(f"{iconset}/icon_{sz}x{sz}@2x.png")
    out = os.path.join(HERE, "appicon.icns")
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
