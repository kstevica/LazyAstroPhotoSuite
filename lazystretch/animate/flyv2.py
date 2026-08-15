"""v2 fly-through — the simple, fully-parametrised one.

The original image is left completely unchanged and used as a moving background:
each frame it is *zoomed*, *rotated* and *panned* about the centre, so it reads as
the view from a slowly turning "spaceship" drifting toward the object. On top, a
synthetic star field flies *toward* the camera — stars rush outward from the
centre and stream past as you move forward. Zoom, rotate and pan are all knobs.

Deliberately separate from the Space-3D engine: no starless, no depth, no volume
— just a background transform plus an incoming star cloud. Cheap and predictable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .depth import _as_rgb


@dataclass
class V2Cam:
    """One frame's pose. ``zoom``/``roll``/``pan`` transform the background;
    ``c`` advances the star inflow (stars fly at you as it grows)."""
    zoom: float = 1.0
    roll: float = 0.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    c: float = 0.0
    twinkle: float = 0.0


class V2Fly:
    """Original image as a zoom/rotate/pan background + a star field flying in."""

    def __init__(self, img: np.ndarray, *, star_count: int = 1400, seed: int = 7,
                 bloom: float = 0.25, overscan: float = 1.08,
                 focal: float = 0.55, near: float = 0.35, star_z=(0.12, 2.6),
                 streaks: bool = False, streak_len: float = 26.0) -> None:
        self.bg = _as_rgb(img).astype(np.float32)        # ORIGINAL, unchanged
        self.h, self.w = self.bg.shape[:2]
        self.overscan = float(overscan)
        self.bloom_strength = float(bloom)
        self.focal, self.near = float(focal), float(near)
        self.streaks = bool(streaks)                     # radial motion-blur trails
        self.streak_len = float(streak_len)
        self._build_stars(star_count, seed, star_z)

    def _build_stars(self, count, seed, zr):
        rng = np.random.default_rng(seed)
        self.star_u = rng.uniform(-1.6, 1.6, count).astype(np.float32)
        self.star_v = rng.uniform(-1.2, 1.2, count).astype(np.float32)
        self.star_z = rng.uniform(zr[0], zr[1], count).astype(np.float32)
        flux = 0.1 + 0.9 * rng.power(0.4, count)
        self.star_flux = flux.astype(np.float32)
        # per-star size: mostly small, a few big; brighter stars trend bigger
        size = 0.35 + 1.0 * rng.power(2.2, count) + 0.5 * flux
        self.star_size = np.clip(size, 0.25, 2.4).astype(np.float32)
        t = rng.uniform(0.0, 1.0, count)
        col = np.stack([0.70 + 0.30 * t, 0.78 + 0.16 * (1 - np.abs(t - 0.5) * 2),
                        1.0 - 0.28 * t], axis=1)
        self.star_col = np.clip(col, 0.0, 1.0).astype(np.float32)

    # ---------------------------------------------------------------- bg warp
    def _warp_bg(self, cam: V2Cam) -> np.ndarray:
        from scipy.ndimage import affine_transform

        h, w = self.h, self.w
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        s = self.overscan * cam.zoom
        th = np.deg2rad(cam.roll)
        cos, sin = np.cos(th), np.sin(th)
        m2 = (1.0 / s) * np.array([[cos, sin], [-sin, cos]])
        t = np.array([cam.pan_y * h, cam.pan_x * w])
        cvec = np.array([cy, cx])
        off2 = cvec - m2 @ (cvec + t)
        m3 = np.eye(3)
        m3[:2, :2] = m2
        return affine_transform(self.bg, m3, offset=[off2[0], off2[1], 0.0],
                                order=1, mode="nearest", prefilter=False)

    # ----------------------------------------------------------------- stars
    _BUCKET_SIGMA = (0.6, 1.1, 1.9)                       # small / medium / large
    _BUCKET_EDGES = (0.8, 1.5)

    def _deposit(self, buf, xx, yy, flux, col):
        """Stamp point flux at (xx, yy) — optionally as a radial motion trail."""
        h, w = self.h, self.w
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        if not self.streaks:
            xi = xx.astype(np.intp); yi = yy.astype(np.intp)
            for ch in range(3):
                np.add.at(buf[..., ch], (yi, xi), flux * col[:, ch])
            return
        rx, ry = xx - cx, yy - cy                         # radial direction outward
        rr = np.hypot(rx, ry) + 1e-6
        dx, dy = rx / rr, ry / rr
        # only faster (nearer) stars streak; length grows with screen speed
        length = np.clip(self.streak_len * (flux ** 0.5 - 0.4), 0.0, self.streak_len)
        steps = 8
        tapers = 0.25 + 0.75 * np.linspace(0.0, 1.0, steps)   # head brighter than tail
        tnorm = float(tapers.sum())
        for k in range(steps):
            f = k / (steps - 1)
            px = xx - dx * length * (1.0 - f)
            py = yy - dy * length * (1.0 - f)
            m = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            if not np.any(m):
                continue
            xi = px[m].astype(np.intp); yi = py[m].astype(np.intp)
            wgt = tapers[k] / tnorm
            for ch in range(3):
                np.add.at(buf[..., ch], (yi, xi), flux[m] * col[m, ch] * wgt)

    def _render_stars(self, cam: V2Cam) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        h, w = self.h, self.w
        buf = np.zeros((h, w, 3), np.float32)
        dist = self.star_z - cam.c + self.near
        vis = dist > 0.08
        if not np.any(vis):
            return buf
        d = dist[vis]
        m = self.focal / d
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        th = np.deg2rad(cam.roll)                         # stars share the roll
        cos, sin = np.cos(th), np.sin(th)
        uu = self.star_u[vis] * m
        vv = self.star_v[vis] * m
        xx = cx + (cos * uu - sin * vv) * (w * 0.5)
        yy = cy + (sin * uu + cos * vv) * (w * 0.5)
        on = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        if not np.any(on):
            return buf
        xx, yy = xx[on], yy[on]
        near_gain = np.clip(m[on] / self.focal, 0.4, 3.0)    # closer = bigger/brighter
        flux = self.star_flux[vis][on] * near_gain
        if cam.twinkle:
            idx = np.nonzero(on)[0]
            flux = flux * (1.0 + 0.18 * np.sin(cam.twinkle + idx * 1.7))
        col = self.star_col[vis][on]
        eff = self.star_size[vis][on] * (0.7 + 0.5 * (near_gain - 0.4))   # size on screen
        bucket = np.digitize(eff, self._BUCKET_EDGES)     # 0 / 1 / 2
        for b, sigma in enumerate(self._BUCKET_SIGMA):
            sel = bucket == b
            if not np.any(sel):
                continue
            layer = np.zeros((h, w, 3), np.float32)
            self._deposit(layer, xx[sel], yy[sel], flux[sel], col[sel])
            for ch in range(3):
                layer[..., ch] = gaussian_filter(layer[..., ch], sigma)
            buf += layer
        return np.clip(buf * 2.2, 0.0, 1.0)

    # ---------------------------------------------------------------- render
    def render_frame(self, cam: V2Cam) -> np.ndarray:
        bg = self._warp_bg(cam)
        stars = self._render_stars(cam)
        out = 1.0 - (1.0 - np.clip(bg, 0, 1)) * (1.0 - stars)
        return np.clip(self._bloom(out), 0.0, 1.0)

    def _bloom(self, out):
        if self.bloom_strength <= 0:
            return out
        from scipy.ndimage import gaussian_filter

        hi = np.clip(out - 0.75, 0.0, 1.0)
        r = max(self.w, self.h) / 175.0
        glow = np.stack([gaussian_filter(hi[..., c], r) for c in range(3)], axis=-1)
        return out + self.bloom_strength * glow


def fly_v2(n: int, *, zoom_end: float = 1.4, rotate_deg: float = 8.0,
           pan: float = 0.035, star_speed: float = 1.6):
    """Parametrised v2 path: eased zoom-in, steady roll, gentle pan drift, and a
    linear star inflow."""
    cams = []
    for i in range(n):
        u = i / max(n - 1, 1)
        ez = u * u * u * (u * (u * 6 - 15) + 10)          # smootherstep zoom
        cams.append(V2Cam(
            zoom=1.0 + (zoom_end - 1.0) * ez,
            roll=rotate_deg * u,                          # steady "spaceship" roll
            pan_x=pan * np.sin(u * np.pi * 0.5),
            pan_y=-pan * 0.55 * u,
            c=star_speed * u,                             # stars fly in
            twinkle=u * 2 * np.pi * 6.0,
        ))
    return cams


def render_v2(img, out_path: str, *, seconds: float = 8.0, fps: int = 24,
              render_width: int = 1280, workers: int = 1, star_count: int = 1400,
              bloom: float = 0.25, zoom_end: float = 1.4, rotate_deg: float = 8.0,
              pan: float = 0.035, star_speed: float = 1.6, streaks: bool = False,
              engine_kw=None, on_frame=None) -> str:
    """Render a v2 fly-through of ``img`` to an mp4 at ``out_path``."""
    from .clip import _resize
    from .encode import write_video
    from .parallel import parallel_frames

    rgb = _resize(_as_rgb(img), int(render_width))
    eng = V2Fly(rgb, star_count=star_count, bloom=bloom, streaks=streaks,
                **(engine_kw or {}))
    n = max(int(round(seconds * fps)), 2)
    cams = fly_v2(n, zoom_end=zoom_end, rotate_deg=rotate_deg, pan=pan,
                  star_speed=star_speed)
    frames = parallel_frames(eng, cams, int(workers), on_frame)
    return write_video(frames, out_path, fps=fps)
