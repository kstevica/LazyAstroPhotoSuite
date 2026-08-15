"""v2 fly-through — the simple, fully-parametrised one.

The original image is left completely unchanged and used as a moving background:
each frame it is *zoomed*, *rotated* and *panned* about the centre, so it reads as
the view from a slowly turning "spaceship" drifting toward the object. On top, a
synthetic star field flies *toward* the camera — stars rush outward from the
travel direction and stream past. Everything is a knob:

* **star_min / star_max** — the smallest and largest rendered star size.
* **streaks** — radial motion trails on the fast incoming stars.
* **pan points** — up to 5 frame targets; the background eases smoothly (an
  elliptical, non-abrupt path) so each comes to centre, and the stars change
  direction to follow the turn.
* **out_w / out_h** — the output frame, independent of the image aspect
  (landscape / portrait / any ratio), cover-fit from the image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .depth import _as_rgb


@dataclass
class V2Cam:
    """One frame's pose. ``zoom``/``roll``/``pan`` transform the background;
    ``focus`` shifts the star expansion centre (so stars follow the pan turn);
    ``c`` advances the star inflow."""
    zoom: float = 1.0
    roll: float = 0.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    focus_x: float = 0.0
    focus_y: float = 0.0
    c: float = 0.0
    twinkle: float = 0.0


def _catmull(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def _smoother(u):
    return u * u * u * (u * (u * 6 - 15) + 10)


def _pan_path(points, n, ellipse_r=0.03, zoom_end=1.4):
    """Smooth closed path through ``points`` (frame-normalised [-1,1], each with a
    zoom) sampled at ``n`` frames. Returns (pan[n,2], focus[n,2], zoom[n]). The
    first point's zoom is the starting zoom; the star focus leads the pan
    velocity so the stars change direction on a turn."""
    src = points if points else [(0.0, 0.0, zoom_end)]
    xy = np.array([(p[0], p[1]) for p in src], dtype=float)
    zs = np.array([p[2] if len(p) > 2 else zoom_end for p in src], dtype=float)
    k = len(xy)
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    if k == 1:                                           # zoom in toward the point
        base = np.repeat(xy, n, axis=0)
        zoom = 1.0 + (zs[0] - 1.0) * _smoother(u)
        er = max(ellipse_r, 0.06)
    else:                                                # keyframe path + zoom
        base = np.zeros((n, 2))
        zoom = np.zeros(n)
        for j in range(n):
            s = (j / n) * k                              # closed loop over k segments
            i = int(s) % k
            tt = s - int(s)
            base[j] = _catmull(xy[(i - 1) % k], xy[i], xy[(i + 1) % k],
                               xy[(i + 2) % k], tt)
            zoom[j] = _catmull(zs[(i - 1) % k], zs[i], zs[(i + 1) % k],
                               zs[(i + 2) % k], tt)
        er = ellipse_r
    wob = np.stack([er * np.cos(2 * np.pi * u), er * 0.6 * np.sin(2 * np.pi * u)], 1)
    pan = base + wob
    vel = np.gradient(pan, axis=0) * n
    focus = np.clip(vel * 0.28, -0.6, 0.6)
    return (pan.astype(np.float32), focus.astype(np.float32),
            np.clip(zoom, 0.2, 8.0).astype(np.float32))


class V2Fly:
    """Original image as a zoom/rotate/pan background + a star field flying in."""

    def __init__(self, img: np.ndarray, *, out_w: Optional[int] = None,
                 out_h: Optional[int] = None, star_count: int = 1400, seed: int = 7,
                 bloom: float = 0.25, overscan: float = 1.06,
                 focal: float = 0.55, near: float = 0.35, star_z=(0.12, 2.6),
                 star_min: float = 0.9, star_max: float = 3.6,
                 streaks: bool = False, streak_len: float = 60.0) -> None:
        self.bg = _as_rgb(img).astype(np.float32)        # ORIGINAL, unchanged
        self.bh, self.bw = self.bg.shape[:2]
        self.out_w = int(out_w) if out_w else self.bw
        self.out_h = int(out_h) if out_h else self.bh
        self.overscan = float(overscan)
        self.bloom_strength = float(bloom)
        self.focal, self.near = float(focal), float(near)
        self.star_min, self.star_max = float(star_min), float(star_max)
        self.streaks = bool(streaks)
        self.streak_len = float(streak_len)
        self._build_stars(star_count, seed, star_z)

    def _build_stars(self, count, seed, zr):
        rng = np.random.default_rng(seed)
        self.star_u = rng.uniform(-1.7, 1.7, count).astype(np.float32)
        self.star_v = rng.uniform(-1.3, 1.3, count).astype(np.float32)
        self.star_z = rng.uniform(zr[0], zr[1], count).astype(np.float32)
        flux = 0.1 + 0.9 * rng.power(0.4, count)
        self.star_flux = flux.astype(np.float32)
        # size rank in [0,1] — mostly small, a few big; brighter stars trend bigger
        rank = np.clip(0.6 * rng.power(2.2, count) + 0.4 * flux, 0.0, 1.0)
        self.star_rank = rank.astype(np.float32)
        t = rng.uniform(0.0, 1.0, count)
        col = np.stack([0.70 + 0.30 * t, 0.78 + 0.16 * (1 - np.abs(t - 0.5) * 2),
                        1.0 - 0.28 * t], axis=1)
        self.star_col = np.clip(col, 0.0, 1.0).astype(np.float32)

    # ---------------------------------------------------------------- bg warp
    def _warp_bg(self, cam: V2Cam) -> np.ndarray:
        from scipy.ndimage import affine_transform

        ow, oh = self.out_w, self.out_h
        cover = max(ow / self.bw, oh / self.bh)          # fill the output frame
        s = self.overscan * cover * cam.zoom
        th = np.deg2rad(cam.roll)
        cos, sin = np.cos(th), np.sin(th)
        m2 = (1.0 / s) * np.array([[cos, sin], [-sin, cos]])
        oc = np.array([(oh - 1) / 2.0, (ow - 1) / 2.0])
        bc = np.array([(self.bh - 1) / 2.0, (self.bw - 1) / 2.0])
        offset_out = np.array([cam.pan_y * oh / 2.0, cam.pan_x * ow / 2.0])
        off2 = bc + m2 @ (offset_out - oc)               # centre bg on the pan point
        m3 = np.eye(3)
        m3[:2, :2] = m2
        return affine_transform(self.bg, m3, offset=[off2[0], off2[1], 0.0],
                                order=1, mode="nearest", prefilter=False,
                                output_shape=(oh, ow, 3))

    # ----------------------------------------------------------------- stars
    def _sigmas(self):
        return np.linspace(self.star_min, self.star_max, 4)

    def _deposit(self, buf, xx, yy, flux, col, cx, cy):
        h, w = self.out_h, self.out_w
        if not self.streaks:
            xi = xx.astype(np.intp); yi = yy.astype(np.intp)
            for ch in range(3):
                np.add.at(buf[..., ch], (yi, xi), flux * col[:, ch])
            return
        rx, ry = xx - cx, yy - cy                         # radial from travel focus
        rr = np.hypot(rx, ry) + 1e-6
        dx, dy = rx / rr, ry / rr
        length = np.clip(self.streak_len * (flux ** 0.4 - 0.15), 0.0, self.streak_len)
        steps = 10
        tapers = 0.35 + 0.65 * np.linspace(0.0, 1.0, steps)
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

        h, w = self.out_h, self.out_w
        buf = np.zeros((h, w, 3), np.float32)
        dist = self.star_z - cam.c + self.near
        vis = dist > 0.08
        if not np.any(vis):
            return buf
        d = dist[vis]
        m = self.focal / d
        # star expansion centre tracks the pan (so stars change direction on a turn)
        cx = (w - 1) / 2.0 + cam.focus_x * w / 2.0
        cy = (h - 1) / 2.0 + cam.focus_y * h / 2.0
        th = np.deg2rad(cam.roll)
        cos, sin = np.cos(th), np.sin(th)
        uu = self.star_u[vis] * m
        vv = self.star_v[vis] * m
        xx = cx + (cos * uu - sin * vv) * (w * 0.5)
        yy = cy + (sin * uu + cos * vv) * (w * 0.5)
        on = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        if not np.any(on):
            return buf
        xx, yy = xx[on], yy[on]
        near_gain = np.clip(m[on] / self.focal, 0.4, 3.2)
        flux = self.star_flux[vis][on] * near_gain
        if cam.twinkle:
            idx = np.nonzero(on)[0]
            flux = flux * (1.0 + 0.18 * np.sin(cam.twinkle + idx * 1.7))
        col = self.star_col[vis][on]
        rank = self.star_rank[vis][on] * np.clip(0.7 + 0.5 * (near_gain - 0.4), 0.6, 2.0)
        sigmas = self._sigmas()
        bucket = np.clip((rank * len(sigmas)).astype(int), 0, len(sigmas) - 1)
        for b, sigma in enumerate(sigmas):
            sel = bucket == b
            if not np.any(sel):
                continue
            layer = np.zeros((h, w, 3), np.float32)
            self._deposit(layer, xx[sel], yy[sel], flux[sel], col[sel], cx, cy)
            for ch in range(3):
                layer[..., ch] = gaussian_filter(layer[..., ch], sigma)
            buf += layer
        return np.clip(buf * 3.2, 0.0, 1.0)               # brighter → clearly visible

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
        r = max(self.out_w, self.out_h) / 175.0
        glow = np.stack([gaussian_filter(hi[..., c], r) for c in range(3)], axis=-1)
        return out + self.bloom_strength * glow


def fly_v2(n: int, *, zoom_end: float = 1.4, rotate_deg: float = 8.0,
           pan_points: Optional[List[Tuple[float, float]]] = None,
           pan: float = 0.035, star_speed: float = 1.8,
           base_pan: Tuple[float, float] = (0.0, 0.0)):
    """Parametrised v2 path: eased zoom-in, steady roll, a smooth (elliptical)
    pan through ``pan_points``, and a star inflow whose focus follows the pan.
    ``base_pan`` is a static offset that frames the background in the output."""
    path, focus, zoom = _pan_path(pan_points, n, ellipse_r=max(pan, 0.02),
                                  zoom_end=zoom_end)
    bx, by = base_pan
    cams = []
    for i in range(n):
        u = i / max(n - 1, 1)
        cams.append(V2Cam(
            zoom=float(zoom[i]),
            roll=rotate_deg * u,
            pan_x=float(path[i, 0]) + bx, pan_y=float(path[i, 1]) + by,
            focus_x=float(focus[i, 0]), focus_y=float(focus[i, 1]),
            c=star_speed * u,
            twinkle=u * 2 * np.pi * 6.0,
        ))
    return cams


def render_v2(img, out_path: str, *, seconds: float = 8.0, fps: int = 24,
              out_w: Optional[int] = None, out_h: Optional[int] = None,
              render_width: int = 1280, workers: int = 1, star_count: int = 1400,
              bloom: float = 0.25, zoom_end: float = 1.4, rotate_deg: float = 8.0,
              pan_points: Optional[List[Tuple[float, float]]] = None,
              pan: float = 0.035, star_speed: float = 1.8,
              base_pan: Tuple[float, float] = (0.0, 0.0),
              star_min: float = 0.9, star_max: float = 3.6, streaks: bool = False,
              streak_len: float = 60.0, engine_kw=None, on_frame=None) -> str:
    """Render a v2 fly-through of ``img`` to an mp4 at ``out_path``."""
    from .clip import _resize
    from .encode import write_video
    from .parallel import parallel_frames

    src = _as_rgb(img)
    bh, bw = src.shape[:2]
    if out_w is None:                                    # default: native aspect
        out_w = int(render_width)
        out_h = int(round(out_w * bh / bw))
    out_w = int(out_w) - int(out_w) % 2                  # even dims for h264
    out_h = int(out_h) - int(out_h) % 2
    # resize the source just large enough to cover the output frame at full res
    if (bw / bh) >= (out_w / out_h):                     # image wider → cover by height
        target_w = int(round(out_h * bw / bh))
    else:                                                # image taller → cover by width
        target_w = out_w
    rgb = _resize(src, max(target_w, 8))
    eng = V2Fly(rgb, out_w=out_w, out_h=out_h, star_count=star_count, bloom=bloom,
                star_min=star_min, star_max=star_max, streaks=streaks,
                streak_len=streak_len, **(engine_kw or {}))
    n = max(int(round(seconds * fps)), 2)
    cams = fly_v2(n, zoom_end=zoom_end, rotate_deg=rotate_deg,
                  pan_points=pan_points, pan=pan, star_speed=star_speed,
                  base_pan=base_pan)
    frames = parallel_frames(eng, cams, int(workers), on_frame)
    return write_video(frames, out_path, fps=fps)
