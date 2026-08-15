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
    """One frame's pose. ``zoom``/``pan`` and the three rotations transform the
    background: ``roll`` = Z (in-plane), ``rot_x`` / ``rot_y`` = 3D tilt of the
    image plane (perspective). ``c`` advances the star inflow (the starfield stays
    centred on the viewport)."""
    zoom: float = 1.0
    roll: float = 0.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    rot_x: float = 0.0
    rot_y: float = 0.0
    c: float = 0.0
    twinkle: float = 0.0


def _catmull(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def _smoother(u):
    return u * u * u * (u * (u * 6 - 15) + 10)


def _pan_path(points, n, ellipse_r=0.03, defaults=(1.4, 0.0, 0.0, 0.0)):
    """Smooth closed path through ``points`` — each ``[x, y, zoom, roll, rot_x,
    rot_y]`` (frame-normalised [-1,1]; channels default to ``defaults``). Returns
    (pan[n,2], chan[n,4]) with chan = zoom, roll, rot_x, rot_y interpolated the
    same way. Point 1's channels are the starting values; with ≥2 points every
    channel loops back to point 1 (closed Catmull → seamless)."""
    dz, dr, dx, dy = defaults
    src = points if points else [[0.0, 0.0, dz, dr, dx, dy]]
    xy = np.array([(p[0], p[1]) for p in src], dtype=float)

    def ch(p, idx, d):
        return p[idx] if len(p) > idx else d
    C = np.array([[ch(p, 2, dz), ch(p, 3, dr), ch(p, 4, dx), ch(p, 5, dy)]
                  for p in src], dtype=float)             # (k, 4) zoom/roll/rx/ry
    k = len(xy)
    if k == 1:                                            # one-shot: zoom into the point
        u = np.linspace(0.0, 1.0, n, endpoint=False)
        pan = np.repeat(xy, n, axis=0)                   # stay EXACTLY on the point
        chan = np.repeat(C, n, axis=0)
        chan[:, 0] = 1.0 + (C[0, 0] - 1.0) * _smoother(u)   # zoom eases in from 1
    else:                                                 # keyframe every channel
        base = np.zeros((n, 2))
        chan = np.zeros((n, 4))
        # segment-based timing: point `seg` lands EXACTLY on frame round(seg*n/k),
        # so the background is perfectly centred on each point (no wobble offset)
        for seg in range(k):
            j0 = int(round(seg * n / k))
            j1 = int(round((seg + 1) * n / k))
            span = max(j1 - j0, 1)
            for j in range(j0, min(j1, n)):
                tt = (j - j0) / span
                base[j] = _catmull(xy[(seg - 1) % k], xy[seg], xy[(seg + 1) % k],
                                   xy[(seg + 2) % k], tt)
                chan[j] = _catmull(C[(seg - 1) % k], C[seg], C[(seg + 1) % k],
                                   C[(seg + 2) % k], tt)
        pan = base
    chan[:, 0] = np.clip(chan[:, 0], 0.2, 8.0)           # zoom bounds
    return pan.astype(np.float32), chan.astype(np.float32)


class V2Fly:
    """Original image as a zoom/rotate/pan background + a star field flying in."""

    def __init__(self, img: np.ndarray, *, out_w: Optional[int] = None,
                 out_h: Optional[int] = None, star_count: int = 1400, seed: int = 7,
                 bloom: float = 0.25, overscan: float = 1.06,
                 focal: float = 0.6, near: float = 0.32, z_span: float = 2.6,
                 star_min: float = 0.9, star_max: float = 3.6, star_gain: float = 5.0,
                 streaks: bool = False, streak_len: float = 60.0) -> None:
        self.bg = _as_rgb(img).astype(np.float32)        # ORIGINAL, unchanged
        self.bh, self.bw = self.bg.shape[:2]
        self.out_w = int(out_w) if out_w else self.bw
        self.out_h = int(out_h) if out_h else self.bh
        self.overscan = float(overscan)
        self.bloom_strength = float(bloom)
        self.focal, self.near, self.z_span = float(focal), float(near), float(z_span)
        self.star_min, self.star_max = float(star_min), float(star_max)
        self.star_gain = float(star_gain)
        self.streaks = bool(streaks)
        self.streak_len = float(streak_len)
        self._build_stars(star_count, seed)

    def _build_stars(self, count, seed):
        rng = np.random.default_rng(seed)
        self.star_u = rng.uniform(-1.7, 1.7, count).astype(np.float32)
        self.star_v = rng.uniform(-1.3, 1.3, count).astype(np.float32)
        # depth phase in [0,1) — the field wraps, so the inflow loops seamlessly
        self.star_z = rng.uniform(0.0, 1.0, count).astype(np.float32)
        flux = 0.28 + 0.72 * rng.power(0.5, count)       # brighter floor → visible w/o bloom
        self.star_flux = flux.astype(np.float32)
        # size rank in [0,1] — mostly small, a few big; brighter stars trend bigger
        rank = np.clip(0.6 * rng.power(2.2, count) + 0.4 * flux, 0.0, 1.0)
        self.star_rank = rank.astype(np.float32)
        t = rng.uniform(0.0, 1.0, count)
        col = np.stack([0.72 + 0.28 * t, 0.80 + 0.14 * (1 - np.abs(t - 0.5) * 2),
                        1.0 - 0.26 * t], axis=1)
        self.star_col = np.clip(col, 0.0, 1.0).astype(np.float32)

    # ---------------------------------------------------------------- bg warp
    def _warp_bg(self, cam: V2Cam) -> np.ndarray:
        """Project the image plane into the output frame with a pinhole camera:
        ``roll`` rotates in-plane (Z), ``rot_x``/``rot_y`` tilt the plane in 3D so
        it keystones (real perspective). A pure Z-roll reduces to the old affine."""
        from scipy.ndimage import map_coordinates

        ow, oh, bw, bh = self.out_w, self.out_h, self.bw, self.bh
        cover = max(ow / bw, oh / bh)
        # clamp zoom-out so the frame can never leave the image (no border shows)
        zmin = max((1.0 + abs(cam.pan_x)) * ow / bw,
                   (1.0 + abs(cam.pan_y)) * oh / bh) / (self.overscan * cover)
        s = self.overscan * cover * max(cam.zoom, zmin)

        rz, rx, ry = (np.deg2rad(cam.roll), np.deg2rad(cam.rot_x),
                      np.deg2rad(cam.rot_y))
        cz, sz = np.cos(rz), np.sin(rz)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        # image rows point down → the in-plane (Z) roll uses the [[c, s],[-s, c]]
        # convention so it matches the legacy 2D affine exactly
        Rz = np.array([[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        R = Rz @ Ry @ Rx                                 # image plane x=col, y=row, z=depth
        D = max(bw, bh) * 1.3                            # camera distance (perspective strength)

        oc_row, oc_col = (oh - 1) / 2.0, (ow - 1) / 2.0
        bc_row, bc_col = (bh - 1) / 2.0, (bw - 1) / 2.0
        # pan focus: the bg point that lands at the output centre (uses the Z-roll)
        m2 = (1.0 / s) * np.array([[cz, sz], [-sz, cz]])
        bc_eff = np.array([bc_row, bc_col]) + m2 @ np.array(
            [cam.pan_y * oh / 2.0, cam.pan_x * ow / 2.0])

        r20, r21 = R[2, 0] / D, R[2, 1] / D
        # forward homography H: plane [X(col), Y(row), 1] → output homog [u, v, w]
        H = np.array([
            [oc_col * r20 + s * R[0, 0], oc_col * r21 + s * R[0, 1], oc_col],
            [oc_row * r20 + s * R[1, 0], oc_row * r21 + s * R[1, 1], oc_row],
            [r20, r21, 1.0],
        ])
        hi = np.linalg.inv(H)
        yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float64)
        wdiv = hi[2, 0] * xx + hi[2, 1] * yy + hi[2, 2]
        px = (hi[0, 0] * xx + hi[0, 1] * yy + hi[0, 2]) / wdiv    # plane X (col)
        py = (hi[1, 0] * xx + hi[1, 1] * yy + hi[1, 2]) / wdiv    # plane Y (row)
        coords = np.stack([bc_eff[0] + py, bc_eff[1] + px])
        out = np.empty((oh, ow, 3), np.float32)
        for ch in range(3):
            out[..., ch] = map_coordinates(self.bg[..., ch], coords, order=1,
                                           mode="nearest", prefilter=False)
        return out

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
        # wrapping depth: each star's distance cycles as c advances, so the inflow
        # is continuous AND loops seamlessly. frac 0 = right in front, 1 = far away.
        frac = np.mod(self.star_z - cam.c, 1.0)
        dist = self.near + frac * self.z_span
        m = self.focal / dist
        # the starfield is centred on the viewport (stars fly straight at the
        # camera); it does NOT follow the background pan
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        th = np.deg2rad(cam.roll)
        cos, sin = np.cos(th), np.sin(th)
        uu = self.star_u * m
        vv = self.star_v * m
        xx = cx + (cos * uu - sin * vv) * (w * 0.5)
        yy = cy + (sin * uu + cos * vv) * (w * 0.5)
        on = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        if not np.any(on):
            return buf
        xx, yy = xx[on], yy[on]
        near_gain = np.clip(m[on] / self.focal, 0.35, 4.0)      # closer = bigger/brighter
        fade = np.clip(1.0 - frac[on], 0.0, 1.0) ** 1.3         # dim near the far wrap
        flux = self.star_flux[on] * near_gain * (0.5 + 0.7 * fade)
        if cam.twinkle:
            idx = np.nonzero(on)[0]
            flux = flux * (1.0 + 0.18 * np.sin(cam.twinkle + idx * 1.7))
        col = self.star_col[on]
        rank = self.star_rank[on] * np.clip(0.7 + 0.5 * (near_gain - 0.4), 0.6, 2.2)
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
            buf += layer * (0.6 + 0.9 * sigma)            # keep big (blurred) stars bright
        return np.clip(buf * self.star_gain, 0.0, 1.0)

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
           rot_x: float = 0.0, rot_y: float = 0.0,
           pan_points: Optional[List[Tuple[float, float]]] = None,
           pan: float = 0.035, star_speed: float = 2.0,
           base_pan: Tuple[float, float] = (0.0, 0.0)):
    """Parametrised v2 path: eased zoom, a smooth (elliptical) pan through
    ``pan_points`` that CLOSES back to point 1, a gentle periodic roll, and a
    wrapping star inflow — so the whole clip loops seamlessly. ``base_pan`` is a
    static offset that frames the background in the output."""
    path, chan = _pan_path(pan_points, n, ellipse_r=max(pan, 0.02),
                           defaults=(zoom_end, rotate_deg, rot_x, rot_y))
    bx, by = base_pan
    cams = []
    for i in range(n):
        u = i / n
        cams.append(V2Cam(
            zoom=float(chan[i, 0]),
            roll=float(chan[i, 1]),                      # all rotations are keyframed
            rot_x=float(chan[i, 2]), rot_y=float(chan[i, 3]),
            pan_x=float(path[i, 0]) + bx, pan_y=float(path[i, 1]) + by,
            c=star_speed * u,                            # wraps in the renderer → loops
            twinkle=u * 2 * np.pi * 6.0,
        ))
    return cams


def render_v2(img, out_path: str, *, seconds: float = 8.0, fps: int = 24,
              out_w: Optional[int] = None, out_h: Optional[int] = None,
              render_width: int = 1280, workers: int = 1, star_count: int = 1400,
              bloom: float = 0.25, zoom_end: float = 1.4, rotate_deg: float = 8.0,
              rot_x: float = 0.0, rot_y: float = 0.0,
              pan_points: Optional[List[Tuple[float, float]]] = None,
              pan: float = 0.035, star_speed: float = 2.0,
              base_pan: Tuple[float, float] = (0.0, 0.0),
              star_min: float = 0.9, star_max: float = 3.6, streaks: bool = False,
              streak_len: float = 60.0, crf: int = 17, bitrate_mbps=None,
              engine_kw=None, on_frame=None) -> str:
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
    cams = fly_v2(n, zoom_end=zoom_end, rotate_deg=rotate_deg, rot_x=rot_x,
                  rot_y=rot_y, pan_points=pan_points, pan=pan,
                  star_speed=star_speed, base_pan=base_pan)
    frames = parallel_frames(eng, cams, int(workers), on_frame)
    return write_video(frames, out_path, fps=fps, crf=crf, bitrate_mbps=bitrate_mbps)
