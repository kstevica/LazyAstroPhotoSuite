"""3D space fly-through — recognisable nebula + synthetic 3D star field + haze.

Lessons from the pure-volume experiment: a homogeneous slab volume that carries
the nebula's real detail *radial-streaks* under perspective (every fine feature
lives in every slab at a different scale) and blows out. So this composes three
layers that each stay clean:

* **Nebula** — the real starless M42, depth-warped by the mask-driven relief
  (the proven, streak-free P1 warp). Recognisable and crisp. Its colour is
  user-definable: ``saturation`` keeps the real hues, ``stylize`` blends toward a
  deep-blue→purple→magenta space palette.
* **Star field** — all stars removed from the nebula; the biggest real ones kept
  as "hero" stars and the rest a synthetic, fully controllable 3D star cloud that
  parallaxes with true perspective as you fly.
* **Haze** — a few heavily-smoothed 3D-noise slabs give volumetric atmosphere to
  fly through; being smooth, they add depth without streaking.

A gentle dolly keeps the object recognisable throughout (the framing the user
asked for).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .depth import _as_rgb, _luma, detect_stars, starless
from .render import Cam, Flythrough3D

# deep-blue → purple → magenta → pink → warm-white (the stylised space palette)
_PALETTE: List[Tuple[float, Tuple[float, float, float]]] = [
    (0.00, (0.05, 0.09, 0.32)),
    (0.35, (0.28, 0.13, 0.46)),
    (0.62, (0.66, 0.20, 0.52)),
    (0.82, (0.96, 0.52, 0.58)),
    (1.00, (1.00, 0.93, 0.86)),
]


def _ramp(lum: np.ndarray, palette) -> np.ndarray:
    ps = np.array([p for p, _ in palette])
    cs = np.array([c for _, c in palette])
    out = np.empty(lum.shape + (3,), np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(lum, ps, cs[:, ch])
    return out


def color_grade(gas: np.ndarray, saturation: float, stylize: float) -> np.ndarray:
    """User-definable gas colour: faithful (saturation) ↔ stylised palette.

    The stylised path remaps only the *hue* through the palette while preserving
    each pixel's original brightness — so empty space (luminance ~0) stays black
    instead of taking the palette's dark-end colour.
    """
    lum = _luma(gas)
    gl = lum[..., None]
    real = np.clip(gl + saturation * (gas - gl), 0.0, 1.0)
    if stylize <= 0.0:
        return real.astype(np.float32)
    hue = _ramp(np.clip(lum, 0, 1), _PALETTE)           # target colour by brightness
    hue = hue / (_luma(hue)[..., None] + 1e-6)          # normalise to unit brightness
    styl = hue * gl                                      # re-apply original luminance
    return np.clip((1.0 - stylize) * real + stylize * styl, 0, 1).astype(np.float32)


def _fbm3d(shape, octaves=4, seed=7):
    """Small smooth 3D fractal-noise volume in [0,1], coherent in z."""
    from scipy.ndimage import zoom

    k, h, w = shape
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        f = 2 ** o
        gz, gy, gx = max(3, 2 * f + 1), max(4, 4 * f), max(6, 6 * f)
        base = rng.random((gz, gy, gx)).astype(np.float32)
        order = 3 if min(gz, gy, gx) >= 4 else 1
        up = zoom(base, (k / gz, h / gy, w / gx), order=order, mode="reflect")
        sl = tuple(slice(0, min(a, b)) for a, b in zip(shape, up.shape))
        out[sl] += amp * up[sl]
        amp, norm = amp * 0.5, norm + amp
    out /= max(norm, 1e-6)
    return np.clip(out, 0.0, 1.0)


@dataclass
class VolCam:
    """Fly pose: ``c`` is forward position (0 = start). ``zoom`` drives the
    nebula parallax; ``pan/roll`` add gentle motion."""
    c: float = 0.0
    zoom: float = 1.0
    pan_u: float = 0.0
    pan_v: float = 0.0
    roll: float = 0.0
    twinkle: float = 0.0


class SpaceFly:
    """Recognisable nebula + synthetic 3D stars + volumetric haze."""

    def __init__(self, img: np.ndarray, masks: Optional[Dict[str, np.ndarray]] = None,
                 *, saturation: float = 1.35, stylize: float = 0.0,
                 star_count: int = 1300, hero_stars: int = 40,
                 haze: float = 0.22, haze_slabs: int = 8, seed: int = 7,
                 focal: float = 1.0, near: float = 1.0, bloom: float = 0.32) -> None:
        rgb = _as_rgb(img)
        self.h, self.w = rgb.shape[:2]
        gas = starless(rgb)                              # remove ALL stars
        colored = color_grade(gas, saturation, stylize)

        # crisp, recognisable nebula via the proven mask-driven depth warp
        # (stars disabled — we build our own 3D field)
        self._neb = Flythrough3D(colored, masks=masks, max_stars=0, bloom=0.0)

        self.focal, self.near = float(focal), float(near)
        self.bloom_strength = float(bloom)
        self.haze_strength = float(haze)
        self._build_haze(colored, masks, haze_slabs, seed)
        self._build_stars(rgb, star_count, hero_stars, seed)

    # ------------------------------------------------------------------ haze
    def _build_haze(self, colored, masks, slabs, seed):
        from scipy.ndimage import gaussian_filter

        h, w = self.h, self.w
        lum = _luma(colored)
        env = None
        if masks:
            neb = masks.get("Nebulosity")
            if neb is not None and np.shape(neb) == lum.shape:
                env = np.asarray(neb, np.float32)
        if env is None:
            env = np.clip(lum / (np.percentile(lum, 99) + 1e-6), 0, 1).astype(np.float32)
        env = gaussian_filter(env, max(h, w) / 90.0)     # broad, smooth → no streak
        noise = _fbm3d((slabs, h, w), octaves=4, seed=seed + 3)
        r = max(h, w) / 80.0
        dens = np.empty((slabs, h, w), np.float32)
        for k in range(slabs):
            dens[k] = gaussian_filter(env * noise[k], r)  # heavy smooth kills streaks
        self.haze_dens = dens
        self.haze_color = colored.astype(np.float32)
        zk = np.linspace(0.0, 1.0, slabs)
        self.haze_z = (0.1 + 0.7 * zk).astype(np.float32)

    def _render_haze(self, cam: VolCam):
        from scipy.ndimage import affine_transform

        if self.haze_strength <= 0:
            return np.zeros((self.h, self.w, 3), np.float32)
        h, w = self.h, self.w
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        cvec = np.array([cy, cx])
        color = np.zeros((h, w, 3), np.float32)
        trans = np.ones((h, w), np.float32)
        dist = self.haze_z - cam.c + self.near
        for k in np.argsort(dist):
            d = float(dist[k])
            if d <= 0.06:
                continue
            m = self.focal / d
            m2 = np.eye(2) / m
            off2 = cvec - m2 @ cvec
            m3 = np.eye(3); m3[:2, :2] = m2
            dk = self.haze_dens[k]
            we = affine_transform(self.haze_color * dk[..., None], m3,
                                  offset=[off2[0], off2[1], 0.0], order=1,
                                  mode="constant", cval=0.0, prefilter=False)
            wa = affine_transform(dk, m2, offset=off2, order=1,
                                  mode="constant", cval=0.0, prefilter=False)
            color += trans[..., None] * we
            trans *= np.clip(1.0 - 2.5 * wa, 0.0, 1.0)
        return np.clip(color * self.haze_strength, 0.0, 1.0)

    # ------------------------------------------------------------------ stars
    def _build_stars(self, rgb, count, hero, seed):
        rng = np.random.default_rng(seed + 1)
        u = rng.uniform(-1.4, 1.4, count)
        v = rng.uniform(-1.0, 1.0, count)
        z = rng.uniform(0.2, 1.6, count)
        flux = 0.08 + 0.92 * rng.power(0.35, count)
        t = rng.uniform(0.0, 1.0, count)
        col = np.stack([0.66 + 0.34 * t, 0.74 + 0.18 * (1 - np.abs(t - 0.5) * 2),
                        1.0 - 0.30 * t], axis=1)
        st = detect_stars(rgb, max_stars=hero)
        if len(st):
            hu = st[:, 1] / self.w * 2.0 - 1.0
            hv = st[:, 0] / self.h * 2.0 - 1.0
            hz = rng.uniform(0.3, 0.7, len(st))
            hflux = 0.7 + 1.5 * (st[:, 2] / (st[:, 2].max() + 1e-6))
            u = np.concatenate([u, hu]); v = np.concatenate([v, hv])
            z = np.concatenate([z, hz]); flux = np.concatenate([flux, hflux])
            col = np.concatenate([col, st[:, 3:6]], axis=0)
        self.star_u = u.astype(np.float32)
        self.star_v = v.astype(np.float32)
        self.star_z = z.astype(np.float32)
        self.star_flux = flux.astype(np.float32)
        self.star_col = np.clip(col, 0.0, 1.0).astype(np.float32)

    def _render_stars(self, cam: VolCam):
        from scipy.ndimage import gaussian_filter

        h, w = self.h, self.w
        buf = np.zeros((h, w, 3), np.float32)
        dist = self.star_z - cam.c + self.near
        vis = dist > 0.1
        if not np.any(vis):
            return buf
        d = dist[vis]
        m = self.focal / d
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        th = np.deg2rad(cam.roll)
        cos, sin = np.cos(th), np.sin(th)
        uu = (self.star_u[vis] + cam.pan_u) * m
        vv = (self.star_v[vis] + cam.pan_v) * m
        xx = cx + (cos * uu - sin * vv) * (w * 0.5)
        yy = cy + (sin * uu + cos * vv) * (w * 0.5)
        on = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        if not np.any(on):
            return buf
        xi = xx[on].astype(np.intp)
        yi = yy[on].astype(np.intp)
        near_gain = np.clip(m[on] / self.focal, 0.5, 2.3)
        flux = self.star_flux[vis][on] * near_gain
        if cam.twinkle:
            idx = np.nonzero(on)[0]
            flux = flux * (1.0 + 0.2 * np.sin(cam.twinkle + idx * 1.7))
        col = self.star_col[vis][on]
        for ch in range(3):
            np.add.at(buf[..., ch], (yi, xi), flux * col[:, ch])
        for ch in range(3):
            buf[..., ch] = gaussian_filter(buf[..., ch], 0.8)
        return np.clip(buf * 2.3, 0.0, 1.0)

    # ------------------------------------------------------------------ render
    def render_frame(self, cam: VolCam) -> np.ndarray:
        neb = self._neb._warp_nebula(Cam(zoom=cam.zoom, pan_x=cam.pan_u,
                                         pan_y=cam.pan_v, roll=cam.roll))
        haze = self._render_haze(cam)
        stars = self._render_stars(cam)
        out = 1.0 - (1.0 - np.clip(neb, 0, 1)) * (1.0 - haze)   # haze over nebula
        out = 1.0 - (1.0 - out) * (1.0 - stars)                 # stars over both
        return np.clip(self._bloom(out), 0.0, 1.0)

    def _bloom(self, out):
        if self.bloom_strength <= 0:
            return out
        from scipy.ndimage import gaussian_filter

        hi = np.clip(out - 0.72, 0.0, 1.0)
        r = max(self.w, self.h) / 175.0
        glow = np.stack([gaussian_filter(hi[..., c], r) for c in range(3)], axis=-1)
        return out + self.bloom_strength * glow


def render_space(img, out_path: str, *, seconds: float = 8.0, fps: int = 24,
                 render_width: int = 1280, workers: int = 1,
                 masks: Optional[Dict[str, np.ndarray]] = None,
                 saturation: float = 1.35, stylize: float = 0.0, haze: float = 0.22,
                 star_count: int = 1300, c_end: float = 0.5, zoom_end: float = 1.22,
                 engine_kw: Optional[dict] = None, on_frame=None) -> str:
    """Render a Space 3D fly-through of ``img`` to an mp4 at ``out_path``."""
    from .clip import _resize
    from .encode import write_video
    from .parallel import parallel_frames

    rgb = _resize(_as_rgb(img), int(render_width))
    eng = SpaceFly(rgb, masks=masks, saturation=saturation, stylize=stylize,
                   haze=haze, star_count=star_count, **(engine_kw or {}))
    n = max(int(round(seconds * fps)), 2)
    cams = fly_volume(n, c_end=c_end, zoom_end=zoom_end)
    frames = parallel_frames(eng, cams, int(workers), on_frame)
    return write_video(frames, out_path, fps=fps)


def fly_volume(n: int, *, c_end: float = 0.5, zoom_end: float = 1.22,
               roll_deg: float = 0.8, sway: float = 0.008):
    """Gentle dolly forward — nebula parallaxes (zoom) while the star field and
    haze stream past (c). The object stays recognisable throughout."""
    cams = []
    for i in range(n):
        u = i / max(n - 1, 1)
        e = u * u * u * (u * (u * 6 - 15) + 10)          # smootherstep
        cams.append(VolCam(
            c=c_end * e,
            zoom=1.0 + (zoom_end - 1.0) * e,
            pan_u=sway * np.sin(u * 2 * np.pi),
            pan_v=sway * 0.7 * np.sin(u * 2 * np.pi + 1.1),
            roll=roll_deg * np.sin(u * np.pi) * 0.5,
            twinkle=u * 2 * np.pi * 6.0,
        ))
    return cams
