"""LazyFlight 3D fly-through — engine unit tests (no ffmpeg needed)."""
import numpy as np
import pytest

from lazystretch.animate import Flythrough3D, Cam, flythrough, flyby, orbit
from lazystretch.animate.depth import depth_field, starless, detect_stars
from lazystretch.animate.clip import build_cameras


def _scene(h=90, w=140):
    """A bright central blob (nebula) + well-separated bright point stars."""
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((yy - h / 2) / (h * 0.28)) ** 2
                    + ((xx - w / 2) / (w * 0.28)) ** 2))
    rgb = np.stack([blob * 0.9, blob * 0.5, blob * 0.35], axis=-1) * 0.7
    # stars on a coarse grid so no two fall inside one opening window; each is a
    # small gaussian PSF (real stars are never a single pixel)
    stars = [(sy, sx) for sy in range(12, h - 6, 20)
             for sx in range(12, w - 6, 22)]
    for sy, sx in stars:
        gy, gx = np.mgrid[sy - 3:sy + 4, sx - 3:sx + 4]
        psf = np.exp(-(((gy - sy) ** 2 + (gx - sx) ** 2) / (2 * 0.9 ** 2)))
        rgb[sy - 3:sy + 4, sx - 3:sx + 4, :] = np.maximum(
            rgb[sy - 3:sy + 4, sx - 3:sx + 4, :], psf[..., None])
    return np.clip(rgb, 0, 1).astype(np.float32), stars


def test_depth_field_shape_and_range():
    rgb, _ = _scene()
    z = depth_field(rgb)
    assert z.shape == rgb.shape[:2]
    assert z.dtype == np.float32
    assert 0.0 <= float(z.min()) and float(z.max()) <= 1.0
    # the bright core should read as nearer than a faint corner
    h, w = z.shape
    assert z[h // 2, w // 2] > z[2, 2]


def test_starless_removes_point_sources():
    rgb, stars = _scene()
    sl = starless(rgb, radius=4)
    lum0 = rgb[..., 0]
    lum1 = sl[..., 0]
    # the bright point sources lose most of their energy...
    drops = [lum1[sy, sx] < lum0[sy, sx] - 0.15 for sy, sx in stars]
    assert sum(drops) >= len(stars) - 1
    # ...while the nebula core is essentially preserved
    h, w = lum0.shape
    assert abs(float(lum1[h // 2, w // 2]) - float(lum0[h // 2, w // 2])) < 0.05


def test_detect_stars_finds_planted_stars():
    rgb, stars = _scene()
    found = detect_stars(rgb, max_stars=100)
    assert found.shape[1] == 6
    assert len(found) >= len(stars) - 2
    # each record: y,x in bounds, flux>0, colour in [0,1]
    assert (found[:, 0] >= 0).all() and (found[:, 1] >= 0).all()
    assert (found[:, 2] > 0).all()
    assert (found[:, 3:] >= 0).all() and (found[:, 3:] <= 1.0 + 1e-4).all()


def test_render_frame_shape_and_bounds():
    rgb, _ = _scene()
    eng = Flythrough3D(rgb, max_stars=100)
    fr = eng.render_frame(Cam(zoom=1.0))
    assert fr.shape == rgb.shape
    assert fr.dtype == np.float32
    assert float(fr.min()) >= 0.0 and float(fr.max()) <= 1.0
    assert float(fr.max()) > 0.05                       # not a black frame


def test_parallax_produces_motion():
    """A dolly-in must change the frame — near sheets/stars move more than far."""
    rgb, _ = _scene()
    eng = Flythrough3D(rgb, max_stars=150)
    a = eng.render_frame(Cam(zoom=1.0))
    b = eng.render_frame(Cam(zoom=1.35, pan_x=0.03))
    diff = float(np.mean(np.abs(a - b)))
    assert diff > 1e-3, f"frames barely changed (diff={diff})"


def test_no_nan_from_nan_input():
    rgb, _ = _scene()
    rgb[10, 10, :] = np.nan
    eng = Flythrough3D(rgb, max_stars=50)
    fr = eng.render_frame(Cam(zoom=1.2))
    assert np.isfinite(fr).all()


@pytest.mark.parametrize("name,fn", [("flythrough", flythrough),
                                     ("flyby", flyby), ("orbit", orbit)])
def test_camera_paths(name, fn):
    cams = fn(24)
    assert len(cams) == 24
    assert cams[0].zoom == pytest.approx(1.0, abs=1e-6)   # starts at rest
    assert cams[-1].zoom > 1.0                            # ends pushed in
    assert build_cameras(name, 10)                        # registered in PATHS


def test_build_cameras_rejects_unknown_path():
    with pytest.raises(ValueError):
        build_cameras("warp-drive", 10)
