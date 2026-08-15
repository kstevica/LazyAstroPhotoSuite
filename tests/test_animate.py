"""LazyFlight 3D fly-through — engine unit tests (no ffmpeg needed)."""
import numpy as np
import pytest

from lazystretch.animate import (Flythrough3D, Cam, flythrough, flyby, orbit,
                                 pullback)
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


def test_volumetric_mode_renders():
    rgb, _ = _scene()
    eng = Flythrough3D(rgb, max_stars=80, mode="volumetric", vol_slabs=8)
    a = eng.render_frame(Cam(zoom=1.0))
    b = eng.render_frame(Cam(zoom=1.3, pan_x=0.02))
    assert a.shape == rgb.shape and a.dtype == np.float32
    assert np.isfinite(a).all()
    assert 0.0 <= float(a.min()) and float(a.max()) <= 1.0
    assert float(np.mean(np.abs(a - b))) > 1e-3           # camera move changes it


def test_invalid_mode_rejected():
    rgb, _ = _scene()
    with pytest.raises(ValueError):
        Flythrough3D(rgb, mode="hyperdrive")


@pytest.mark.parametrize("name,fn", [("flythrough", flythrough),
                                     ("flyby", flyby), ("orbit", orbit)])
def test_camera_paths(name, fn):
    cams = fn(24)
    assert len(cams) == 24
    assert cams[0].zoom == pytest.approx(1.0, abs=1e-6)   # starts at rest
    assert cams[-1].zoom > 1.0                            # ends pushed in
    assert build_cameras(name, 10)                        # registered in PATHS


def test_spacefly_renders_and_moves():
    from lazystretch.animate.volume3d import SpaceFly, VolCam, fly_volume

    rgb, _ = _scene()
    eng = SpaceFly(rgb, star_count=200, haze_slabs=4)
    a = eng.render_frame(VolCam(c=0.0, zoom=1.0))
    b = eng.render_frame(VolCam(c=0.3, zoom=1.2))
    assert a.shape == rgb.shape and a.dtype == np.float32
    assert np.isfinite(a).all()
    assert 0.0 <= float(a.min()) and float(a.max()) <= 1.0
    assert float(a.max()) > 0.05                          # not black
    assert float(np.mean(np.abs(a - b))) > 1e-3           # fly-in changes it
    cams = fly_volume(24)
    assert len(cams) == 24 and cams[0].c == pytest.approx(0.0)
    assert cams[-1].c > 0.0                               # flew forward


def test_v2fly_bg_unchanged_and_moves():
    from lazystretch.animate.flyv2 import V2Fly, V2Cam, fly_v2

    rgb, _ = _scene()
    eng = V2Fly(rgb, star_count=200)
    # the background is the ORIGINAL image, untouched
    assert np.array_equal(eng.bg, rgb.astype(np.float32))
    a = eng.render_frame(V2Cam(zoom=1.0, roll=0.0))
    b = eng.render_frame(V2Cam(zoom=1.4, roll=8.0, pan_x=0.03, c=1.2))
    assert a.shape == rgb.shape and a.dtype == np.float32
    assert np.isfinite(a).all() and float(a.max()) > 0.05
    assert float(np.mean(np.abs(a - b))) > 1e-3          # zoom/rotate/stars change it
    cams = fly_v2(24, zoom_end=1.5, rotate_deg=10.0, pan=0.04)
    assert len(cams) == 24
    assert cams[0].zoom == pytest.approx(1.0, abs=1e-6)  # zoom eases in from 1
    assert cams[-1].zoom > 1.0
    # with no pan points, roll is a constant orientation from rotate_deg
    assert all(c.roll == pytest.approx(10.0) for c in cams)
    assert cams[-1].c > 0.0                              # stars flew in


def test_v2fly_star_flow_follows_zoom_direction():
    from lazystretch.animate.flyv2 import fly_v2

    # zoom IN → stars stream toward the camera (c advances, flow +1)
    cin = fly_v2(40, zoom_end=1.8)
    assert cin[0].c == pytest.approx(0.0, abs=1e-6)
    assert cin[-1].c > 0.2                                # flew in
    assert np.median([c.flow for c in cin]) > 0.5

    # zoom OUT → stars recede the OTHER way (c retreats, flow -1)
    cout = fly_v2(40, zoom_end=0.5)
    assert cout[-1].c < -0.2                              # receded
    assert np.median([c.flow for c in cout]) < -0.5

    # a two-point clip that dollies in then back out reverses the flow
    pts = [[-0.3, -0.1, 1.0, 0.0, 0.0, 0.0],
           [0.3, 0.1, 2.2, 0.0, 0.0, 0.0]]               # loops back to 1.0 → in, then out
    mid = fly_v2(60, pan_points=pts)
    flows = np.array([c.flow for c in mid])
    cpos = np.array([c.c for c in mid])
    assert flows.max() > 0.3 and flows.min() < -0.3      # both directions occur
    assert float(np.abs(np.diff(flows)).max()) < 1.2     # no hard +1→-1 snap
    # what must be smooth is the star POSITION c: it eases through the turn-around
    # (velocity passes through zero over several frames), so per-frame
    # acceleration stays well under a hard one-frame reversal (~2*pace = 0.066)
    assert float(np.abs(np.diff(cpos, 2)).max()) < 0.04


def test_v2fly_receding_stars_render_and_streaks_flip():
    from lazystretch.animate.flyv2 import V2Fly, V2Cam

    rgb, _ = _scene()
    eng = V2Fly(rgb, star_count=250, streaks=True, streak_len=90.0)
    # negative c (receding) must render cleanly (mod-wrap handles the sign)
    a = eng._render_stars(V2Cam(c=-0.4, flow=-1.0))
    assert np.isfinite(a).all() and float(a.max()) > 0.0
    # streak direction is tied to flow: inflow vs receding trails differ
    fin = eng._render_stars(V2Cam(c=0.4, flow=1.0))
    frec = eng._render_stars(V2Cam(c=0.4, flow=-1.0))     # same positions, opposite trail
    assert float(np.mean(np.abs(fin - frec))) > 1e-4


def test_v2fly_rotations_keyframed_and_loop():
    from lazystretch.animate.flyv2 import fly_v2

    # each point carries its own roll / tilt-x / tilt-y; the camera interpolates
    # them (like zoom) and loops back to point 1
    pts = [[-0.4, -0.2, 1.1, 0.0, 0.0, 0.0],
           [0.3, 0.1, 1.6, 10.0, 3.0, -2.0],
           [0.0, 0.4, 2.0, -6.0, -3.0, 1.0]]
    cams = fly_v2(60, pan_points=pts)
    rolls = [c.roll for c in cams]
    assert cams[0].roll == pytest.approx(0.0, abs=0.3)   # start = point 1 roll
    assert max(rolls) > 5.0 and min(rolls) < -3.0        # sweeps the keyframes
    assert cams[-1].roll < 2.0                           # loops back toward point 1
    assert any(abs(c.rot_x) > 1.0 for c in cams)
    assert any(abs(c.rot_y) > 1.0 for c in cams)


def test_v2fly_3d_tilt_perspective():
    from lazystretch.animate.flyv2 import V2Fly, V2Cam

    rgb, _ = _scene()
    eng = V2Fly(rgb, star_count=0)
    flat = eng._warp_bg(V2Cam(zoom=1.2))
    tx = eng._warp_bg(V2Cam(zoom=1.2, rot_x=4.0))        # tilt about X
    ty = eng._warp_bg(V2Cam(zoom=1.2, rot_y=-4.0))       # tilt about Y
    assert flat.shape == (eng.out_h, eng.out_w, 3)
    assert np.isfinite(tx).all() and np.isfinite(ty).all()
    # perspective tilt changes the frame (keystone), and the two axes differ
    assert float(np.mean(np.abs(flat - tx))) > 1e-3
    assert float(np.mean(np.abs(flat - ty))) > 1e-3
    assert not np.array_equal(tx, ty)


def test_v2fly_star_field_wraps_seamlessly():
    from lazystretch.animate.flyv2 import V2Fly, V2Cam

    rgb, _ = _scene()
    eng = V2Fly(rgb, star_count=300)
    # the wrapping depth means c and c+1 (one full cycle) render identically →
    # the star inflow loops with no jump
    a = eng._render_stars(V2Cam(c=0.3))
    b = eng._render_stars(V2Cam(c=1.3))
    assert np.allclose(a, b, atol=1e-4)                   # one full cycle → identical


def test_v2fly_zoom_clamped_to_border():
    from lazystretch.animate.flyv2 import V2Fly, V2Cam

    rgb, _ = _scene(h=80, w=160)                          # landscape source
    eng = V2Fly(rgb, out_w=160, out_h=80, star_count=0)   # same aspect → tight fill
    # zooming out past the fill point is clamped: all sub-fill zooms render the
    # same frame (can't reveal a border), while a zoomed-in frame differs
    a = eng.render_frame(V2Cam(zoom=0.3))
    b = eng.render_frame(V2Cam(zoom=0.9))
    c = eng.render_frame(V2Cam(zoom=1.6))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_v2fly_output_frame_and_pan_points():
    from lazystretch.animate.flyv2 import V2Fly, V2Cam, fly_v2

    rgb, _ = _scene(h=80, w=160)                          # landscape source
    # portrait output frame, independent of the image aspect
    eng = V2Fly(rgb, out_w=90, out_h=160, star_count=100, star_min=1.2, star_max=5.0)
    fr = eng.render_frame(V2Cam(zoom=1.0))
    assert fr.shape == (160, 90, 3)                       # portrait, cover-fit
    assert np.isfinite(fr).all()
    # multi-point pan: smooth path (no abrupt jumps between frames)
    cams = fly_v2(48, pan_points=[(-0.5, -0.3), (0.4, 0.1), (0.0, 0.45)])
    steps = np.array([[c.pan_x, c.pan_y] for c in cams])
    hops = np.linalg.norm(np.diff(steps, axis=0), axis=1)
    assert float(hops.max()) < 0.25                      # smooth, no abrupt corners


def test_v2fly_points_centre_exactly():
    from lazystretch.animate.flyv2 import _pan_path

    pts = [[-0.5, -0.25, 1.1, 5.0, 0.0, 0.0], [0.45, 0.1, 1.6, -3.0, 2.0, -1.0],
           [0.0, 0.4, 2.0, 0.0, -2.0, 1.0]]
    for n in (96, 143):                                  # incl. n not divisible by k
        pan, chan = _pan_path(pts, n)
        for seg, p in enumerate(pts):
            j = int(round(seg * n / len(pts)))           # the frame that lands on it
            assert np.allclose(pan[j], p[:2], atol=1e-6)      # centred exactly
            assert np.allclose(chan[j], p[2:], atol=1e-6)     # its zoom/roll/tilt too


def test_v2fly_streaks_and_size_change_output():
    from lazystretch.animate.flyv2 import V2Fly, V2Cam

    rgb, _ = _scene()
    cam = V2Cam(zoom=1.3, c=1.0)
    dots = V2Fly(rgb, star_count=300, streaks=False).render_frame(cam)
    trails = V2Fly(rgb, star_count=300, streaks=True, streak_len=60).render_frame(cam)
    assert float(np.mean(np.abs(dots - trails))) > 1e-4  # streaks change the render
    small = V2Fly(rgb, star_count=300, star_max=2.0).render_frame(cam)
    big = V2Fly(rgb, star_count=300, star_max=6.0).render_frame(cam)
    assert float(np.mean(np.abs(small - big))) > 1e-4    # size range matters


def test_color_grade_black_stays_black():
    from lazystretch.animate.volume3d import color_grade

    rgb, _ = _scene()
    rgb[:] = 0.0                                           # empty space
    faithful = color_grade(rgb, saturation=1.4, stylize=0.0)
    stylized = color_grade(rgb, saturation=1.4, stylize=0.8)
    assert float(faithful.max()) < 1e-3                   # black in → black out
    assert float(stylized.max()) < 1e-3                   # palette must not tint space


def test_spacefly_parallel_render_matches():
    from lazystretch.animate.volume3d import SpaceFly, fly_volume
    from lazystretch.animate.parallel import parallel_frames

    rgb, _ = _scene()
    eng = SpaceFly(rgb, star_count=120, haze_slabs=3)
    cams = fly_volume(4)
    seq = list(parallel_frames(eng, cams, workers=1))
    par = list(parallel_frames(eng, cams, workers=2))     # SpaceFly must pickle
    assert len(seq) == len(par) == 4
    for a, b in zip(seq, par):
        assert np.array_equal(a, b)


def test_parallel_frames_match_sequential():
    from lazystretch.animate.parallel import parallel_frames, auto_workers

    rgb, _ = _scene()
    eng = Flythrough3D(rgb, max_stars=60)
    cams = flythrough(4, zoom_end=1.2)
    seq = list(parallel_frames(eng, cams, workers=1))
    par = list(parallel_frames(eng, cams, workers=2))     # spawns processes
    assert len(seq) == len(par) == 4
    assert all(f.dtype == np.uint8 for f in par)
    for a, b in zip(seq, par):
        assert np.array_equal(a, b)                        # deterministic → identical
    assert auto_workers() >= 1


def test_pullback_reveals_out():
    cams = pullback(24, zoom_end=1.4)
    assert build_cameras("pullback", 10)                  # registered in PATHS
    assert cams[0].zoom == pytest.approx(1.4, abs=1e-6)   # starts pushed in
    assert cams[-1].zoom == pytest.approx(1.0, abs=1e-6)  # ends at the full field


def test_build_cameras_rejects_unknown_path():
    with pytest.raises(ValueError):
        build_cameras("warp-drive", 10)
