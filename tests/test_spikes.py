"""Star detection + diffraction-spike rendering (lazystretch.develop.spikes)."""
import numpy as np

from lazystretch.develop.spikes import detect_stars, render_spikes


def _sky_with_stars(h=120, w=160, centers=((0.3, 0.4), (0.7, 0.6), (0.5, 0.2))):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    img = np.full((h, w, 3), 0.03, np.float64)
    for (fx, fy) in centers:
        cx, cy = fx * w, fy * h
        img += (0.9 * np.exp(-(((xx - cx) / 1.6) ** 2 + ((yy - cy) / 1.6) ** 2)))[..., None]
    return np.clip(img, 0, 1), centers


def test_detect_stars_finds_the_blobs():
    img, centers = _sky_with_stars()
    stars = detect_stars(img, max_stars=10)
    assert len(stars) >= len(centers)
    # every planted centre has a detected star near it (within ~3% of frame)
    for (fx, fy) in centers:
        assert any(abs(s["x"] - fx) < 0.03 and abs(s["y"] - fy) < 0.03 for s in stars)
    # records are normalised and carry flux + colour
    s = stars[0]
    assert 0 <= s["x"] <= 1 and 0 <= s["y"] <= 1
    assert 0.0 <= s["flux"] <= 1.0 and len(s["col"]) == 3


def test_render_spikes_adds_light_along_the_arms():
    img, _ = _sky_with_stars()
    star = {"x": 0.5, "y": 0.5, "len": 0.15, "flux": 1.0, "col": [1, 1, 1]}
    out = render_spikes(img, [star], count=4, angle_deg=0.0, intensity=1.0)
    assert out.shape == img.shape and np.isfinite(out).all()
    assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6
    assert np.mean(np.abs(out - img)) > 1e-4                # something was drawn
    # a horizontal arm (angle 0) brightens pixels to the right of centre, on the mid row
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    right = out[cy, cx + 20].mean() - img[cy, cx + 20].mean()
    assert right > 0.02


def test_render_spikes_length_is_per_star():
    img, _ = _sky_with_stars(centers=())
    short = render_spikes(img, [{"x": 0.5, "y": 0.5, "len": 0.05, "flux": 1.0}], count=4)
    long = render_spikes(img, [{"x": 0.5, "y": 0.5, "len": 0.25, "flux": 1.0}], count=4)
    # the longer star reaches farther: brightened pixels extend further from centre
    def reach(o):
        h, w = o.shape[:2]
        cy = h // 2
        lit = np.where(o[cy, w // 2:].mean(axis=1) > img[cy, w // 2:].mean(axis=1) + 0.01)[0]
        return int(lit.max()) if lit.size else 0
    assert reach(long) > reach(short) + 5


def test_render_spikes_count_clamped_and_noop():
    img, _ = _sky_with_stars(centers=())
    # empty star list → unchanged
    assert np.array_equal(render_spikes(img, [], count=8), np.clip(img, 0, 1))
    # count is clamped into [3, 32] without error
    star = {"x": 0.5, "y": 0.5, "len": 0.1, "flux": 1.0}
    for c in (1, 3, 32, 100):
        out = render_spikes(img, [star], count=c)
        assert np.isfinite(out).all() and out.max() <= 1 + 1e-6


def test_fringe_adds_chromatic_colour():
    img, _ = _sky_with_stars(centers=())
    star = {"x": 0.5, "y": 0.5, "len": 0.2, "flux": 1.0, "col": [1, 1, 1]}   # white star
    plain = render_spikes(img, [star], count=4, fringe=0.0)
    fringed = render_spikes(img, [star], count=4, fringe=1.0)
    assert not np.allclose(plain, fringed)
    # on the outer part of the right arm, the fringe introduces channel spread (colour)
    h, w = img.shape[:2]
    cy = int(round(0.5 * (h - 1)))
    x = w // 2 + int(0.16 * np.hypot(h, w))
    chroma = lambda o: float(o[cy, x].max() - o[cy, x].min())
    assert chroma(fringed) > chroma(plain) + 0.02


def test_arm_length_jitter_is_optional_and_deterministic():
    img, _ = _sky_with_stars(centers=())
    star = {"x": 0.5, "y": 0.5, "len": 0.2, "flux": 1.0, "col": [1, 1, 1]}
    even = render_spikes(img, [star], count=6, jitter=0.0)
    jit = render_spikes(img, [star], count=6, jitter=0.8)
    assert not np.allclose(even, jit)                    # arms now differ in length
    # deterministic: the same star jitters identically every render (stable preview↔apply)
    assert np.array_equal(jit, render_spikes(img, [star], count=6, jitter=0.8))


def test_render_spikes_does_not_mutate_input():
    img, _ = _sky_with_stars()
    before = img.copy()
    render_spikes(img, [{"x": 0.5, "y": 0.5, "len": 0.1, "flux": 1.0}], count=6)
    assert np.array_equal(img, before)
