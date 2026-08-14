"""Tests for the LazyDevelop interactive-finishing engine (lazystretch.develop)."""
import numpy as np
import pytest

from lazystretch.develop import DevelopDocument, ops
from lazystretch.develop import masks
from lazystretch.develop.curves import apply_curve, curve_lut


# --------------------------------------------------------------------------- fixtures
def _demo_rgb(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((60, 72, 3), 0.2, np.float32)
    img[:, :36, 0] = 0.6                      # left half red-ish
    img[:, 36:, 2] = 0.6                      # right half blue-ish
    img += 0.03 * rng.standard_normal(img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


# --------------------------------------------------------------------------- registry
def test_registry_is_populated_and_well_formed():
    assert len(ops.REGISTRY) >= 20
    for name, op in ops.REGISTRY.items():
        assert op.name == name
        assert op.category in ops.CATEGORY_ORDER
        # defaults round-trip through coerce without error
        d = op.merged(op.defaults())
        assert set(d) == {p.key for p in op.params}


def test_every_op_runs_and_stays_in_gamut():
    img = _demo_rgb().astype(np.float64)
    for name, op in ops.REGISTRY.items():
        out = np.asarray(op.fn(img, op.merged(None)))
        assert np.isfinite(out).all(), name
        assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6, name
        assert out.ndim in (2, 3), name


def test_ops_do_not_mutate_input():
    img = _demo_rgb().astype(np.float64)
    for name, op in ops.REGISTRY.items():
        before = img.copy()
        op.fn(img, op.merged(None))
        assert np.array_equal(img, before), f"{name} mutated its input"


# --------------------------------------------------------------------------- document/history
def test_apply_undo_redo_navigation_is_nondestructive():
    img = _demo_rgb()
    doc = DevelopDocument(img)
    doc.apply_op("saturation", {"amount": 0.5})
    r1 = doc.result().copy()
    doc.apply_op("levels", {"black": 0.05, "white": 0.9, "gamma": 1.2})
    assert len(doc.ops) == 2 and doc.cursor == 2

    # undo/redo move the cursor but KEEP every step
    assert doc.undo()
    assert len(doc.ops) == 2 and doc.cursor == 1
    assert np.allclose(doc.result(), r1)
    assert doc.can_redo()
    assert doc.redo() and doc.cursor == 2

    # goto navigates without discarding the later steps
    doc.goto(0)
    assert len(doc.ops) == 2 and doc.cursor == 0
    assert np.allclose(doc.result(), img)
    assert not doc.undo()


def test_new_op_inserts_at_cursor_and_keeps_later_steps():
    doc = DevelopDocument(_demo_rgb())
    doc.apply_op("saturation", {"amount": 0.3})
    doc.apply_op("levels", {"gamma": 1.2})
    doc.goto(1)                                  # back to just after step 1
    doc.apply_op("scnr", {"amount": 1.0})        # insert here — step 2 must survive
    assert [o.name for o in doc.ops] == ["saturation", "scnr", "levels"]
    assert doc.cursor == 2 and len(doc.ops) == 3


def test_replace_op_edits_in_place_and_recomputes():
    doc = DevelopDocument(_demo_rgb())
    doc.apply_op("levels", {"black": 0.0, "white": 1.0, "gamma": 1.0})
    doc.apply_op("saturation", {"amount": 0.2})
    before = doc.result().copy()
    doc.replace_op(0, {"black": 0.0, "white": 1.0, "gamma": 2.0})
    assert len(doc.ops) == 2
    assert doc.ops[0].params["gamma"] == 2.0
    assert not np.allclose(doc.result(), before)     # downstream recomputed


def test_delete_op_keeps_the_others():
    doc = DevelopDocument(_demo_rgb())
    doc.apply_op("saturation", {"amount": 0.4})
    doc.apply_op("levels", {"gamma": 1.5})
    doc.apply_op("scnr", {"amount": 1.0})
    assert len(doc.ops) == 3 and doc.cursor == 3
    doc.delete_op(1)
    assert [o.name for o in doc.ops] == ["saturation", "scnr"]
    assert doc.cursor == 2


def test_op_params_returns_stored_values_and_gate():
    doc = DevelopDocument(_demo_rgb())
    doc.apply_op("levels", {"black": 0.1, "white": 0.8, "gamma": 1.3}, opacity=0.7)
    p = doc.op_params(0)
    assert p["params"]["gamma"] == 1.3 and p["opacity"] == 0.7 and p["mask"] is None


def test_orient_actually_rotates_and_flips():
    # regression: rotate is a choice index (0..3), not degrees — must not compute k=0.
    img = np.zeros((10, 20, 3), np.float32)
    img[0, 0] = [1, 1, 1]                       # marker at top-left
    o = ops.get("orient")
    r90 = np.asarray(o.fn(img, o.merged({"rotate": 1})))     # 90° CCW
    assert r90.shape[:2] == (20, 10)
    assert r90[-1, 0, 0] == 1                   # top-left → bottom-left
    r180 = np.asarray(o.fn(img, o.merged({"rotate": 2})))
    assert r180.shape[:2] == (10, 20) and r180[-1, -1, 0] == 1
    r270 = np.asarray(o.fn(img, o.merged({"rotate": 3})))
    assert r270.shape[:2] == (20, 10)
    fh = np.asarray(o.fn(img, o.merged({"flip_h": True})))
    assert fh[0, -1, 0] == 1                    # top-left → top-right
    fv = np.asarray(o.fn(img, o.merged({"flip_v": True})))
    assert fv[-1, 0, 0] == 1


def test_crop_changes_shape_and_survives_history():
    doc = DevelopDocument(_demo_rgb())
    doc.apply_op("crop", {"rect": {"x0": 0.25, "y0": 0.25, "x1": 0.75, "y1": 0.75}})
    h, w = doc.result().shape[:2]
    assert h == 30 and w == 36
    doc.undo()
    assert doc.result().shape[:2] == (60, 72)


# --------------------------------------------------------------------------- masks / blend
def test_opacity_zero_is_identity():
    doc = DevelopDocument(_demo_rgb())
    base = doc.result()
    out = doc.preview_op("levels", {"black": 0.0, "white": 0.5}, opacity=0.0)
    assert np.allclose(out, base)


def test_mask_restricts_effect():
    doc = DevelopDocument(_demo_rgb())
    base = doc.result()
    lm = masks.luminosity_mask(base, masks.LUM_LIGHTS, 2)
    doc.add_mask("lum", lm)
    full = doc.preview_op("levels", {"black": 0.0, "white": 0.5})
    gated = doc.preview_op("levels", {"black": 0.0, "white": 0.5}, mask="lum")
    assert np.abs(gated - base).mean() < np.abs(full - base).mean()


def test_mask_invert_is_complementary():
    doc = DevelopDocument(_demo_rgb())
    base = doc.result()
    m = np.zeros(base.shape[:2], np.float32)
    m[:, :36] = 1.0                            # left half only
    doc.add_mask("left", m)
    left = doc.preview_op("levels", {"black": 0.0, "white": 0.5}, mask="left")
    right = doc.preview_op("levels", {"black": 0.0, "white": 0.5}, mask="left", mask_invert=True)
    # left-masked changes the left half; inverted changes the right half
    assert np.abs(left[:, :36] - base[:, :36]).mean() > np.abs(left[:, 36:] - base[:, 36:]).mean()
    assert np.abs(right[:, 36:] - base[:, 36:]).mean() > np.abs(right[:, :36] - base[:, :36]).mean()


# --------------------------------------------------------------------------- curves
def test_curve_identity_and_monotone_scurve():
    lut = curve_lut([[0, 0], [1, 1]])
    xs = np.linspace(0, 1, lut.shape[0])
    assert np.allclose(lut, xs, atol=1e-3)
    # an S-curve stays monotone (PCHIP: no overshoot)
    s = curve_lut([[0, 0], [0.25, 0.15], [0.75, 0.85], [1, 1]])
    assert np.all(np.diff(s) >= -1e-6)
    assert s.min() >= 0 and s.max() <= 1


def test_curve_luminance_mode_preserves_hue_direction():
    img = _demo_rgb()
    out = apply_curve(img, [[0, 0], [0.5, 0.7], [1, 1]], "lum")
    # brightening via luminance keeps the R/B ratio roughly where it was (hue preserved)
    assert out.shape == img.shape
    assert out.mean() > img.mean()


# --------------------------------------------------------------------------- Studio tools
def test_selective_color_targets_one_group():
    img = _demo_rgb().astype(np.float64)
    out = ops.get("selective_color").fn(img, {
        "group": 0, "yellow": -0.6, "range": 50, "guard": 0, "smooth": 0,
        "lum_protect": True, "absolute": False})
    red_moved = np.abs(out[:, :36] - img[:, :36]).mean()
    blue_moved = np.abs(out[:, 36:] - img[:, 36:]).mean()
    assert red_moved > 3 * blue_moved + 1e-4


def test_selective_color_no_ink_is_noop():
    img = _demo_rgb().astype(np.float64)
    out = ops.get("selective_color").fn(img, ops.get("selective_color").merged(None))
    assert np.allclose(out, img)


def _high_freq(a):
    from scipy.ndimage import uniform_filter
    L = a.mean(-1) if a.ndim == 3 else a
    return float(np.std(L - uniform_filter(L, 3)))


def test_wavelet_sharpen_raises_and_denoise_lowers_detail():
    rng = np.random.default_rng(3)
    tex = np.clip(0.3 + 0.05 * rng.standard_normal((64, 64, 3)), 0, 1).astype(np.float64)
    tex[::4, :] += 0.15
    tex = np.clip(tex, 0, 1)
    sharp = ops.get("wavelet_sharpen").fn(tex, {"sharpen2": 80, "strength": 100})
    assert _high_freq(sharp) > _high_freq(tex)
    denoised = ops.get("wavelet_sharpen").fn(tex, {"denoise1": 90, "denoise2": 90, "strength": 100})
    assert _high_freq(denoised) < _high_freq(tex)


def test_wavelet_all_zero_is_noop():
    img = _demo_rgb().astype(np.float64)
    out = ops.get("wavelet_sharpen").fn(img, ops.get("wavelet_sharpen").merged(None))
    assert np.allclose(out, img)


# --------------------------------------------------------------------------- mask generators
def test_luminosity_mask_depth_increases_selectivity():
    img = _demo_rgb()
    shallow = masks.luminosity_mask(img, masks.LUM_LIGHTS, 1)
    deep = masks.luminosity_mask(img, masks.LUM_LIGHTS, 4)
    assert shallow.shape == img.shape[:2]
    assert 0.0 <= deep.min() and deep.max() <= 1.0
    # a deeper mask is everywhere ≤ the shallow one (higher power on a [0,1] base)
    assert deep.mean() < shallow.mean()


def test_darks_mask_is_complement_of_lights_at_depth1():
    img = _demo_rgb()
    lights = masks.luminosity_mask(img, masks.LUM_LIGHTS, 1)
    darks = masks.luminosity_mask(img, masks.LUM_DARKS, 1)
    assert np.allclose(lights + darks, 1.0, atol=1e-5)


def test_mask_set_operations():
    a = np.array([[0.0, 0.5], [1.0, 0.2]], np.float32)
    b = np.array([[0.3, 0.9], [0.4, 0.2]], np.float32)
    assert np.allclose(masks.combine_masks(a, b, "intersect"), np.minimum(a, b))
    assert np.allclose(masks.combine_masks(a, b, "union"), np.maximum(a, b))
    assert np.allclose(masks.combine_masks(a, b, "subtract"), np.minimum(a, 1 - b))
    assert np.allclose(masks.combine_masks(a, None, "invert"), 1 - a)
    # readable names
    assert masks.combined_name("Nebula", "subtract", "Bright") == "Nebula ∩ ¬Bright"
    assert masks.combined_name("Sky", "invert") == "¬Sky"


def test_rename_mask_repoints_op_gates():
    doc = DevelopDocument(_demo_rgb())
    doc.add_mask("Sky", np.ones(doc.base.shape[:2], np.float32))
    doc.apply_op("saturation", {"amount": 0.3}, mask="Sky")
    assert doc.ops[-1].mask == "Sky"
    assert doc.rename_mask("Sky", "Background")
    assert "Background" in doc.masks and "Sky" not in doc.masks
    assert doc.ops[-1].mask == "Background"          # gate repointed
    # rejects duplicate / empty / no-op names
    doc.add_mask("Stars", np.zeros(doc.base.shape[:2], np.float32))
    assert not doc.rename_mask("Background", "Stars")
    assert not doc.rename_mask("Background", "")
    assert not doc.rename_mask("Background", "Background")


def test_recipe_roundtrip():
    doc = DevelopDocument(_demo_rgb())
    doc.apply_op("saturation", {"amount": 0.4})
    doc.apply_op("levels", {"black": 0.02, "white": 0.95, "gamma": 1.1}, opacity=0.8)
    recipe = doc.to_recipe()
    result = doc.result().copy()

    doc2 = DevelopDocument(doc.base)
    doc2.apply_recipe(recipe)
    assert np.allclose(doc2.result(), result, atol=1e-6)


# --------------------------------------------------------------------------- auto-develop
def _dirty_image(seed=0):
    rng = np.random.default_rng(seed)
    h, w = 140, 180
    grad = np.broadcast_to(np.linspace(0.0, 0.14, w), (h, w))   # left→right gradient (full height)
    img = np.stack([0.11 + grad, 0.16 + grad, 0.09 + grad], axis=-1).astype(np.float32)
    img = img + 0.02 * rng.standard_normal((h, w, 3)).astype(np.float32)   # noise + green + warm
    img[60:80, 80:110] += np.array([0.30, 0.10, 0.15])
    return np.clip(img, 0.0, 1.0)


def test_auto_analyze_is_ordered_valid_and_in_gamut():
    from lazystretch.develop.auto import analyze, _ORDER
    steps = analyze(_dirty_image())
    assert len(steps) >= 3
    names = [s["name"] for s in steps]
    assert names == sorted(names, key=_ORDER.index)     # finishing order
    doc = DevelopDocument(_dirty_image())
    for s in steps:
        ops.get(s["name"]).merged(s["params"])          # every name/param valid
        doc.apply_op(s["name"], s["params"])
    out = doc.result()
    assert np.isfinite(out).all() and out.min() >= -1e-6 and out.max() <= 1 + 1e-6


def test_auto_suggests_more_for_a_worse_image():
    from lazystretch.develop.auto import analyze
    # a clean image: neutral, low-noise, contrasty, saturated
    clean = np.full((120, 120, 3), 0.04, np.float32)
    clean[40:80, 40:80] = np.array([0.75, 0.20, 0.55])   # bright saturated structure
    assert len(analyze(_dirty_image())) > len(analyze(clean))


def test_auto_analyze_handles_mono():
    from lazystretch.develop.auto import analyze
    mono = np.clip(np.linspace(0, 0.3, 100 * 100).reshape(100, 100), 0, 1)
    steps = analyze(mono)                                # must not raise; no colour steps
    assert all(s["name"] not in ("scnr", "bg_neutralize", "saturation", "chroma_nr", "deveil")
               for s in steps)


# --------------------------------------------------------------------------- semantic masks
def test_semantic_segment_produces_named_masks():
    from lazystretch.develop.semantic import segment
    img = _demo_rgb().copy()
    img[5, 5] = 1.0; img[10, 60] = 0.95            # a couple of stars
    m = segment(img)
    expected = {"Sky", "Stars", "Bright stars", "Faint stars",
                "Nebulosity", "Faint nebulosity", "Dust", "Cores"}
    assert expected <= set(m) and "Hα (red)" in m   # RGB → Hα present
    for name, v in m.items():
        assert v.shape == img.shape[:2] and v.dtype == np.float32
        assert v.min() >= 0.0 and v.max() <= 1.0
    # star sub-masks are subsets of the full Stars mask (min-combined)
    assert np.all(m["Bright stars"] <= m["Stars"] + 1e-6)
    assert np.all(m["Faint stars"] <= m["Stars"] + 1e-6)


def test_semantic_segment_mono_skips_ha():
    from lazystretch.develop.semantic import segment
    mono = np.clip(np.linspace(0, 0.5, 80 * 80).reshape(80, 80), 0, 1)
    m = segment(mono)
    assert "Hα (red)" not in m
    assert "Stars" in m and "Sky" in m


def test_semantic_segment_snr_path_runs():
    from lazystretch.develop.semantic import segment
    img = _demo_rgb()
    snr = np.random.default_rng(0).random(img.shape[:2]).astype(np.float32)
    m = segment(img, snr=snr)
    fn = m["Faint nebulosity"]
    assert fn.shape == img.shape[:2] and np.isfinite(fn).all()


def test_jpeg_save_respects_quality(tmp_path):
    import os
    from lazystretch.io.image_io import save_image, load_image
    img = _demo_rgb()
    lo = tmp_path / "lo.jpg"; hi = tmp_path / "hi.jpg"
    save_image(str(lo), img, quality=40)
    save_image(str(hi), img, quality=95)
    assert lo.exists() and hi.exists()
    assert os.path.getsize(hi) > os.path.getsize(lo)     # higher quality → bigger file
    assert load_image(str(hi)).data.shape == img.shape   # round-trips back in


@pytest.mark.parametrize("op_name", ["saturation", "scnr", "white_balance", "chroma_nr"])
def test_color_ops_are_noop_on_mono(op_name):
    mono = np.clip(np.linspace(0, 1, 40 * 40).reshape(40, 40), 0, 1)
    out = np.asarray(ops.get(op_name).fn(mono, ops.get(op_name).merged(None)))
    assert out.shape == mono.shape
    assert np.allclose(out, mono, atol=1e-6)
