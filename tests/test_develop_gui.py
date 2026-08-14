"""GUI smoke tests for the LazyDevelop panel — construct offscreen, verify wiring."""
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from lazystretch.develop import DevelopDocument, ops as dev_ops  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _demo(seed=0):
    img = np.full((50, 60, 3), 0.25, np.float32)
    img[:, :30, 0] = 0.55
    img[:, 30:, 2] = 0.55
    return img


def _loaded_panel(qapp):
    from lazystretch.gui.develop_window import LazyDevelopPanel
    p = LazyDevelopPanel()
    p.doc = DevelopDocument(_demo(), path="demo.tif")
    p._set_enabled_tools(True)
    p._refresh_canvas(fit=True)
    p._refresh_history()
    return p


def test_panel_builds_a_tree_leaf_for_every_op(qapp):
    p = _loaded_panel(qapp)
    assert len(p._tool_items) == len(dev_ops.REGISTRY)
    # every category with ops is a top-level node
    assert p.toolbox.topLevelItemCount() == len(dev_ops.by_category())
    assert p.canvas.current_array() is not None


def test_tree_click_opens_tool(qapp):
    from PySide6.QtCore import Qt
    p = _loaded_panel(qapp)
    # find the leaf for "curves" and click it
    leaf = next(i for i in p._tool_items if i.data(0, Qt.UserRole) == "curves")
    p._tree_item_clicked(leaf)
    assert p.tool_panel is not None and p.tool_panel.op.name == "curves"
    # a category (non-leaf) click is a no-op
    cat = p.toolbox.topLevelItem(0)
    p._tree_item_clicked(cat)
    assert p.tool_panel.op.name == "curves"


def test_shell_registers_develop_tool(qapp):
    from lazystretch.gui import shell
    keys = [t["key"] for t in shell._TOOLS]
    assert "develop" in keys
    assert shell._TITLES["develop"] == "LazyDevelop"


def test_launcher_builds_grouped_cards(qapp):
    from lazystretch.gui.shell import LauncherPage, ToolCard, _ASSETS, _TOOLS
    opened = []
    page = LauncherPage(on_open=lambda k: opened.append(k))
    cards = page.findChildren(ToolCard)
    assert {c._spec["key"] for c in cards} == {"stack", "stretch", "develop", "moonsun"}
    # the pipeline groups: stack is "master", stretch+develop "process", moonsun "solar"
    groups = {t["key"]: t["group"] for t in _TOOLS}
    assert groups["stack"] == "master"
    assert groups["stretch"] == groups["develop"] == "process"
    assert groups["moonsun"] == "solar"
    # bundled background + thumbnails exist
    assert (_ASSETS / "launcher_bg.jpg").exists()
    for c in cards:
        assert (_ASSETS / c._spec["thumb"]).exists()
    # clicking a card opens its tool
    next(c for c in cards if c._spec["key"] == "develop")._on_open("develop")
    assert opened == ["develop"]


def test_launcher_animations(qapp):
    from lazystretch.gui.shell import LauncherPage, ToolCard
    page = LauncherPage(on_open=lambda k: None)
    # background rotate/zoom loops and drives a bounded 0..1 phase
    assert page._bg_anim.loopCount() == -1
    page._set_phase(0.7)
    assert abs(page._phase - 0.7) < 1e-9
    # card hover zoom: stopping (mouse-out) FREEZES the zoom, it is not reset to 1.0
    c = page.findChildren(ToolCard)[0]
    c._zoom_anim.start()
    c._set_zoom(1.04)
    c._zoom_anim.stop()
    assert c._zoom == 1.04


def test_open_apply_light_tool_grows_history(qapp):
    p = _loaded_panel(qapp)
    p._open_tool(dev_ops.get("levels"))
    assert p.tool_panel is not None
    p.tool_panel._set("gamma", 1.3)
    assert p.tool_panel.params()["gamma"] == pytest.approx(1.3)
    p._apply_tool()
    assert len(p.doc.ops) == 1
    assert p.history_list.count() == 2               # base + 1 op
    assert p.tool_panel is None                        # tool closed after apply


def test_mask_creation_and_gated_apply(qapp):
    p = _loaded_panel(qapp)
    p._make_lum_mask("lights")
    p._make_highlights_mask()
    assert p.mask_list.count() == 2
    assert set(p.doc.mask_names()) == {"Lum lights", "Highlights"}

    p._open_tool(dev_ops.get("saturation"))
    p.tool_panel.mask_combo.setCurrentIndex(1)         # first named mask
    p.tool_panel._set("amount", 0.4)
    gate = p.tool_panel.gate()
    assert gate["mask"] == "Lum lights"
    p._apply_tool()
    assert "Lum lights" in p.doc.ops[-1].title()


def test_history_navigation_is_nondestructive(qapp):
    p = _loaded_panel(qapp)
    p._open_tool(dev_ops.get("levels")); p._apply_tool()
    p._open_tool(dev_ops.get("saturation")); p._apply_tool()
    assert len(p.doc.ops) == 2 and p.doc.cursor == 2
    # undo/redo move the cursor but keep the steps
    p._undo(); assert p.doc.cursor == 1 and len(p.doc.ops) == 2
    p._redo(); assert p.doc.cursor == 2
    # clicking the base row navigates to 0 without discarding the steps
    p._history_clicked(p.history_list.item(0))
    assert p.doc.cursor == 0 and len(p.doc.ops) == 2
    # the history list still shows all steps (base + 2)
    assert p.history_list.count() == 3


def test_click_history_step_opens_editor_with_stored_values(qapp):
    from PySide6.QtCore import Qt
    p = _loaded_panel(qapp)
    p._open_tool(dev_ops.get("levels"))
    p.tool_panel._set("gamma", 1.4)
    p._apply_tool()
    p._open_tool(dev_ops.get("saturation")); p._apply_tool()
    # click the first op's row (row 1) → editor opens with its values + tree selection
    p._history_clicked(p.history_list.item(1))
    assert p.tool_panel is not None
    assert p.tool_panel.edit_index == 0
    assert p.tool_panel.op.name == "levels"
    assert p.tool_panel.params()["gamma"] == pytest.approx(1.4)
    cur = p.toolbox.currentItem()
    assert cur is not None and cur.data(0, Qt.UserRole) == "levels"


def test_edit_last_step_updates_in_place(qapp):
    p = _loaded_panel(qapp)
    p._open_tool(dev_ops.get("levels")); p._apply_tool()
    p._open_tool(dev_ops.get("saturation")); p.tool_panel._set("amount", 0.3); p._apply_tool()
    assert len(p.doc.ops) == 2
    p._history_clicked(p.history_list.item(2))    # edit the last step (saturation)
    assert p.tool_panel.edit_index == 1
    p.tool_panel._set("amount", 0.6)
    p._apply_tool()                                # "Update step" (light, inline)
    assert len(p.doc.ops) == 2                     # edited in place, not appended
    assert p.doc.ops[1].params["amount"] == pytest.approx(0.6)


def test_delete_step_from_editor_and_button(qapp):
    p = _loaded_panel(qapp)
    p._open_tool(dev_ops.get("levels")); p._apply_tool()
    p._open_tool(dev_ops.get("saturation")); p._apply_tool()
    p._open_tool(dev_ops.get("scnr")); p._apply_tool()
    assert len(p.doc.ops) == 3
    # delete the middle step via the editor
    p._history_clicked(p.history_list.item(2))     # saturation (index 1)
    p._delete_current_step()
    assert [o.name for o in p.doc.ops] == ["levels", "scnr"]
    # delete via the History "Delete step" button (select row 1 = levels)
    p.history_list.setCurrentRow(1)
    p._delete_selected_step()
    assert [o.name for o in p.doc.ops] == ["scnr"]


def test_geometry_tool_has_no_gate(qapp):
    from lazystretch.gui.develop_window import ToolPanel
    panel = ToolPanel(dev_ops.get("crop"), [])
    assert panel.wants_rect()
    assert panel.gate() == {"mask": None, "mask_invert": False, "opacity": 1.0}
    color_panel = ToolPanel(dev_ops.get("saturation"), ["m1"])
    assert color_panel._gateable


def test_curve_editor_roundtrip(qapp):
    from lazystretch.gui.develop_window import CurveEditor
    ce = CurveEditor()
    ce.set_points([[0, 0], [0.3, 0.2], [1, 1]])
    assert ce.points() == [[0.0, 0.0], [0.3, 0.2], [1.0, 1.0]]


def test_canvas_rect_mode(qapp):
    from lazystretch.gui.develop_window import DevelopCanvas
    c = DevelopCanvas()
    c.set_image(_demo(), keep_view=False)
    c.set_rect_mode(True)
    c.show_rect({"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.9})   # must not raise
    c.set_rect_mode(False)


def test_crop_rect_shows_live_pixel_size(qapp):
    from lazystretch.gui.develop_window import DevelopCanvas
    c = DevelopCanvas()
    img = np.zeros((100, 200, 3), np.float32)          # h=100, w=200
    c.set_image(img, keep_view=False)
    c.set_rect_mode(True)
    c.show_rect({"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 1.0})
    assert c._rect_label is not None
    assert c._rect_label.text() == "100 × 100 px"       # 200*0.5 × 100*1.0
    c.show_rect({"x0": 0.25, "y0": 0.1, "x1": 0.75, "y1": 0.6})
    assert c._rect_label.text() == "100 × 50 px"        # 200*0.5 × 100*0.5
    c.set_rect_mode(False)                               # label removed with the rect
    assert c._rect_label is None


def test_heavy_tool_previews_on_downscaled_proxy(qapp):
    from lazystretch.gui.develop_window import LazyDevelopPanel, PROXY_MAX_DIM
    big = np.clip(0.2 + 0.05 * np.random.default_rng(0).standard_normal((1500, 2400, 3)),
                  0, 1).astype(np.float32)
    p = LazyDevelopPanel()
    p.doc = DevelopDocument(big)
    p._set_enabled_tools(True)
    p._refresh_canvas(fit=True)
    p._open_tool(dev_ops.get("wavelet_sharpen"))     # heavy tool
    p.tool_panel._set("sharpen3", 60)
    p._do_live_preview()                              # bypass the debounce timer
    shown = p.canvas.current_array()
    assert shown is not None
    assert max(shown.shape[:2]) <= PROXY_MAX_DIM      # displayed a downscaled proxy
    assert p._preview_kind == "proxy"
    # a light tool previews at full resolution
    p._open_tool(dev_ops.get("levels"))
    p.tool_panel._set("gamma", 1.2)
    p._do_live_preview()
    assert p._preview_kind == "full"
    assert p.canvas.current_array().shape[:2] == (1500, 2400)


def test_apply_worker_decision_accounts_for_downstream_heavy(qapp):
    # regression: inserting a LIGHT op ahead of a kept HEAVY op must use the worker,
    # not recompute the heavy downstream op synchronously on the UI thread.
    p = _loaded_panel(qapp)
    p.doc.apply_op("levels", {})          # light
    p.doc.apply_op("hdr", {})             # heavy (applied directly on the doc)
    light = dev_ops.get("levels")
    assert not light.heavy
    p.doc.goto(1)                          # cursor between levels and hdr
    assert p._apply_is_heavy(light, None) is True     # insert here → hdr recomputed → worker
    assert p._apply_is_heavy(light, 0) is True        # edit levels → hdr downstream → worker
    assert p._apply_is_heavy(dev_ops.get("hdr"), 1) is True   # hdr itself heavy
    p.doc.goto(2)                          # tip, nothing downstream
    assert p._apply_is_heavy(light, None) is False


def test_history_controls_guarded_while_busy(qapp, monkeypatch):
    # regression: while the worker mutates the doc, undo/redo/history/delete must no-op
    # (otherwise a cross-thread cache race crashes or corrupts the history).
    p = _loaded_panel(qapp)
    p._open_tool(dev_ops.get("levels")); p._apply_tool()
    p._open_tool(dev_ops.get("saturation")); p._apply_tool()
    n, cur = len(p.doc.ops), p.doc.cursor
    monkeypatch.setattr(p, "_busy", lambda: True)
    p._undo(); assert p.doc.cursor == cur
    p._redo(); assert p.doc.cursor == cur
    p._history_clicked(p.history_list.item(0)); assert p.doc.cursor == cur
    p.history_list.setCurrentRow(1); p._delete_selected_step()
    assert len(p.doc.ops) == n


def test_edit_crop_shows_input_not_cropped_output(qapp):
    # regression: editing a Crop step must display the crop's INPUT so the rubber-band
    # rectangle maps to the right pixels (not the already-cropped, smaller preview).
    p = _loaded_panel(qapp)                 # 50 x 60 image
    p._open_tool(dev_ops.get("crop"))
    p.tool_panel.set_rect({"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 1.0})
    p._apply_tool()
    assert p.doc.result().shape[1] == 30    # cropped to the left half
    p._history_clicked(p.history_list.item(1))   # edit the crop step
    assert p.tool_panel.edit_index == 0
    assert p.canvas.current_array().shape[1] == 60   # canvas shows the full input


def test_preview_clears_show_original_without_reentry(qapp):
    p = _loaded_panel(qapp)
    p._open_tool(dev_ops.get("levels"))
    p.before_btn.setChecked(True)
    p.tool_panel._set("gamma", 1.3)
    p._do_live_preview()
    assert not p.before_btn.isChecked()
    assert p._preview_kind == "full"


def test_recipe_file_roundtrip_through_panel(qapp, tmp_path, monkeypatch):
    import json
    from PySide6.QtWidgets import QFileDialog
    p = _loaded_panel(qapp)
    p.save_recipe_btn.setEnabled(True); p.load_recipe_btn.setEnabled(True)
    p._open_tool(dev_ops.get("levels")); p._apply_tool()
    p._open_tool(dev_ops.get("saturation")); p._apply_tool()
    result = p.doc.result().copy()

    recipe_path = tmp_path / "grade.ldrecipe"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(recipe_path), ""))
    p._save_recipe()
    assert recipe_path.exists()
    assert len(json.loads(recipe_path.read_text())) == 2

    # replay the recipe on a fresh document
    from lazystretch.develop import DevelopDocument
    p.doc = DevelopDocument(_demo())
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(recipe_path), ""))
    p._load_recipe()
    assert len(p.doc.ops) == 2
    assert np.allclose(p.doc.result(), result, atol=1e-6)


def test_auto_panel_flow(qapp):
    from lazystretch.gui.develop_window import LazyDevelopPanel
    from lazystretch.develop import DevelopDocument
    rng = np.random.default_rng(0)
    h, w = 140, 180
    grad = np.broadcast_to(np.linspace(0, 0.14, w), (h, w))
    img = np.stack([0.11 + grad, 0.16 + grad, 0.09 + grad], -1).astype(np.float32)
    img = np.clip(img + 0.02 * rng.standard_normal((h, w, 3)).astype(np.float32), 0, 1)
    p = LazyDevelopPanel()
    p.doc = DevelopDocument(img)
    p._set_enabled_tools(True); p._refresh_canvas(fit=True)

    from lazystretch.develop.auto import auto_develop_plan
    plan = auto_develop_plan(p.doc.result())
    steps = plan["steps"]
    assert len(steps) >= 2
    p._show_auto(plan)
    assert p.auto_panel is not None
    assert len(p.auto_panel.selected_steps()) == len(steps)
    # the recipe's semantic-mask gates were installed into the library
    assert set(plan["masks"]) <= set(p.doc.mask_names())
    gated = [s for s in steps if s.get("mask")]
    assert gated and all(s["mask"] in p.doc.masks for s in gated)   # gates resolve by name
    p.auto_panel._checks[0][0].setChecked(False)               # untick one
    assert len(p.auto_panel.selected_steps()) == len(steps) - 1
    p._auto_preview()                                           # previews on the proxy
    assert p._preview_kind == "auto"

    sel = list(p.auto_panel.selected_steps())
    for s in sel:                                              # simulate the worker commit
        p.doc.apply_op(s["name"], s["params"],
                       mask=s.get("mask"), mask_invert=s.get("mask_invert", False))
    p._finish_auto(sel)
    assert p.auto_panel is None
    assert [o.name for o in p.doc.ops] == [s["name"] for s in sel]
    # a gated auto step actually carries its semantic mask
    if gated:
        applied = next(o for o in p.doc.ops if o.name == gated[0]["name"])
        assert applied.mask == gated[0]["mask"]


def test_mask_combine_and_rename_flow(qapp, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    p = _loaded_panel(qapp)
    p._make_lum_mask("lights")
    p._make_highlights_mask()
    assert set(p.doc.mask_names()) == {"Lum lights", "Highlights"}
    # intersect the two → a new composite mask appears
    p.combine_a.setCurrentText("Lum lights")
    p.combine_op.setCurrentIndex(0)                    # ∩ intersect
    p.combine_b.setCurrentText("Highlights")
    p._combine_masks()
    comp = [n for n in p.doc.mask_names() if "∩" in n]
    assert comp and np.allclose(
        p.doc.masks[comp[0]],
        np.minimum(p.doc.masks["Lum lights"], p.doc.masks["Highlights"]))
    # invert disables operand B and still works
    p.combine_op.setCurrentIndex(3)                    # ¬ invert A
    assert not p.combine_b.isEnabled()
    p.combine_a.setCurrentText("Highlights")
    p._combine_masks()
    assert any(n.startswith("¬") for n in p.doc.mask_names())
    # rename the first mask via the panel (dialog stubbed)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Renamed", True))
    first = p.mask_list.item(0).text()
    p._rename_mask(p.mask_list.item(0))
    assert "Renamed" in p.doc.mask_names() and first not in p.doc.mask_names()


def test_auto_masks_populate_the_library(qapp):
    from lazystretch.develop.semantic import segment
    p = _loaded_panel(qapp)
    masks = segment(p.doc.result())
    p._add_auto_masks(masks)
    assert p.mask_list.count() == len(masks)
    assert {"Sky", "Stars", "Nebulosity"} <= set(p.doc.mask_names())
    assert p.combine_a.count() == len(masks)              # combos see them too
    p.doc.apply_op("saturation", {"amount": 0.3}, mask="Nebulosity")
    assert p.doc.ops[-1].mask == "Nebulosity"
    p._add_auto_masks(segment(p.doc.result()))            # re-run overwrites, no dupes
    names = p.doc.mask_names()
    assert len(names) == len(set(names))


def test_auto_masks_do_not_clobber_user_masks(qapp):
    from lazystretch.develop.semantic import segment
    p = _loaded_panel(qapp)
    p.doc.add_mask("Nebulosity", np.ones(p.doc.base.shape[:2], np.float32))   # a user mask
    p._add_auto_masks(segment(p.doc.result()))
    assert np.allclose(p.doc.masks["Nebulosity"], 1.0)         # user mask untouched
    assert "Nebulosity 2" in p.doc.mask_names()                # auto one bumped
    # re-running is idempotent (clears its own previous set, no growth)
    n = len(p.doc.mask_names())
    p._add_auto_masks(segment(p.doc.result()))
    assert len(p.doc.mask_names()) == n


def test_auto_masks_keep_masks_used_by_committed_steps(qapp):
    # running Auto masks after Auto-develop must not strand a committed step's gate
    from lazystretch.develop.auto import auto_develop_plan
    from lazystretch.develop.semantic import segment
    rng = np.random.default_rng(0); h, w = 120, 150
    grad = np.broadcast_to(np.linspace(0, 0.14, w), (h, w))
    img = np.clip(np.stack([0.11 + grad, 0.16 + grad, 0.09 + grad], -1).astype(np.float32)
                  + 0.02 * rng.standard_normal((h, w, 3)).astype(np.float32), 0, 1)
    from lazystretch.gui.develop_window import LazyDevelopPanel
    from lazystretch.develop import DevelopDocument
    p = LazyDevelopPanel(); p.doc = DevelopDocument(img)
    p._set_enabled_tools(True); p._refresh_canvas(fit=True)

    plan = auto_develop_plan(p.doc.result())
    p._install_auto_masks(plan["masks"], plan["steps"])
    for s in plan["steps"]:
        p.doc.apply_op(s["name"], s["params"], mask=s.get("mask"),
                       mask_invert=s.get("mask_invert", False))
    used = {s["mask"] for s in plan["steps"] if s.get("mask")}
    assert used                                         # composites like "Star cores"
    p._add_auto_masks(segment(p.doc.result()))          # now click Auto masks
    assert used <= set(p.doc.mask_names())              # in-use masks survived
    assert all(op.mask in p.doc.masks for op in p.doc.ops if op.mask)   # every gate resolves


def test_busy_disables_view_and_mask_controls(qapp):
    p = _loaded_panel(qapp)
    p._set_busy(True)
    assert not p.before_btn.isEnabled() and not p.fit_btn.isEnabled()
    assert not p.mask_group.isEnabled() and not p.auto_masks_btn.isEnabled()
    p._set_busy(False)
    assert p.before_btn.isEnabled() and p.fit_btn.isEnabled()


def test_show_auto_empty_is_noop(qapp):
    p = _loaded_panel(qapp)
    p._show_auto({"steps": [], "masks": {}})
    assert p.auto_panel is None


def test_save_jpeg_prompts_quality(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QInputDialog
    p = _loaded_panel(qapp)
    out = tmp_path / "out.jpg"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: (70, True))
    p._save()
    assert out.exists()


def test_needs_color_tool_blocked_on_mono(qapp):
    from lazystretch.gui.develop_window import LazyDevelopPanel
    p = LazyDevelopPanel()
    mono = np.clip(np.linspace(0, 1, 40 * 40).reshape(40, 40), 0, 1).astype(np.float32)
    p.doc = DevelopDocument(mono)
    p._set_enabled_tools(True)
    p._open_tool(dev_ops.get("saturation"))            # needs_color → should not open
    assert p.tool_panel is None
