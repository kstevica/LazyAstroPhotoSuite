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
    keys = [t[0] for t in shell._TOOLS]
    assert "develop" in keys
    assert shell._TITLES["develop"] == "LazyDevelop"


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


def test_undo_redo_and_revert(qapp):
    p = _loaded_panel(qapp)
    p._open_tool(dev_ops.get("levels")); p._apply_tool()
    p._open_tool(dev_ops.get("saturation")); p._apply_tool()
    n = len(p.doc.ops)
    assert n == 2
    p._undo(); assert len(p.doc.ops) == n - 1
    p._redo(); assert len(p.doc.ops) == n
    # click the base row → revert to 0
    p._history_clicked(p.history_list.item(0))
    assert len(p.doc.ops) == 0


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


def test_needs_color_tool_blocked_on_mono(qapp):
    from lazystretch.gui.develop_window import LazyDevelopPanel
    p = LazyDevelopPanel()
    mono = np.clip(np.linspace(0, 1, 40 * 40).reshape(40, 40), 0, 1).astype(np.float32)
    p.doc = DevelopDocument(mono)
    p._set_enabled_tools(True)
    p._open_tool(dev_ops.get("saturation"))            # needs_color → should not open
    assert p.tool_panel is None
