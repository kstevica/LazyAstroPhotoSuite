"""GUI smoke tests — construct the window offscreen; verify wiring (no display needed)."""
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_mainwindow_constructs_and_class_coupling(qapp):
    from lazystretch.gui.main_window import MainWindow

    w = MainWindow()
    # galaxy profile enables HDR + star reduction
    w.class_combo.setCurrentText("galaxy")
    assert w.checks["doHDR"].isChecked()
    assert w.checks["doStarReduce"].isChecked()
    # open profile disables HDR + SCNR
    w.class_combo.setCurrentText("open")
    assert not w.checks["doHDR"].isChecked()
    assert not w.checks["doSCNR"].isChecked()


def test_collect_params_reflects_controls(qapp):
    from lazystretch.gui.main_window import MainWindow

    w = MainWindow()
    w.class_combo.setCurrentText("emission")
    w.dials["satAdj"].set_value(0.3)
    w.crop_slider.set_value(6.0)
    w.checks["reduceCast"].setChecked(True)
    p = w._collect_params()
    assert p.object_class == "emission"
    assert abs(p.satAdj - 0.3) < 1e-6
    assert abs(p.cropPercent - 6.0) < 1e-6
    assert p.reduceCast is True


def test_ndarray_to_qimage(qapp):
    from lazystretch.gui.preview import ndarray_to_qimage

    rgb = np.linspace(0, 1, 30 * 40 * 3).reshape(30, 40, 3)
    img = ndarray_to_qimage(rgb)
    assert img.width() == 40 and img.height() == 30
    mono = np.zeros((12, 16))
    assert ndarray_to_qimage(mono).width() == 16


def test_float_slider_maps_range(qapp):
    from lazystretch.gui.widgets import FloatSlider

    fs = FloatSlider("x", -0.5, 0.5, 0.0)
    assert abs(fs.value()) < 1e-6
    fs.set_value(0.25)
    assert abs(fs.value() - 0.25) < 1e-3


def test_gui_recipe_controls_roundtrip(qapp, tmp_path):
    """Set controls -> save recipe -> load into a fresh window -> controls match."""
    from lazystretch.gui.main_window import MainWindow
    from lazystretch.io.recipes import apply_recipe, load_recipe, save_recipe
    from lazystretch.objects.model import Parameters

    w = MainWindow()
    w.class_combo.setCurrentText("emission")
    w.dials["satAdj"].set_value(0.25)
    w.dials["deepen"].set_value(0.40)
    w.checks["reduceCast"].setChecked(True)
    w.crop_slider.set_value(6.0)
    path = tmp_path / "gui.lsrecipe"
    save_recipe(path, w._collect_params(), w.data)

    w2 = MainWindow()
    recipe = load_recipe(path)
    q = Parameters.for_object(recipe.get("objectClass", "generic"), data=w2.data)
    apply_recipe(q, recipe, w2.data)
    w2._load_params_into_controls(q)
    assert w2.class_combo.currentText() == "emission"
    assert abs(w2.dials["satAdj"].value() - 0.25) < 1e-2
    assert abs(w2.dials["deepen"].value() - 0.40) < 1e-2
    assert w2.checks["reduceCast"].isChecked()
    assert abs(w2.crop_slider.value() - 6.0) < 1e-2
