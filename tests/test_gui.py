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


def test_run_log_export_to_text_and_save(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from lazystretch.gui.widgets import RunLogView

    log = RunLogView()
    log.start()
    log.append("Auto-stretch (STF -> HistogramTransformation)")
    log.append("   via NoiseXTerminator (linear)")
    log.finish()
    text = log.log_to_text()
    assert "Auto-stretch" in text and "NoiseXTerminator" in text

    path = tmp_path / "run.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    assert log.save_log() == str(path) and path.exists()
    body = path.read_text()
    assert "LazyStretch run log" in body and "Auto-stretch" in body

    # cancelling the dialog writes nothing and returns None
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    assert log.save_log() is None

    # empty log exports empty text (no crash)
    log.start()
    assert log.log_to_text() == ""


def test_mainwindow_save_log_button_writes_file(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from lazystretch.gui.main_window import MainWindow

    w = MainWindow()
    assert hasattr(w, "save_log_btn")
    w.log_view.append("hello from the run log")
    path = tmp_path / "m.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    w.save_log_btn.click()
    assert path.exists() and "hello from the run log" in path.read_text()


def test_develop_log_export(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from lazystretch.gui.develop_window import DevelopLog

    log = DevelopLog()
    log.add("Curves", "+0.20")
    assert "Curves" in log.log_to_text() and "+0.20" in log.log_to_text()
    path = tmp_path / "d.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    assert log.save_log() == str(path)
    assert "LazyDevelop log" in path.read_text()


def test_shell_launcher_and_lazy_panel(qapp):
    from lazystretch.gui.main_window import LazyStretchPanel, MainWindow
    from lazystretch.gui.shell import AppShell

    assert MainWindow is LazyStretchPanel                 # backward-compat alias
    s = AppShell()
    assert s.stack.currentWidget() is s.launcher          # starts on the launcher
    assert s.windowTitle() == "LazyAstroPhotoSuite"
    assert s._panels == {}                                # panels built lazily
    s.open_tool("stretch")
    assert isinstance(s._panels.get("stretch"), LazyStretchPanel)
    assert s.stack.currentWidget() is s._panels["stretch"]
    assert s.windowTitle() == "LazyStretch"
    same = s._panels["stretch"]
    s.show_home(); s.open_tool("stretch")
    assert s._panels["stretch"] is same                   # reuses, no rebuild
    assert s._tool_actions["stack"].isEnabled()            # all three tools now available
    assert s._tool_actions["moonsun"].isEnabled()
    s.show_home()
    assert s.stack.currentWidget() is s.launcher


def test_shell_launcher_toolbar_toggles(qapp):
    from lazystretch.gui.shell import AppShell

    s = AppShell()
    assert s.toolbar.isHidden()                            # hidden on the launcher
    s.open_tool("stretch")
    assert not s.toolbar.isHidden()                        # back-to-launcher button shown in a tool
    assert s._tool_name_label.text() == "LazyStretch"
    s.show_home()
    assert s.toolbar.isHidden()                            # hidden again at home


def test_shell_opens_stack_panel(qapp):
    from lazystretch.gui.shell import AppShell
    from lazystretch.gui.stack_window import LazyStackPanel
    from lazystretch.lazystack.model import LazyStackParams

    s = AppShell()
    s.open_tool("stack")
    assert isinstance(s._panels.get("stack"), LazyStackPanel)
    assert s.windowTitle() == "LazyStack"
    panel = s._panels["stack"]
    p = panel._collect_params()
    assert isinstance(p, LazyStackParams)


def test_shell_opens_flight_panel(qapp):
    from lazystretch.gui.flight_window import LazyFlightPanel
    from lazystretch.gui.shell import AppShell
    from lazystretch.animate.render import Cam
    from lazystretch.animate.render import Flythrough3D
    from lazystretch.animate.volume3d import SpaceFly
    from lazystretch.animate.flyv2 import V2Fly

    s = AppShell()
    s.open_tool("fly")
    panel = s._panels.get("fly")
    assert isinstance(panel, LazyFlightPanel)
    assert s.windowTitle() == "LazyFlight"

    # feed a synthetic image (bypass the file dialog) and drive the preview
    h, w = 120, 180
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((yy - h / 2) / 30.0) ** 2 + ((xx - w / 2) / 45.0) ** 2))
    img = np.stack([blob * 0.9, blob * 0.5, blob * 0.35], axis=-1).astype(np.float32)
    img[40, 60] = 1.0
    panel.img = img
    panel._masks = None

    # v2 is the only offered mode (mode switch + non-v2 controls are hidden)
    assert panel._mode() == "v2"
    assert panel._mode_row.isHidden() and panel._path_row.isHidden()
    assert panel.style.isHidden() and panel.semantic.isHidden()
    panel._reopen_engine()
    assert isinstance(panel._engine, V2Fly) and len(panel._cams) > 1
    panel.scrub.setValue(50)
    # with no pan points, Rotate/Tilt are a constant orientation from the sliders
    panel.rotate.set_value(12.0); panel.tilt_x.set_value(3.0); panel.tilt_y.set_value(-2.0)
    panel._rebuild_cams()
    assert all(c.roll == pytest.approx(12.0) for c in panel._cams)
    assert panel._cams[0].rot_x == pytest.approx(3.0, abs=0.1)
    assert panel._cams[0].rot_y == pytest.approx(-2.0, abs=0.1)
    panel._render_preview()                              # must not raise
    # portrait output frame is taller than wide
    panel.orient_combo.setCurrentText("Portrait"); panel._reopen_engine()
    assert panel._engine.out_h > panel._engine.out_w
    panel.star_min.set_value(1.5); panel.star_max.set_value(6.0); panel._on_star_size()
    assert panel._engine.star_max == pytest.approx(6.0, abs=0.02)   # slider quantises
    # pan points are keyframes: x, y, zoom, roll, rot_x, rot_y (all interpolated)
    panel.zoom.set_value(1.1); panel._on_point_added(-0.4, -0.2)
    panel.zoom.set_value(2.0); panel._on_point_added(0.3, 0.4)
    assert len(panel._pan_points) == 2 and len(panel._pan_points[0]) == 6
    assert panel._pan_points[0][2] == pytest.approx(1.1, abs=0.02)  # point 1 zoom
    # selecting point 1 reflects its channels onto the sliders
    panel._on_point_selected(0)
    assert panel.zoom.value() == pytest.approx(1.1, abs=0.02)
    # Rotate/Tilt sliders now keyframe the SELECTED point (like zoom)
    panel._on_point_channel(3, 9.0)                      # roll of point 1
    panel._on_point_channel(4, 2.0)                      # tilt X of point 1
    assert panel._pan_points[0][3] == pytest.approx(9.0)
    assert panel._pan_points[0][4] == pytest.approx(2.0)
    panel._on_point_moved(1, -0.1, 0.2)
    assert panel._pan_points[1][0] == pytest.approx(-0.1, abs=1e-6)
    panel._rebuild_cams()
    assert panel._cams[0].zoom == pytest.approx(1.1, abs=0.05)      # start = point 1
    assert panel._cams[0].roll == pytest.approx(9.0, abs=0.2)       # start roll = point 1
    panel._toggle_pick(True); assert panel.canvas.pick_mode
    panel._toggle_pick(False)

    # dragging moves the BACKGROUND within the frame, CLAMPED to the image borders
    panel._pan_points = []; panel._base_pan = [0.0, 0.0]; panel._rebuild_cams()
    mx, my = panel._base_pan_limit()
    panel._on_bg_pan(0.05, 0.0); assert panel._base_pan[0] > 0.0   # moves
    panel._on_bg_pan(9.0, 9.0)                                      # huge drag
    assert panel._base_pan[0] <= mx + 1e-6 and panel._base_pan[1] <= my + 1e-6
    panel._render_preview()                                        # applies base pan

    # the other engines still exist under the hood (mode combo is hidden but live)
    panel.mode_combo.setCurrentIndex(0)
    assert panel._mode() == "space" and isinstance(panel._engine, SpaceFly)
    panel.mode_combo.setCurrentIndex(3)
    assert panel._mode() == "volumetric"
    assert isinstance(panel._engine, Flythrough3D)
    # only dolly paths are offered (flyby/orbit shear the gas)
    assert {panel.path_combo.itemText(i) for i in range(panel.path_combo.count())} \
        == {"flythrough", "pullback"}
    panel.path_combo.setCurrentText("pullback"); panel._rebuild_cams()
    assert isinstance(panel._cams[0], Cam)
    # playback advances the scrub and renders directly (no debounce starvation)
    panel.play_btn.setChecked(True)
    v0 = panel.scrub.value()
    panel._advance_play()
    assert panel.scrub.value() != v0
    panel.play_btn.setChecked(False)


def test_flight_recipe_save_load_roundtrip(qapp, tmp_path, monkeypatch):
    """v2 fly-through settings + pan keyframes survive a save/load cycle."""
    from PySide6.QtWidgets import QFileDialog
    from lazystretch.gui.flight_window import LazyFlightPanel
    from lazystretch.gui.shell import AppShell

    s = AppShell()
    s.open_tool("fly")
    panel = s._panels.get("fly")
    h, w = 100, 160
    panel.img = np.zeros((h, w, 3), np.float32)
    panel._masks = None
    panel._reopen_engine()

    # set a distinctive configuration
    panel.dur.set_value(20.0)
    panel.zoom.set_value(1.6)
    panel.rotate.set_value(-5.0)
    panel.tilt_x.set_value(2.0)
    panel.tilt_y.set_value(-1.0)
    panel.bloom.set_value(0.4)
    panel.streaks.setChecked(True)
    panel.streak_len.set_value(80.0)
    panel.star_min.set_value(1.2); panel.star_max.set_value(5.0)
    panel.orient_combo.setCurrentText("Portrait")
    panel.ratio_combo.setCurrentText("4:3")
    panel.quality_combo.setCurrentText("Max")
    panel.zoom.set_value(1.1); panel._on_point_added(-0.3, -0.1)
    panel.zoom.set_value(1.9); panel._on_point_added(0.2, 0.3)
    panel._base_pan = [0.04, -0.02]

    path = tmp_path / "fly.lfrecipe"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    panel._save_recipe()
    assert path.exists()

    # fresh panel, load it back
    s2 = AppShell(); s2.open_tool("fly")
    p2 = s2._panels.get("fly")
    p2.img = np.zeros((h, w, 3), np.float32); p2._masks = None; p2._reopen_engine()
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    p2._load_recipe()

    assert p2.dur.value() == pytest.approx(20.0, abs=0.1)
    assert p2.zoom.value() == pytest.approx(1.9, abs=0.05)   # last set = point 2 slider
    assert p2.tilt_x.value() == pytest.approx(2.0, abs=0.05)
    assert p2.streaks.isChecked()
    assert p2.streak_len.value() == pytest.approx(80.0, abs=0.5)
    assert p2.orient_combo.currentText() == "Portrait"
    assert p2.ratio_combo.currentText() == "4:3"
    assert p2.quality_combo.currentText() == "Max"
    assert len(p2._pan_points) == 2 and len(p2._pan_points[0]) == 6
    assert p2._pan_points[0][2] == pytest.approx(1.1, abs=0.02)
    assert p2._pan_points[1][0] == pytest.approx(0.2, abs=1e-6)
    assert p2._base_pan[0] == pytest.approx(0.04, abs=1e-6)
    assert p2._engine.out_h > p2._engine.out_w                # portrait applied

    # a non-recipe file is rejected without touching state
    bad = tmp_path / "bad.json"
    bad.write_text('{"kind": "something-else"}')
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(bad), "")))
    p2._load_recipe()
    assert "Not a LazyFlight recipe" in p2.status.text()
    assert len(p2._pan_points) == 2                           # unchanged


def test_shell_opens_moonsun_panel(qapp):
    from lazystretch.gui.moonsun_window import LazyMoonSunPanel
    from lazystretch.gui.shell import AppShell

    s = AppShell()
    s.open_tool("moonsun")
    assert isinstance(s._panels.get("moonsun"), LazyMoonSunPanel)
    assert s.windowTitle() == "LazyMoonSun"
    panel = s._panels["moonsun"]
    # preset round-trips through the dials
    from lazystretch.moonsun.model import MoonSunParams
    panel._apply_params(MoonSunParams.preset("moon"))
    p = panel._collect_params()
    assert p.tone == "neutral" and abs(p.sat - 0.60) < 1e-6 and abs(p.surface) < 1e-6


def test_moonsun_finish_runs_from_source_not_shown(qapp):
    from lazystretch.gui.moonsun_window import LazyMoonSunPanel

    p = LazyMoonSunPanel()
    master = np.full((8, 8, 3), 0.3)
    shown = np.full((8, 8, 3), 0.7)
    # a stack result: 'master' is the base, 'image' is the (finished) display
    p._on_finished({"image": shown, "master": master})
    assert np.allclose(p._source_image, master)        # base = the stacked master
    assert np.allclose(p.result_image, shown)
    # a finish result has no 'master' -> the base is left intact (finish is from ground 0)
    finished = np.full((8, 8, 3), 0.9)
    p._on_finished({"image": finished})
    assert np.allclose(p._source_image, master)        # unchanged, not the finished output
    assert np.allclose(p.result_image, finished)
    # nothing loaded/stacked -> finish is a guarded no-op
    p._source_image = None
    p.worker = None
    p._do_finish()
    assert p.worker is None


def test_left_panel_has_setup_and_adjust_tabs(qapp):
    from PySide6.QtWidgets import QTabWidget

    from lazystretch.gui.main_window import MainWindow

    w = MainWindow()
    tabs = w.findChild(QTabWidget)
    assert tabs is not None
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Setup", "Adjust", "Process"]


def test_fullscreen_button_needs_an_image_then_opens(qapp):
    from lazystretch.gui.main_window import MainWindow
    from lazystretch.gui.preview import FullScreenViewer

    w = MainWindow()
    assert w.preview.current_array() is None
    w._show_fullscreen()                      # nothing shown yet -> no viewer, just a message
    assert w._fs_viewer is None
    w.preview.set_image(np.zeros((20, 30, 3)), keep_view=False)
    w._show_fullscreen()
    assert isinstance(w._fs_viewer, FullScreenViewer)
    assert w._fs_viewer.current_array() is not None
    w._fs_viewer.close()


def test_continue_from_history_sets_base_and_polish_mode(qapp, tmp_path):
    """Continuing from a history item overrides the master as the base and enters polish mode."""
    from lazystretch.gui.main_window import MainWindow
    from lazystretch.io.history import HistoryStore

    w = MainWindow()
    master = tmp_path / "M42.tif"
    master.write_bytes(b"x")
    w._hist_store = HistoryStore(str(master))
    rendered = np.clip(np.random.default_rng(0).random((24, 32, 3)), 0, 1)
    w._hist_store.add(rendered, {"objectClass": "emission"}, "run A", "preview", 12)
    w._refresh_history_list()
    w.history_list.setCurrentRow(0)

    w.crop_slider.set_value(5.0)
    w.checks["inputStretched"].setChecked(False)
    w._continue_from_selected()

    assert w._work_image is not None
    assert w._work_image.shape == (24, 32, 3)
    assert w.checks["inputStretched"].isChecked()          # finished frame -> polish only
    assert abs(w.crop_slider.value()) < 1e-6               # crop dropped (already cropped)
    # _run would now feed the history image, not the master
    p = w._collect_params()
    assert p.inputStretched is True


def test_open_history_folder_creates_and_opens_dir(qapp, tmp_path, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    from lazystretch.gui.main_window import MainWindow
    from lazystretch.io.history import HistoryStore

    w = MainWindow()
    w._hist_store = None
    w._open_history_folder()                       # no master -> graceful, no crash
    master = tmp_path / "M42.tif"
    master.write_bytes(b"x")
    w._hist_store = HistoryStore(str(master))
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl",
                        lambda url: (opened.append(url.toLocalFile()) or True))
    w._open_history_folder()
    assert w._hist_store.dir.is_dir()              # created even before the first run
    assert opened and opened[0] == str(w._hist_store.dir)


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


def test_nightscape_graded_brush_and_cursor(qapp):
    from PySide6.QtCore import QPointF
    from lazystretch.gui.stack_window import LazyStackPanel
    sp = LazyStackPanel()
    assert (sp.ns_bias._lo, sp.ns_bias._hi, sp.ns_bias._dec) == (-0.10, 0.10, 3)   # finer bias
    pv = sp.preview
    pv.set_image(np.stack([np.linspace(0, 1, 200)[None, :].repeat(160, 0)] * 3, -1), keep_view=False)
    pv.set_paint_mode("sky"); pv.set_brush(20); pv.set_falloff(0.6); pv.set_strength(0.8)
    pv._paint_at(QPointF(50, 80))
    s = pv.scribbles()
    assert s.dtype == np.float32 and abs(float(s[80, 50]) - 0.8) < 0.05      # graded, not binary
    assert ((np.abs(s) > 0.05) & (np.abs(s) < 0.7)).any()                    # soft falloff edge
    pv._update_cursor_geom(QPointF(50, 80))
    assert abs(pv._cursor_outer.rect().width() / 2 - 20) < 1                 # cursor = brush size
    assert abs(pv._cursor_inner.rect().width() / 2 - 8) < 1                  # core = size*(1-falloff)


def test_nightscape_fill_opposite(qapp):
    from PySide6.QtCore import QPointF
    from lazystretch.gui.stack_window import LazyStackPanel
    sp = LazyStackPanel()
    pv = sp.preview
    pv.set_image(np.zeros((160, 200, 3)), keep_view=False)
    pv.set_paint_mode("sky"); pv.set_brush(15); pv.set_strength(1.0)
    pv._paint_at(QPointF(40, 40))
    assert not (pv.scribbles() < -0.5).any()                                 # only sky so far
    assert pv.fill_opposite() is True                                        # one class → fillable
    assert (pv.scribbles() < -0.5).any()                                     # rest filled as earth
    pv.clear_scribbles()
    assert pv.fill_opposite() is False                                       # nothing painted → no-op
