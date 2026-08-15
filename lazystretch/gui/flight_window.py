"""LazyFlight panel — turn a finished still into a 3D fly-through video.

Open a stretched/developed master, dial the camera move and look, scrub a live
low-res preview, then render the full-resolution mp4 off the UI thread. Modes:
``space 3D`` (the default — recognisable nebula + synthetic 3D star field + haze,
user-definable colour), ``parallax`` (fast depth warp), ``volumetric`` (glow).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGraphicsEllipseItem,
    QGraphicsSimpleTextItem, QGraphicsView, QGroupBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from ..animate import render_flythrough
from ..animate.clip import PATHS, _resize, build_cameras
from ..animate.encode import ffmpeg_available
from ..animate.parallel import auto_workers
from ..animate.render import Flythrough3D
from ..animate.volume3d import SpaceFly, fly_volume, render_space
from ..animate.flyv2 import V2Fly, V2Cam, fly_v2, render_v2
from ..io.image_io import load_image
from .preview import PreviewView
from .widgets import FloatSlider
from .worker import CallableWorker

_PREVIEW_W = 760          # live-preview render width (fast)
_PREVIEW_FPS = 24         # frames the scrub bar addresses
_RATIOS = ["16:9", "3:2", "4:3", "5:4", "1:1"]
_QUALITY = {"High": 16, "Max": 12, "Good": 20, "Draft": 24}   # name → x264 CRF


class FlightCanvas(PreviewView):
    """Preview that (a) drags the BACKGROUND within the output frame, and (b) in
    pan-point edit mode adds/selects/moves numbered pan points."""

    pointAdded = Signal(float, float)                    # normalised [-1, 1]
    pointMoved = Signal(int, float, float)
    pointSelected = Signal(int)
    backgroundPanned = Signal(float, float)              # incremental pan delta

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.NoDrag)           # we move the background, not the view
        self.pick_mode = False
        self._pts = []                                   # [(x, y), …] normalised
        self._sel = -1
        self._drag = -1
        self._markers = []
        self._bg_last = None                             # left-drag: background move (scene)
        self._view_last = None                           # middle/right-drag: pan the view (screen)

    def set_image(self, a, keep_view=None):
        """Fast-path preview updates: on a same-size frame just swap the pixmap so
        the view (and any zoom) is untouched; a new size fits to window."""
        a = np.asarray(a)
        if (self._has_image and self._img_size == (a.shape[1], a.shape[0])
                and keep_view is not False):
            from PySide6.QtGui import QPixmap
            from .preview import ndarray_to_qimage
            self._array = a
            self._item.setPixmap(QPixmap.fromImage(ndarray_to_qimage(a)))
            return
        super().set_image(a, keep_view=keep_view)

    def _norm(self, ev):
        sp = self.mapToScene(ev.position().toPoint())
        w, h = self._img_size
        return (float(np.clip(sp.x() / w * 2.0 - 1.0, -1.0, 1.0)),
                float(np.clip(sp.y() / h * 2.0 - 1.0, -1.0, 1.0)))

    def _hit(self, nx, ny):
        for i, (px, py) in enumerate(self._pts):
            if (px - nx) ** 2 + (py - ny) ** 2 < 0.06 ** 2:
                return i
        return -1

    def mousePressEvent(self, ev):
        if ev.button() in (Qt.MiddleButton, Qt.RightButton):   # pan the zoomed-in view
            self._view_last = ev.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if not self._img_size:
            return super().mousePressEvent(ev)
        if self.pick_mode:
            nx, ny = self._norm(ev)
            i = self._hit(nx, ny)
            if i >= 0:
                self._drag = i
                self.pointSelected.emit(i)
            else:
                self.pointAdded.emit(nx, ny)
            return
        if ev.button() == Qt.LeftButton:                 # start dragging the background
            self._bg_last = self.mapToScene(ev.position().toPoint())
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._view_last is not None:                  # scroll the view (hand-pan)
            d = ev.position() - self._view_last
            self._view_last = ev.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(d.y()))
            return
        if self.pick_mode and self._drag >= 0 and self._img_size:
            nx, ny = self._norm(ev)
            self.pointMoved.emit(self._drag, nx, ny)
            return
        if self._bg_last is not None and (ev.buttons() & Qt.LeftButton):
            cur = self.mapToScene(ev.position().toPoint())
            w, h = self._img_size
            # move the background under the cursor (content follows the drag)
            self.backgroundPanned.emit(-2.0 * (cur.x() - self._bg_last.x()) / w,
                                       -2.0 * (cur.y() - self._bg_last.y()) / h)
            self._bg_last = cur
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._view_last is not None:
            self._view_last = None
            self.unsetCursor()
            return
        if self.pick_mode and self._drag >= 0:
            self._drag = -1
            return
        if self._bg_last is not None:
            self._bg_last = None
            return
        super().mouseReleaseEvent(ev)

    def set_points(self, points, selected=-1):
        self._pts = list(points)
        self._sel = selected
        for m in self._markers:
            self._scene.removeItem(m)
        self._markers = []
        if not self._img_size:
            return
        w, h = self._img_size
        r = max(w, h) * 0.022
        for i, (px, py) in enumerate(self._pts):
            x = (px + 1.0) * w / 2.0
            y = (py + 1.0) * h / 2.0
            sel = i == self._sel
            col = QColor(255, 215, 90) if sel else QColor(120, 220, 255)
            pen = QPen(col); pen.setCosmetic(True); pen.setWidth(3 if sel else 2)
            e = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
            e.setPen(pen); e.setBrush(QColor(col.red(), col.green(), col.blue(), 70))
            e.setZValue(6)
            self._scene.addItem(e); self._markers.append(e)
            t = QGraphicsSimpleTextItem(str(i + 1))
            f = QFont(); f.setPixelSize(max(int(r * 1.3), 6)); f.setBold(True)
            t.setFont(f)
            br = t.boundingRect()
            t.setPos(x - br.width() / 2.0, y - br.height() / 2.0)
            t.setBrush(QColor(15, 18, 28)); t.setZValue(7)
            self._scene.addItem(t); self._markers.append(t)


class LazyFlightPanel(QWidget):
    """Compose a fly-through from a still and render it to mp4."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.img: Optional[np.ndarray] = None          # full-res float RGB
        self._engine: Optional[Flythrough3D] = None    # preview engine (low-res)
        self._cams = []                                # preview camera path
        self._masks = None                             # semantic masks (optional)
        self._pan_points = []                          # v2 keyframes [x, y, zoom]
        self._sel_point = -1                           # selected pan point index
        self._base_pan = [0.0, 0.0]                     # v2 background framing offset
        self.worker: Optional[CallableWorker] = None
        self._busy = False
        self._play = QTimer(self)
        self._play.setInterval(int(1000 / _PREVIEW_FPS))
        self._play.timeout.connect(self._advance_play)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(self._render_preview)
        self._rebuild_timer = QTimer(self)               # engine rebuild (style/haze/…)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(350)
        self._rebuild_timer.timeout.connect(self._reopen_engine)

        self._build_ui()
        self._set_controls_enabled(False)
        if not ffmpeg_available():
            self.status.setText("ffmpeg not found — install it or `pip install "
                                "imageio-ffmpeg` to render video.")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # --- left: controls ---------------------------------------------------
        side = QVBoxLayout()
        side.setSpacing(8)
        panel = QWidget()
        panel.setLayout(side)
        panel.setFixedWidth(320)

        self.open_btn = QPushButton("Open image…")
        self.open_btn.clicked.connect(self._open)
        side.addWidget(self.open_btn)
        self.info = QLabel("No image loaded.")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: gray;")
        side.addWidget(self.info)

        cam_box = QGroupBox("Camera move")
        cam = QVBoxLayout(cam_box)
        self.path_combo = QComboBox()
        # dolly moves only — they read cleanly through the depth warp; lateral
        # paths (flyby/orbit) shear the gas and are intentionally not offered here
        self.path_combo.addItems(["flythrough", "pullback"])
        self.path_combo.currentIndexChanged.connect(self._rebuild_cams)
        self._path_row = self._row("Path", self.path_combo)
        cam.addWidget(self._path_row)
        self.dur = FloatSlider("Duration (s)", 3.0, 120.0, 8.0, decimals=0)
        self.dur.valueChanged.connect(self._rebuild_cams)
        cam.addWidget(self.dur)
        self.zoom = FloatSlider("Zoom", 1.0, 3.0, 1.4, decimals=2)
        self.zoom.valueChanged.connect(lambda v: self._on_point_channel(2, v))
        cam.addWidget(self.zoom)
        self.rotate = FloatSlider("Rotate Z (°)", -12.0, 12.0, 0.0, decimals=0)
        self.rotate.valueChanged.connect(lambda v: self._on_point_channel(3, v))
        cam.addWidget(self.rotate)
        self.tilt_x = FloatSlider("Tilt X (°)", -4.0, 4.0, 0.0, decimals=1)
        self.tilt_x.valueChanged.connect(lambda v: self._on_point_channel(4, v))
        cam.addWidget(self.tilt_x)
        self.tilt_y = FloatSlider("Tilt Y (°)", -4.0, 4.0, 0.0, decimals=1)
        self.tilt_y.valueChanged.connect(lambda v: self._on_point_channel(5, v))
        cam.addWidget(self.tilt_y)
        side.addWidget(cam_box)

        look_box = QGroupBox("Look")
        look = QVBoxLayout(look_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["space 3D", "v2 (image + stars)",
                                  "parallax (fast)", "volumetric (glow)"])
        self.mode_combo.setCurrentIndex(1)               # v2 is the only offered mode
        self.mode_combo.currentIndexChanged.connect(self._reopen_engine)
        self._mode_row = self._row("Mode", self.mode_combo)
        look.addWidget(self._mode_row)
        self.bloom = FloatSlider("Bloom", 0.0, 1.0, 0.32, decimals=2)
        self.bloom.valueChanged.connect(self._on_bloom)
        look.addWidget(self.bloom)
        # --- Space-3D controls (rebuild the engine → debounced) ---------------
        self.style = FloatSlider("Style (real→space)", 0.0, 1.0, 0.0, decimals=2)
        self.style.valueChanged.connect(self._rebuild_soon)
        look.addWidget(self.style)
        self.saturation = FloatSlider("Saturation", 0.5, 2.5, 1.35, decimals=2)
        self.saturation.valueChanged.connect(self._rebuild_soon)
        look.addWidget(self.saturation)
        self.haze = FloatSlider("Haze", 0.0, 0.6, 0.22, decimals=2)
        self.haze.valueChanged.connect(self._rebuild_soon)
        look.addWidget(self.haze)
        self.stars = FloatSlider("Stars", 300, 2500, 1300, decimals=0)
        self.stars.valueChanged.connect(self._rebuild_soon)
        look.addWidget(self.stars)
        self.star_min = FloatSlider("Star min size", 0.3, 3.0, 0.9, decimals=2)
        self.star_min.valueChanged.connect(self._on_star_size)
        look.addWidget(self.star_min)
        self.star_max = FloatSlider("Star max size", 1.0, 8.0, 3.6, decimals=2)
        self.star_max.valueChanged.connect(self._on_star_size)
        look.addWidget(self.star_max)
        self.streaks = QCheckBox("Radial star streaks (v2)")
        self.streaks.toggled.connect(self._on_streaks)
        look.addWidget(self.streaks)
        self.streak_len = FloatSlider("Streak strength", 0.0, 160.0, 60.0, decimals=0)
        self.streak_len.valueChanged.connect(self._on_streak_len)
        look.addWidget(self.streak_len)
        self.semantic = QCheckBox("Mask-driven depth (recommended)")
        self.semantic.setChecked(True)                   # the high-quality default
        self.semantic.toggled.connect(self._reopen_engine)
        look.addWidget(self.semantic)
        self.show_depth = QCheckBox("Show depth map")
        self.show_depth.toggled.connect(self._render_preview)
        look.addWidget(self.show_depth)
        side.addWidget(look_box)

        out_box = QGroupBox("Output")
        out = QVBoxLayout(out_box)
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["24", "30"])
        out.addWidget(self._row("FPS", self.fps_combo))
        self.width_combo = QComboBox()
        self.width_combo.addItems(["960", "1280", "1600", "1920", "2560", "3840"])
        self.width_combo.setCurrentText("1280")
        out.addWidget(self._row("Long edge", self.width_combo))
        self.orient_combo = QComboBox()                  # v2 output framing
        self.orient_combo.addItems(["Landscape", "Portrait", "Square"])
        self.orient_combo.currentIndexChanged.connect(self._reopen_engine)
        out.addWidget(self._row("Orientation", self.orient_combo))
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(_RATIOS)
        self.ratio_combo.currentIndexChanged.connect(self._reopen_engine)
        out.addWidget(self._row("Aspect", self.ratio_combo))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(list(_QUALITY.keys()) + ["Custom bitrate"])
        self.quality_combo.setCurrentText("High")
        self.quality_combo.currentIndexChanged.connect(self._sync_quality)
        out.addWidget(self._row("Quality", self.quality_combo))
        self.bitrate = FloatSlider("Bitrate (Mbps)", 5.0, 150.0, 60.0, decimals=0)
        out.addWidget(self.bitrate)
        self.parallel = QCheckBox(f"Parallel render ({auto_workers()} cores)")
        self.parallel.setChecked(True)
        out.addWidget(self.parallel)
        rec = QHBoxLayout()
        self.save_recipe_btn = QPushButton("Save recipe…")
        self.save_recipe_btn.clicked.connect(self._save_recipe)
        rec.addWidget(self.save_recipe_btn)
        self.load_recipe_btn = QPushButton("Load recipe…")
        self.load_recipe_btn.clicked.connect(self._load_recipe)
        rec.addWidget(self.load_recipe_btn)
        out.addLayout(rec)
        self.render_btn = QPushButton("Render video…")
        self.render_btn.clicked.connect(self._render_video)
        out.addWidget(self.render_btn)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        out.addWidget(self.progress)
        side.addWidget(out_box)

        side.addStretch(1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: rgba(150,170,200,0.9); font-size: 12px;")
        side.addWidget(self.status)
        root.addWidget(panel)

        # --- right: canvas + scrub -------------------------------------------
        right = QVBoxLayout()
        self.canvas = FlightCanvas()
        self.canvas.setToolTip("Left-drag: move background in the frame · "
                               "scroll: zoom the view · middle/right-drag: pan the view")
        self.canvas.pointAdded.connect(self._on_point_added)
        self.canvas.pointMoved.connect(self._on_point_moved)
        self.canvas.pointSelected.connect(self._on_point_selected)
        self.canvas.backgroundPanned.connect(self._on_bg_pan)
        right.addWidget(self.canvas, 1)
        pp = QHBoxLayout()
        self.pick_btn = QPushButton("Edit pan points (v2)")
        self.pick_btn.setCheckable(True)
        self.pick_btn.toggled.connect(self._toggle_pick)
        pp.addWidget(self.pick_btn)
        self.clear_pts_btn = QPushButton("Clear")
        self.clear_pts_btn.clicked.connect(self._clear_points)
        pp.addWidget(self.clear_pts_btn)
        self.pts_label = QLabel("0 points")
        self.pts_label.setStyleSheet("color: gray;")
        pp.addWidget(self.pts_label)
        pp.addStretch(1)
        right.addLayout(pp)
        bar = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._toggle_play)
        bar.addWidget(self.play_btn)
        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, 100)
        self.scrub.valueChanged.connect(self._on_scrub)
        bar.addWidget(self.scrub, 1)
        right.addLayout(bar)
        root.addLayout(right, 1)

        # v2 is the only offered mode → hide the mode switch and the controls that
        # belong to the other (space/parallax/volumetric) engines
        for w in (self._mode_row, self._path_row, self.style, self.saturation,
                  self.haze, self.semantic, self.show_depth):
            w.hide()

    def _row(self, label: str, widget: QWidget) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(label)
        lab.setMinimumWidth(96)
        lay.addWidget(lab)
        lay.addWidget(widget, 1)
        return w

    # --------------------------------------------------------------- helpers
    def _set_controls_enabled(self, on: bool):
        for w in (self.path_combo, self.dur, self.zoom, self.rotate, self.tilt_x,
                  self.tilt_y, self.bloom, self.style, self.saturation,
                  self.haze, self.stars,
                  self.star_min, self.star_max, self.streaks, self.streak_len,
                  self.mode_combo, self.semantic, self.show_depth, self.fps_combo,
                  self.width_combo, self.orient_combo, self.ratio_combo,
                  self.quality_combo, self.bitrate, self.render_btn, self.pick_btn,
                  self.clear_pts_btn, self.save_recipe_btn, self.load_recipe_btn,
                  self.play_btn, self.scrub):
            w.setEnabled(on)
        if on:
            self._sync_mode_controls()
            self._sync_quality()

    def _sync_quality(self, *_):
        self.bitrate.setEnabled(self.quality_combo.currentText() == "Custom bitrate")

    def _sync_mode_controls(self):
        mode = self._mode()
        v2 = mode == "v2"
        for w in (self.style, self.saturation, self.haze):
            w.setEnabled(mode == "space")                # space-only look controls
        self.stars.setEnabled(mode in ("space", "v2"))   # both synth star fields
        for w in (self.rotate, self.tilt_x, self.tilt_y, self.streaks,
                  self.streak_len, self.star_min, self.star_max, self.orient_combo,
                  self.ratio_combo, self.pick_btn, self.clear_pts_btn):
            w.setEnabled(v2)                              # v2-only
        self.semantic.setEnabled(mode in ("space", "parallax", "volumetric"))
        self.path_combo.setEnabled(mode in ("parallax", "volumetric"))

    def _aspect_wh(self) -> float:
        a, b = (float(x) for x in self.ratio_combo.currentText().split(":"))
        wh = a / b
        orient = self.orient_combo.currentText()
        if orient == "Portrait":
            return 1.0 / wh
        if orient == "Square":
            return 1.0
        return wh

    def _frame_size(self, long_edge: int):
        wh = self._aspect_wh()
        if wh >= 1.0:
            w, h = long_edge, int(round(long_edge / wh))
        else:
            h, w = long_edge, int(round(long_edge * wh))
        return w - w % 2, h - h % 2

    def _on_star_size(self, *_):
        if isinstance(self._engine, V2Fly):              # read at render time, no rebuild
            lo, hi = float(self.star_min.value()), float(self.star_max.value())
            self._engine.star_min = min(lo, hi)
            self._engine.star_max = max(lo, hi)
        self._render_preview()

    def _on_point_channel(self, idx: int, val: float):
        # in v2 the Zoom/Rotate/Tilt sliders edit the selected pan point's channel
        # (point 1 = starting value); with no point/selection they are the defaults
        if self._mode() == "v2" and 0 <= self._sel_point < len(self._pan_points):
            self._pan_points[self._sel_point][idx] = float(val)
            if self.canvas.pick_mode:
                self._draw_points()
        self._rebuild_cams()

    # ------------------------------------------------------------ pan points
    _POINT_SLIDERS = ((2, "zoom"), (3, "rotate"), (4, "tilt_x"), (5, "tilt_y"))

    def _draw_points(self):
        self.canvas.set_points([(p[0], p[1]) for p in self._pan_points],
                               self._sel_point)

    def _select_point(self, i: int):
        self._sel_point = i
        if 0 <= i < len(self._pan_points):
            p = self._pan_points[i]
            for idx, name in self._POINT_SLIDERS:        # reflect this point's channels
                sl = getattr(self, name)
                sl.blockSignals(True); sl.set_value(p[idx]); sl.blockSignals(False)
        self.pts_label.setText(f"{len(self._pan_points)} pts"
                               + (f" · #{i + 1}" if 0 <= i < len(self._pan_points)
                                  else ""))

    def _toggle_pick(self, on: bool):
        self.canvas.pick_mode = bool(on) and self._mode() == "v2"
        self.pick_btn.setText("Done" if self.canvas.pick_mode else
                              "Edit pan points (v2)")
        if self.canvas.pick_mode:
            if self.play_btn.isChecked():
                self.play_btn.setChecked(False)
            self._show_pick_frame()
        else:
            self.canvas.set_points([])
            self._rebuild_cams()

    def _show_pick_frame(self):
        if isinstance(self._engine, V2Fly):
            cam = V2Cam(pan_x=self._base_pan[0], pan_y=self._base_pan[1])
            self.canvas.set_image(self._engine._warp_bg(cam), keep_view=True)
            self._draw_points()

    def _on_point_added(self, nx: float, ny: float):
        if len(self._pan_points) >= 5:
            return
        self._pan_points.append([nx, ny, float(self.zoom.value()),
                                 float(self.rotate.value()),
                                 float(self.tilt_x.value()), float(self.tilt_y.value())])
        self._select_point(len(self._pan_points) - 1)
        self._draw_points()

    def _on_point_moved(self, i: int, nx: float, ny: float):
        if 0 <= i < len(self._pan_points):
            self._pan_points[i][0] = nx
            self._pan_points[i][1] = ny
            self._select_point(i)
            self._draw_points()

    def _on_point_selected(self, i: int):
        self._select_point(i)
        self._draw_points()

    def _clear_points(self):
        self._pan_points = []
        self._sel_point = -1
        self.pts_label.setText("0 pts")
        if self.canvas.pick_mode:
            self._show_pick_frame()
        else:
            self.canvas.set_points([])
            self._rebuild_cams()

    def _rebuild_soon(self, *_):
        """A space-look control changed → rebuild the engine, debounced."""
        if self.img is not None and not self._busy:
            self._rebuild_timer.start()

    def _set_busy(self, busy: bool):
        self._busy = busy
        self.open_btn.setEnabled(not busy)
        self._set_controls_enabled(not busy and self.img is not None)

    def _mode(self) -> str:
        return {0: "space", 1: "v2", 2: "parallax", 3: "volumetric"}.get(
            self.mode_combo.currentIndex(), "space")

    # ------------------------------------------------------------------ open
    def _open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open stretched image", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.fits *.fit *.fts *.xisf);;All files (*)")
        if not path:
            return
        try:
            loaded = load_image(path)
        except Exception as exc:                        # non-modal per GUI rule
            self.status.setText(f"Open failed: {exc}")
            return
        self.img = np.clip(np.nan_to_num(np.asarray(loaded.data, np.float32)), 0, 1)
        if self.img.ndim == 2:
            self.img = np.repeat(self.img[..., None], 3, axis=2)
        h, w = self.img.shape[:2]
        self.info.setText(f"{Path(path).name}\n{w} × {h}")
        self._masks = None
        self._base_pan = [0.0, 0.0]
        self._reopen_engine()
        self._set_controls_enabled(True)

    def _reopen_engine(self):
        """(Re)build the low-res preview engine for the current image/options."""
        if self.img is None or self._busy:
            return
        self.status.setText("Preparing preview…")
        QWidget.repaint(self)
        small = _resize(self.img, _PREVIEW_W)
        mode = self._mode()
        need_masks = mode != "v2" and self.semantic.isChecked()   # v2 needs no masks
        if need_masks:
            if self._masks is None:                      # cached per image (slow)
                from ..develop.semantic import segment
                self._masks = segment(small)
        elif mode != "v2":
            self._masks = None
        if mode == "space":
            self._engine = SpaceFly(small, masks=self._masks,
                                    saturation=float(self.saturation.value()),
                                    stylize=float(self.style.value()),
                                    haze=float(self.haze.value()),
                                    star_count=int(self.stars.value()),
                                    haze_slabs=5, bloom=float(self.bloom.value()))
        elif mode == "v2":
            ow, oh = self._frame_size(_PREVIEW_W)
            self._engine = V2Fly(small, out_w=ow, out_h=oh,
                                 star_count=int(self.stars.value()),
                                 bloom=float(self.bloom.value()),
                                 star_min=float(self.star_min.value()),
                                 star_max=float(self.star_max.value()),
                                 streaks=self.streaks.isChecked(),
                                 streak_len=float(self.streak_len.value()))
        else:
            self._engine = Flythrough3D(small, masks=self._masks,
                                        bloom=float(self.bloom.value()), mode="parallax")
        self._sync_mode_controls()
        self.status.setText("")
        self._rebuild_cams()

    def _rebuild_cams(self):
        if self._engine is None:
            return
        n = max(int(round(self.dur.value() * _PREVIEW_FPS)), 2)
        zoom_end = float(self.zoom.value())
        mode = self._mode()
        if mode == "space":
            c_end = float(np.clip((zoom_end - 1.0) * 2.2, 0.1, 0.9))
            self._cams = fly_volume(n, c_end=c_end, zoom_end=zoom_end)
        elif mode == "v2":
            self._cams = fly_v2(n, zoom_end=zoom_end,
                                rotate_deg=float(self.rotate.value()),
                                rot_x=float(self.tilt_x.value()),
                                rot_y=float(self.tilt_y.value()),
                                pan_points=list(self._pan_points))
            self._clamp_base_pan()                       # keep framing within borders
        else:
            try:
                self._cams = build_cameras(self.path_combo.currentText(), n,
                                           zoom_end=zoom_end)
            except TypeError:                            # path without zoom_end
                self._cams = build_cameras(self.path_combo.currentText(), n)
        self._render_preview()

    def _on_bloom(self, val: float):
        if self._engine is not None:
            self._engine.bloom_strength = float(val)
        self._render_preview()

    def _on_streaks(self, on: bool):
        if isinstance(self._engine, V2Fly):              # read at render-time, no rebuild
            self._engine.streaks = bool(on)
        self._render_preview()

    def _on_streak_len(self, val: float):
        if isinstance(self._engine, V2Fly):              # live, no rebuild
            self._engine.streak_len = float(val)
        self._render_preview()

    def _base_pan_limit(self):
        """Max base pan (per axis) that keeps the output frame within the image —
        so the drag can't reveal a border. Uses the tightest (min-zoom) frame and
        leaves room for the animated pan path."""
        e = self._engine
        if not isinstance(e, V2Fly):
            return 1.5, 1.5
        zmin = min((c.zoom for c in self._cams), default=1.0)
        cover = max(e.out_w / e.bw, e.out_h / e.bh)
        s = e.overscan * cover * max(zmin, 1e-3)
        px = max((abs(c.pan_x) for c in self._cams), default=0.0)   # path excursion
        py = max((abs(c.pan_y) for c in self._cams), default=0.0)
        mx = max(0.0, e.bw * s / e.out_w - 1.0 - px)
        my = max(0.0, e.bh * s / e.out_h - 1.0 - py)
        return mx, my

    def _clamp_base_pan(self):
        mx, my = self._base_pan_limit()
        self._base_pan[0] = float(np.clip(self._base_pan[0], -mx, mx))
        self._base_pan[1] = float(np.clip(self._base_pan[1], -my, my))

    def _on_bg_pan(self, dx: float, dy: float):
        # drag the background within the output frame (v2 base framing offset),
        # clamped so it can't be dragged past the image borders
        if self._mode() != "v2":
            return
        self._base_pan[0] += dx
        self._base_pan[1] += dy
        self._clamp_base_pan()
        if self.canvas.pick_mode:
            self._show_pick_frame()
        else:
            self._render_preview()

    # --------------------------------------------------------------- preview
    def _render_preview(self):
        if self._engine is None or self._busy or self.canvas.pick_mode:
            return
        if self.show_depth.isChecked():
            z = getattr(self._engine, "depth", None)
            if z is None:                                 # SpaceFly keeps it on ._neb
                z = getattr(getattr(self._engine, "_neb", None), "depth", None)
            if z is not None:
                z = z.astype(np.float32)
                self.canvas.set_image(np.repeat(z[..., None], 3, axis=2), keep_view=True)
                return
        if not self._cams:
            return
        i = int(round(self.scrub.value() / 100.0 * (len(self._cams) - 1)))
        cam = self._cams[i]
        if self._mode() == "v2" and (self._base_pan[0] or self._base_pan[1]):
            cam = replace(cam, pan_x=cam.pan_x + self._base_pan[0],
                          pan_y=cam.pan_y + self._base_pan[1])
        frame = self._engine.render_frame(cam)
        self.canvas.set_image(frame, keep_view=True)

    def _on_scrub(self, _v: int):
        # manual drags debounce; during playback frames are rendered directly
        if not self._play.isActive():
            self._debounce.start()

    def _toggle_play(self, on: bool):
        self.play_btn.setText("❚❚ Pause" if on else "▶ Play")
        if on and not self._busy and self._engine is not None and self._cams:
            self._play.start()
        else:
            self._play.stop()
            self.play_btn.setChecked(False)

    def _advance_play(self):
        if self._engine is None or self._busy or not self._cams:
            self._play.stop()
            return
        v = self.scrub.value() + 2
        if v > 100:
            v = 0
        self.scrub.blockSignals(True)                   # don't route through debounce
        self.scrub.setValue(v)
        self.scrub.blockSignals(False)
        self._render_preview()                          # render this frame now

    # ---------------------------------------------------------------- render
    def _render_video(self):
        if self.img is None or self._busy:
            return
        if not ffmpeg_available():
            self.status.setText("ffmpeg not found — cannot render video.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "Render video", "flythrough.mp4",
                                             "MP4 video (*.mp4)")
        if not out:
            return
        if self.play_btn.isChecked():
            self.play_btn.setChecked(False)

        img = self.img
        use_masks = self.semantic.isChecked()
        seconds = float(self.dur.value())
        fps = int(self.fps_combo.currentText())
        path = self.path_combo.currentText()
        width = int(self.width_combo.currentText())
        mode = self._mode()
        zoom_end = float(self.zoom.value())
        rot_x = float(self.tilt_x.value())
        rot_y = float(self.tilt_y.value())
        bloom = float(self.bloom.value())
        sat = float(self.saturation.value())
        stylize = float(self.style.value())
        haze = float(self.haze.value())
        star_count = int(self.stars.value())
        rotate_deg = float(self.rotate.value())
        streaks = self.streaks.isChecked()
        streak_len = float(self.streak_len.value())
        lo, hi = float(self.star_min.value()), float(self.star_max.value())
        star_min, star_max = min(lo, hi), max(lo, hi)
        pan_points = list(self._pan_points)
        base_pan = tuple(self._base_pan)
        out_w, out_h = self._frame_size(width)
        qname = self.quality_combo.currentText()
        if qname == "Custom bitrate":
            crf, bitrate = 17, float(self.bitrate.value())
        else:
            crf, bitrate = _QUALITY[qname], None
        c_end = float(np.clip((zoom_end - 1.0) * 2.2, 0.1, 0.9))
        workers = auto_workers() if self.parallel.isChecked() else 1

        def job(log, progress):
            m = None
            if use_masks and mode != "v2":              # match render res for depth
                from ..develop.semantic import segment
                m = segment(_resize(img, width))
            def on_frame(i, n, _fr):
                progress(i + 1, n, "frame")
            if mode == "v2":
                return render_v2(
                    img, out, seconds=seconds, fps=fps, out_w=out_w, out_h=out_h,
                    workers=workers, star_count=star_count, bloom=bloom,
                    zoom_end=zoom_end, rotate_deg=rotate_deg, rot_x=rot_x,
                    rot_y=rot_y, pan_points=pan_points, base_pan=base_pan,
                    star_min=star_min, star_max=star_max, streaks=streaks,
                    streak_len=streak_len, crf=crf, bitrate_mbps=bitrate,
                    on_frame=on_frame)
            if mode == "space":
                return render_space(
                    img, out, seconds=seconds, fps=fps, render_width=width,
                    workers=workers, masks=m, saturation=sat, stylize=stylize,
                    haze=haze, star_count=star_count, c_end=c_end,
                    zoom_end=zoom_end, engine_kw={"bloom": bloom}, on_frame=on_frame)
            return render_flythrough(
                img, out, seconds=seconds, fps=fps, path=path, mode=mode,
                render_width=width, workers=workers, masks=m,
                engine_kw={"bloom": bloom}, path_kw={"zoom_end": zoom_end},
                on_frame=on_frame)

        self._set_busy(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)                    # indeterminate until first frame
        w_note = f"  ·  {workers} cores" if workers > 1 else ""
        self.status.setText(f"Rendering {mode} @ {width}px{w_note}…")
        self.worker = CallableWorker(job, mode="render")
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _on_progress(self, i: int, n: int, _msg: str):
        if self.progress.maximum() == 0 and n > 0:
            self.progress.setRange(0, n)
        self.progress.setValue(i)

    def _on_done(self, out_path):
        self._set_busy(False)
        self.progress.setVisible(False)
        self.status.setText(f"Saved {Path(str(out_path)).name}")

    def _on_fail(self, msg: str):
        self._set_busy(False)
        self.progress.setVisible(False)
        self.status.setText(f"Render failed: {msg}")

    # ---------------------------------------------------------------- recipe
    def _collect_recipe(self) -> dict:
        """All v2 settings + pan-point keyframes, for save/load."""
        return {
            "kind": "lazyflight-v2", "version": 1,
            "duration": float(self.dur.value()),
            "zoom": float(self.zoom.value()),
            "rotate_z": float(self.rotate.value()),
            "tilt_x": float(self.tilt_x.value()),
            "tilt_y": float(self.tilt_y.value()),
            "bloom": float(self.bloom.value()),
            "stars": int(self.stars.value()),
            "star_min": float(self.star_min.value()),
            "star_max": float(self.star_max.value()),
            "streaks": bool(self.streaks.isChecked()),
            "streak_len": float(self.streak_len.value()),
            "fps": self.fps_combo.currentText(),
            "long_edge": self.width_combo.currentText(),
            "orientation": self.orient_combo.currentText(),
            "aspect": self.ratio_combo.currentText(),
            "quality": self.quality_combo.currentText(),
            "bitrate": float(self.bitrate.value()),
            "pan_points": [[float(v) for v in p] for p in self._pan_points],
            "base_pan": [float(self._base_pan[0]), float(self._base_pan[1])],
        }

    def _apply_recipe(self, r: dict):
        sliders = {"duration": self.dur, "zoom": self.zoom, "rotate_z": self.rotate,
                   "tilt_x": self.tilt_x, "tilt_y": self.tilt_y, "bloom": self.bloom,
                   "stars": self.stars, "star_min": self.star_min,
                   "star_max": self.star_max, "streak_len": self.streak_len,
                   "bitrate": self.bitrate}
        for key, sl in sliders.items():
            if key in r:
                sl.blockSignals(True); sl.set_value(float(r[key])); sl.blockSignals(False)
        self.streaks.blockSignals(True)
        self.streaks.setChecked(bool(r.get("streaks", False)))
        self.streaks.blockSignals(False)
        for key, combo in (("fps", self.fps_combo), ("long_edge", self.width_combo),
                           ("orientation", self.orient_combo), ("aspect", self.ratio_combo),
                           ("quality", self.quality_combo)):
            if r.get(key) is not None:
                combo.blockSignals(True); combo.setCurrentText(str(r[key]))
                combo.blockSignals(False)
        self._pan_points = [list(map(float, p)) for p in r.get("pan_points", [])]
        self._sel_point = -1
        bp = r.get("base_pan", [0.0, 0.0])
        self._base_pan = [float(bp[0]), float(bp[1])]
        self.pts_label.setText(f"{len(self._pan_points)} pts")
        self._sync_quality()
        self._reopen_engine()                            # stars/size/output frame changed
        if self.canvas.pick_mode:
            self._show_pick_frame()

    def _save_recipe(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save fly-through recipe",
                                              "flythrough.lfrecipe",
                                              "LazyFlight recipe (*.lfrecipe);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self._collect_recipe(), f, indent=2)
            self.status.setText(f"Saved recipe {Path(path).name}")
        except Exception as exc:                          # non-modal per GUI rule
            self.status.setText(f"Save recipe failed: {exc}")

    def _load_recipe(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load fly-through recipe", "",
                                              "LazyFlight recipe (*.lfrecipe);;All files (*)")
        if not path:
            return
        try:
            with open(path) as f:
                r = json.load(f)
            if not isinstance(r, dict) or r.get("kind") != "lazyflight-v2":
                self.status.setText("Not a LazyFlight recipe.")
                return
            self._apply_recipe(r)
            self.status.setText(f"Loaded recipe {Path(path).name}")
        except Exception as exc:
            self.status.setText(f"Load recipe failed: {exc}")
