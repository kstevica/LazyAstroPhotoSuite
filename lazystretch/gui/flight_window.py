"""LazyFlight panel — turn a finished still into a 3D fly-through video.

Open a stretched/developed master, dial the camera move and look, scrub a live
low-res preview, then render the full-resolution mp4 off the UI thread. Modes:
``space 3D`` (the default — recognisable nebula + synthetic 3D star field + haze,
user-definable colour), ``parallax`` (fast depth warp), ``volumetric`` (glow).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGraphicsEllipseItem, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget,
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


class FlightCanvas(PreviewView):
    """Preview that can also capture up to N pan-point clicks (v2)."""

    pointPicked = Signal(float, float)                   # normalised [-1, 1]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pick_mode = False
        self._markers = []

    def mousePressEvent(self, ev):
        if self.pick_mode and self._img_size:
            sp = self.mapToScene(ev.position().toPoint())
            w, h = self._img_size
            nx = float(np.clip(sp.x() / w * 2.0 - 1.0, -1.0, 1.0))
            ny = float(np.clip(sp.y() / h * 2.0 - 1.0, -1.0, 1.0))
            self.pointPicked.emit(nx, ny)
            return
        super().mousePressEvent(ev)

    def set_markers(self, points):
        for m in self._markers:
            self._scene.removeItem(m)
        self._markers = []
        if not self._img_size:
            return
        w, h = self._img_size
        r = max(w, h) * 0.013
        pen = QPen(QColor(120, 220, 255)); pen.setCosmetic(True); pen.setWidth(2)
        for (px, py) in points:
            x = (px + 1.0) * w / 2.0
            y = (py + 1.0) * h / 2.0
            it = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
            it.setPen(pen); it.setBrush(QColor(120, 220, 255, 70)); it.setZValue(6)
            self._scene.addItem(it)
            self._markers.append(it)


class LazyFlightPanel(QWidget):
    """Compose a fly-through from a still and render it to mp4."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.img: Optional[np.ndarray] = None          # full-res float RGB
        self._engine: Optional[Flythrough3D] = None    # preview engine (low-res)
        self._cams = []                                # preview camera path
        self._masks = None                             # semantic masks (optional)
        self._pan_points = []                          # v2 pan targets (normalised)
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
        cam.addWidget(self._row("Path", self.path_combo))
        self.dur = FloatSlider("Duration (s)", 3.0, 30.0, 8.0, decimals=0)
        self.dur.valueChanged.connect(self._rebuild_cams)
        cam.addWidget(self.dur)
        self.zoom = FloatSlider("Zoom", 1.05, 2.0, 1.4, decimals=2)
        self.zoom.valueChanged.connect(self._rebuild_cams)
        cam.addWidget(self.zoom)
        self.rotate = FloatSlider("Rotate (°)", 0.0, 30.0, 8.0, decimals=0)
        self.rotate.valueChanged.connect(self._rebuild_cams)
        cam.addWidget(self.rotate)
        self.pan = FloatSlider("Pan", 0.0, 0.12, 0.035, decimals=3)
        self.pan.valueChanged.connect(self._rebuild_cams)
        cam.addWidget(self.pan)
        side.addWidget(cam_box)

        look_box = QGroupBox("Look")
        look = QVBoxLayout(look_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["space 3D", "v2 (image + stars)",
                                  "parallax (fast)", "volumetric (glow)"])
        self.mode_combo.currentIndexChanged.connect(self._reopen_engine)
        look.addWidget(self._row("Mode", self.mode_combo))
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
        self.width_combo.addItems(["960", "1280", "1600", "1920"])
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
        self.parallel = QCheckBox(f"Parallel render ({auto_workers()} cores)")
        self.parallel.setChecked(True)
        out.addWidget(self.parallel)
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
        self.canvas.pointPicked.connect(self._on_point_picked)
        right.addWidget(self.canvas, 1)
        pp = QHBoxLayout()
        self.pick_btn = QPushButton("Pick pan points (v2)")
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
        for w in (self.path_combo, self.dur, self.zoom, self.rotate, self.pan,
                  self.bloom, self.style, self.saturation, self.haze, self.stars,
                  self.star_min, self.star_max, self.streaks, self.mode_combo,
                  self.semantic, self.show_depth, self.fps_combo, self.width_combo,
                  self.orient_combo, self.ratio_combo, self.render_btn,
                  self.pick_btn, self.clear_pts_btn, self.play_btn, self.scrub):
            w.setEnabled(on)
        if on:
            self._sync_mode_controls()

    def _sync_mode_controls(self):
        mode = self._mode()
        v2 = mode == "v2"
        for w in (self.style, self.saturation, self.haze):
            w.setEnabled(mode == "space")                # space-only look controls
        self.stars.setEnabled(mode in ("space", "v2"))   # both synth star fields
        for w in (self.rotate, self.pan, self.streaks, self.star_min, self.star_max,
                  self.orient_combo, self.ratio_combo, self.pick_btn,
                  self.clear_pts_btn):
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

    # ------------------------------------------------------------ pan points
    def _toggle_pick(self, on: bool):
        self.canvas.pick_mode = bool(on) and self._mode() == "v2"
        self.pick_btn.setText("Done picking" if self.canvas.pick_mode else
                              "Pick pan points (v2)")
        if self.canvas.pick_mode:
            if self.play_btn.isChecked():
                self.play_btn.setChecked(False)
            self._show_pick_frame()                      # static base for placing
        else:
            self.canvas.set_markers([])
            self._rebuild_cams()

    def _show_pick_frame(self):
        if isinstance(self._engine, V2Fly):
            self.canvas.set_image(self._engine._warp_bg(V2Cam()), keep_view=True)
            self.canvas.set_markers(self._pan_points)

    def _on_point_picked(self, nx: float, ny: float):
        if len(self._pan_points) >= 5:
            self._pan_points = self._pan_points[1:]      # keep the last 5
        self._pan_points.append((nx, ny))
        self.pts_label.setText(f"{len(self._pan_points)} points")
        self.canvas.set_markers(self._pan_points)

    def _clear_points(self):
        self._pan_points = []
        self.pts_label.setText("0 points")
        self.canvas.set_markers([])
        if self.canvas.pick_mode:
            self._show_pick_frame()
        else:
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
                                 streaks=self.streaks.isChecked())
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
                                pan_points=list(self._pan_points),
                                pan=float(self.pan.value()))
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
        frame = self._engine.render_frame(self._cams[i])
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
        bloom = float(self.bloom.value())
        sat = float(self.saturation.value())
        stylize = float(self.style.value())
        haze = float(self.haze.value())
        star_count = int(self.stars.value())
        rotate_deg = float(self.rotate.value())
        pan = float(self.pan.value())
        streaks = self.streaks.isChecked()
        lo, hi = float(self.star_min.value()), float(self.star_max.value())
        star_min, star_max = min(lo, hi), max(lo, hi)
        pan_points = list(self._pan_points)
        out_w, out_h = self._frame_size(width)
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
                    zoom_end=zoom_end, rotate_deg=rotate_deg, pan_points=pan_points,
                    pan=pan, star_min=star_min, star_max=star_max,
                    streaks=streaks, on_frame=on_frame)
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
