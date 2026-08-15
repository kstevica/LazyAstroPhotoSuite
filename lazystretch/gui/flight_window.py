"""LazyFlight panel — turn a finished still into a 3D fly-through video.

Open a stretched/developed master, dial the camera move and look, scrub a live
low-res preview, then render the full-resolution mp4 off the UI thread. The
preview always runs the fast ``parallax`` engine for responsiveness; the final
render honours the chosen mode (``parallax`` or the slower ``volumetric`` glow).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from ..animate import render_flythrough
from ..animate.clip import PATHS, _resize, build_cameras
from ..animate.encode import ffmpeg_available
from ..animate.render import Flythrough3D
from ..io.image_io import load_image
from .preview import PreviewView
from .widgets import FloatSlider
from .worker import CallableWorker

_PREVIEW_W = 760          # live-preview render width (fast)
_PREVIEW_FPS = 24         # frames the scrub bar addresses


class LazyFlightPanel(QWidget):
    """Compose a fly-through from a still and render it to mp4."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.img: Optional[np.ndarray] = None          # full-res float RGB
        self._engine: Optional[Flythrough3D] = None    # preview engine (low-res)
        self._cams = []                                # preview camera path
        self._masks = None                             # semantic masks (optional)
        self.worker: Optional[CallableWorker] = None
        self._busy = False
        self._play = QTimer(self)
        self._play.setInterval(int(1000 / _PREVIEW_FPS))
        self._play.timeout.connect(self._advance_play)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(self._render_preview)

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
        self.path_combo.addItems(list(PATHS.keys()))
        self.path_combo.currentIndexChanged.connect(self._rebuild_cams)
        cam.addWidget(self._row("Path", self.path_combo))
        self.dur = FloatSlider("Duration (s)", 3.0, 30.0, 8.0, decimals=0)
        self.dur.valueChanged.connect(self._rebuild_cams)
        cam.addWidget(self.dur)
        self.zoom = FloatSlider("Zoom / 3D", 1.05, 1.8, 1.35, decimals=2)
        self.zoom.valueChanged.connect(self._rebuild_cams)
        cam.addWidget(self.zoom)
        side.addWidget(cam_box)

        look_box = QGroupBox("Look")
        look = QVBoxLayout(look_box)
        self.bloom = FloatSlider("Bloom", 0.0, 1.0, 0.45, decimals=2)
        self.bloom.valueChanged.connect(self._on_bloom)
        look.addWidget(self.bloom)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["parallax (fast)", "volumetric (glow)"])
        look.addWidget(self._row("Render mode", self.mode_combo))
        self.semantic = QCheckBox("Use semantic depth (slower open)")
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
        out.addWidget(self._row("Width", self.width_combo))
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
        self.canvas = PreviewView()
        right.addWidget(self.canvas, 1)
        bar = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._toggle_play)
        bar.addWidget(self.play_btn)
        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, 100)
        self.scrub.valueChanged.connect(lambda _v: self._debounce.start())
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
        for w in (self.path_combo, self.dur, self.zoom, self.bloom,
                  self.mode_combo, self.semantic, self.show_depth,
                  self.fps_combo, self.width_combo, self.render_btn,
                  self.play_btn, self.scrub):
            w.setEnabled(on)

    def _set_busy(self, busy: bool):
        self._busy = busy
        self.open_btn.setEnabled(not busy)
        self._set_controls_enabled(not busy and self.img is not None)

    def _mode(self) -> str:
        return "volumetric" if self.mode_combo.currentIndex() == 1 else "parallax"

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
        if self.img is None:
            return
        self.status.setText("Preparing preview…")
        QWidget.repaint(self)
        small = _resize(self.img, _PREVIEW_W)
        if self.semantic.isChecked():
            from ..develop.semantic import segment
            self._masks = segment(small)
        else:
            self._masks = None
        self._engine = Flythrough3D(small, masks=self._masks,
                                    bloom=self.bloom.value(), mode="parallax")
        self.status.setText("")
        self._rebuild_cams()

    def _rebuild_cams(self):
        if self._engine is None:
            return
        n = max(int(round(self.dur.value() * _PREVIEW_FPS)), 2)
        try:
            self._cams = build_cameras(self.path_combo.currentText(), n,
                                       zoom_end=float(self.zoom.value()))
        except TypeError:                               # path without zoom_end
            self._cams = build_cameras(self.path_combo.currentText(), n)
        self._render_preview()

    def _on_bloom(self, val: float):
        if self._engine is not None:
            self._engine.bloom_strength = float(val)
        self._render_preview()

    # --------------------------------------------------------------- preview
    def _render_preview(self):
        if self._engine is None or self._busy:
            return
        if self.show_depth.isChecked():
            z = self._engine.depth.astype(np.float32)
            self.canvas.set_image(np.repeat(z[..., None], 3, axis=2), keep_view=True)
            return
        if not self._cams:
            return
        i = int(round(self.scrub.value() / 100.0 * (len(self._cams) - 1)))
        frame = self._engine.render_frame(self._cams[i])
        self.canvas.set_image(frame, keep_view=True)

    def _toggle_play(self, on: bool):
        self.play_btn.setText("❚❚ Pause" if on else "▶ Play")
        if on and not self._busy:
            self._play.start()
        else:
            self._play.stop()

    def _advance_play(self):
        v = self.scrub.value() + 1
        if v > 100:
            v = 0
        self.scrub.setValue(v)                          # triggers debounced preview

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
        masks = self._masks if self.semantic.isChecked() else None
        seconds = float(self.dur.value())
        fps = int(self.fps_combo.currentText())
        path = self.path_combo.currentText()
        width = int(self.width_combo.currentText())
        mode = self._mode()
        zoom_end = float(self.zoom.value())
        bloom = float(self.bloom.value())

        def job(log, progress):
            if masks is not None:                       # match render res for depth
                from ..develop.semantic import segment
                m = segment(_resize(img, width))
            else:
                m = None
            def on_frame(i, n, _fr):
                progress(i + 1, n, "frame")
            return render_flythrough(
                img, out, seconds=seconds, fps=fps, path=path, mode=mode,
                render_width=width, masks=m, engine_kw={"bloom": bloom},
                path_kw={"zoom_end": zoom_end}, on_frame=on_frame)

        self._set_busy(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)                    # indeterminate until first frame
        self.status.setText(f"Rendering {mode} @ {width}px…")
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
