"""LazyMoonSun panel — burst stacking & finishing for the Sun and Moon.

A sibling of the LazyStretch panel, hosted by the same shell. Picks a burst folder, exposes
the dial set + Sun/Moon/Neutral presets, and runs the global / multi-point stack and the
finish off the UI thread via :class:`CallableWorker`.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..io.image_io import load_image, save_image
from ..moonsun import run as msrun
from ..moonsun.model import MoonSunParams
from .preview import PreviewView
from .widgets import FloatSlider
from .worker import CallableWorker

# (attr, label, lo, hi, decimals)
_MS_DIALS = [
    ("keep", "Keep best", 0.05, 1.0, 2),
    ("sharp", "Sharpen", 0.0, 0.8, 2),
    ("disc_target", "Disc target", 0.5, 0.9, 2),
    ("surface", "Surface boost", 0.0, 8.0, 1),
    ("flatten", "Flatten limb", 0.0, 1.0, 2),
    ("contrast", "Contrast", 0.0, 0.5, 2),
    ("lhe", "Local contrast", 0.0, 0.5, 2),
    ("sat", "Saturation", 0.0, 1.0, 2),
    ("ht_sh", "Black point", 0.0, 0.2, 3),
    ("ht_mid", "Midtones", 0.35, 0.65, 3),
    ("ht_hi", "Highlights", 0.7, 1.0, 3),
    ("crop_margin", "Crop margin", 0.0, 0.5, 2),
    ("ap_size", "AP size", 48, 192, 0),
    ("ap_keep", "Keep per AP", 0.05, 1.0, 2),
    ("ap_search", "AP search", 4, 16, 0),
    ("ap_max", "Max APs", 50, 900, 0),
]


class LazyMoonSunPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LazyMoonSun")
        self._folder: Optional[str] = None
        self.result_image: Optional[np.ndarray] = None    # what's shown / saved
        self._source_image: Optional[np.ndarray] = None   # the pristine loaded/stacked base
        self.worker: Optional[CallableWorker] = None
        self.dials: Dict[str, FloatSlider] = {}

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_view())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 1000])
        root = QHBoxLayout(self)
        root.addWidget(splitter)
        self._apply_params(MoonSunParams.preset("sun"))

    # ------------------------------------------------------------------ build

    def _build_controls(self) -> QWidget:
        col = QVBoxLayout()

        g_src = QGroupBox("Burst folder")
        v = QVBoxLayout(g_src)
        self.folder_btn = QPushButton("Choose burst folder…")
        self.folder_btn.clicked.connect(self._choose_folder)
        self.folder_label = QLabel("(none) — one folder = one burst")
        self.folder_label.setStyleSheet("color: gray;")
        self.folder_label.setWordWrap(True)
        v.addWidget(self.folder_btn)
        v.addWidget(self.folder_label)
        col.addWidget(g_src)

        g_pre = QGroupBox("Preset")
        pv = QHBoxLayout(g_pre)
        for name in ("Sun", "Moon", "Neutral"):
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, n=name: self._apply_params(MoonSunParams.preset(n)))
            pv.addWidget(b)
        col.addWidget(g_pre)

        g_adj = QGroupBox("Dials")
        av = QVBoxLayout(g_adj)
        for attr, label, lo, hi, dec in _MS_DIALS:
            fs = FloatSlider(label, lo, hi, lo, decimals=dec)
            self.dials[attr] = fs
            av.addWidget(fs)
        self.tone_combo = QComboBox()
        self.tone_combo.addItems(["neutral", "golden"])
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Tone:"))
        trow.addWidget(self.tone_combo, 1)
        av.addLayout(trow)
        self.wb_check = QCheckBox("Neutralize filter tint")
        self.finish_check = QCheckBox("Finish after stacking")
        self.crop_check = QCheckBox("Crop to disc when finished")
        av.addWidget(self.wb_check)
        av.addWidget(self.finish_check)
        av.addWidget(self.crop_check)
        col.addWidget(g_adj)
        col.addStretch(1)

        holder = QWidget()
        holder.setLayout(col)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        scroll.setMinimumWidth(480)
        return scroll

    def _build_view(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        btns = QHBoxLayout()
        self.stack_btn = QPushButton("Stack burst (global)")
        self.mp_btn = QPushButton("Stack multi-point")
        self.finish_btn = QPushButton("Finish active")
        self.open_btn = QPushButton("Open image…")
        self.save_btn = QPushButton("Save result…")
        self.save_btn.setEnabled(False)
        self.stack_btn.clicked.connect(lambda: self._do_stack(multipoint=False))
        self.mp_btn.clicked.connect(lambda: self._do_stack(multipoint=True))
        self.finish_btn.clicked.connect(self._do_finish)
        self.open_btn.clicked.connect(self._open_image)
        self.save_btn.clicked.connect(self._save_result)
        for b in (self.stack_btn, self.mp_btn, self.finish_btn, self.open_btn, self.save_btn):
            btns.addWidget(b)
        v.addLayout(btns)

        self.preview = PreviewView()
        v.addWidget(self.preview, 7)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(600)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status_label = QLabel("Pick a burst folder, then Stack.")
        v.addWidget(self.log_view, 3)
        v.addWidget(self.progress)
        v.addWidget(self.status_label)
        return w

    # ---------------------------------------------------------------- params

    def _collect_params(self) -> MoonSunParams:
        p = MoonSunParams()
        for attr, fs in self.dials.items():
            setattr(p, attr, fs.value())
        p.tone = self.tone_combo.currentText()
        p.do_wb = self.wb_check.isChecked()
        p.auto_finish = self.finish_check.isChecked()
        p.crop = self.crop_check.isChecked()
        return p

    def _apply_params(self, p: MoonSunParams):
        for attr, fs in self.dials.items():
            fs.set_value(float(getattr(p, attr)))
        self.tone_combo.setCurrentText(p.tone)
        self.wb_check.setChecked(p.do_wb)
        self.finish_check.setChecked(p.auto_finish)
        self.crop_check.setChecked(p.crop)

    # ---------------------------------------------------------------- actions

    def _choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose burst folder")
        if d:
            self._folder = d
            scan = msrun.find_frames(d)
            note = f"{len(scan['frames'])} frames"
            if scan["raws"]:
                note += f" (+{len(scan['raws'])} raws — need rawpy)"
            self.folder_label.setText(f"{Path(d).name} — {note}")
            self.folder_label.setStyleSheet("")
            self.status_label.setText("Folder ready. Stack burst, then finish.")

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image to finish", "",
            "Images (*.fits *.fit *.fts *.tif *.tiff *.png *.xisf "
            "*.cr2 *.cr3 *.nef *.arw *.raf *.dng *.orf *.rw2 *.pef *.srw *.raw)")
        if not path:
            return
        try:
            data = load_image(path).data
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self._source_image = np.asarray(data, dtype=np.float64)   # the base to finish from
        self.result_image = self._source_image
        self.preview.set_image(self.result_image, keep_view=False)
        self.save_btn.setEnabled(True)
        self.status_label.setText(f"Loaded {Path(path).name} — Finish active to process it.")

    def _do_stack(self, *, multipoint: bool):
        if self._folder is None:
            self.status_label.setText("Pick a burst folder first.")
            return
        folder = self._folder
        params = self._collect_params()

        def fn(log, _progress):
            res = (msrun.stack_multipoint_folder(folder, params, log=log) if multipoint
                   else msrun.stack_burst_folder(folder, params, log=log))
            if res is None:
                return None
            out = dict(res)
            out["image"] = res["master"]
            if params.auto_finish:
                log("Auto-finish…")
                fimg = msrun.finish_array(res["master"], params, log=log)
                if fimg is not None:
                    out["image"] = fimg
            return out

        self._start(fn, "multi-point stack" if multipoint else "global stack")

    def _do_finish(self):
        # always finish from the pristine loaded/stacked base — never re-finish the
        # currently-shown result (that would compound the finish each click).
        if self._source_image is None:
            self.status_label.setText("Nothing to finish — stack a burst or Open an image.")
            return
        src = self._source_image
        params = self._collect_params()

        def fn(log, _progress):
            log("Finishing from the loaded/stacked base…")
            fimg = msrun.finish_array(src, params, log=log)
            return {"image": fimg} if fimg is not None else None

        self._start(fn, "finish")

    def _start(self, fn, what: str):
        if self.worker is not None and self.worker.isRunning():
            return
        self.log_view.clear()
        self._set_busy(True)
        self.progress.setRange(0, 0)
        self.status_label.setText(f"Running {what}…")
        self.worker = CallableWorker(fn, mode=what)
        self.worker.logline.connect(lambda s: self.log_view.appendPlainText(s))
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, result):
        self._set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        if not result or result.get("image") is None:
            self.status_label.setText("Done — no output (see the log).")
            return
        # a stack produces a fresh base ('master'); a finish does not — leave the base intact
        if result.get("master") is not None:
            self._source_image = np.asarray(result["master"], dtype=np.float64)
        self.result_image = np.asarray(result["image"], dtype=np.float64)
        self.preview.set_image(self.result_image, keep_view=False)
        self.save_btn.setEnabled(True)
        self.status_label.setText("Done.")

    def _on_failed(self, msg: str):
        self._set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label.setText("Failed.")
        QMessageBox.critical(self, "LazyMoonSun error", msg)

    def _set_busy(self, busy: bool):
        for b in (self.stack_btn, self.mp_btn, self.finish_btn, self.open_btn):
            b.setEnabled(not busy)

    def _save_result(self):
        if self.result_image is None:
            return
        stem = Path(self._folder).name if self._folder else "lazymoonsun"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", f"{stem}_LazyMoonSun.tif",
            "TIFF (*.tif *.tiff);;PNG (*.png);;FITS (*.fits *.fit)")
        if not path:
            return
        try:
            save_image(path, self.result_image, bit_depth=16)
            self.status_label.setText(f"Saved {Path(path).name}.")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
