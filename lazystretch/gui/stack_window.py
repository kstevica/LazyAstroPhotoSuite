"""LazyStack panel — calibrate, measure/cull, register and integrate a burst of subs.

Sibling of the LazyStretch / LazyMoonSun panels, hosted by the same shell. Picks a dataset
folder (lights/darks/flats/biases subfolders, or a lights-only folder), exposes the cull +
integration dials, and runs the measure-only advisor or the full stack off the UI thread.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..io.image_io import save_image
from ..lazystack import run as lsrun
from ..lazystack.model import LazyStackParams
from .preview import PreviewView
from .widgets import FloatSlider, RunLogView
from .worker import CallableWorker

_DIALS = [
    ("sigma_low", "Reject σ low", 1.0, 6.0, 1),
    ("sigma_high", "Reject σ high", 1.0, 6.0, 1),
    ("max_reject_frac", "Max cull frac", 0.0, 0.6, 2),
    ("ecc_hard", "Ecc reject", 0.4, 0.9, 2),
]
_CHECKS = [
    ("do_calibrate", "Calibrate (bias/dark/flat)"),
    ("do_cosmetic", "Cosmetic correction (hot/cold)"),
    ("do_register", "Register frames"),
    ("normalize", "Normalize to reference (background + scale)"),
    ("local_normalize", "Local normalization (spatially-varying gradient match)"),
    ("reuse_cache", "Reuse cached intermediates"),
    ("stage_to_disk", "Stage to disk (low memory; off = in-RAM, no work files)"),
]


class LazyStackPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LazyStack")
        self._folder: Optional[str] = None
        self.result_image: Optional[np.ndarray] = None
        self.worker: Optional[CallableWorker] = None
        self.dials: Dict[str, FloatSlider] = {}
        self.checks: Dict[str, QCheckBox] = {}

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_view())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 1000])
        root = QHBoxLayout(self)
        root.addWidget(splitter)
        self._apply_params(LazyStackParams())

    def _build_controls(self) -> QWidget:
        col = QVBoxLayout()

        g_src = QGroupBox("Dataset folder")
        v = QVBoxLayout(g_src)
        self.folder_btn = QPushButton("Choose dataset folder…")
        self.folder_btn.clicked.connect(self._choose_folder)
        self.folder_label = QLabel("(none) — lights/darks/flats/biases subfolders, "
                                   "or a folder of lights")
        self.folder_label.setStyleSheet("color: gray;")
        self.folder_label.setWordWrap(True)
        v.addWidget(self.folder_btn)
        v.addWidget(self.folder_label)
        col.addWidget(g_src)

        g_opt = QGroupBox("Options")
        ov = QVBoxLayout(g_opt)
        for attr, label in _CHECKS:
            cb = QCheckBox(label)
            self.checks[attr] = cb
            ov.addWidget(cb)
        for attr, label, lo, hi, dec in _DIALS:
            fs = FloatSlider(label, lo, hi, lo, decimals=dec)
            self.dials[attr] = fs
            ov.addWidget(fs)
        col.addWidget(g_opt)

        note = QLabel("Registration uses astroalign when installed (rotation-aware); "
                      "otherwise a translation-only fallback. For full fidelity: "
                      "pip install astroalign ccdproc astroscrappy rawpy")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        col.addWidget(note)
        col.addStretch(1)

        holder = QWidget()
        holder.setLayout(col)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        scroll.setMinimumWidth(420)
        return scroll

    def _build_view(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        btns = QHBoxLayout()
        self.measure_btn = QPushButton("Measure only")
        self.stack_btn = QPushButton("Stack")
        self.save_btn = QPushButton("Save master…")
        self.save_btn.setEnabled(False)
        self.measure_btn.clicked.connect(self._do_measure)
        self.stack_btn.clicked.connect(self._do_stack)
        self.save_btn.clicked.connect(self._save_result)
        for b in (self.measure_btn, self.stack_btn, self.save_btn):
            btns.addWidget(b)
        btns.addStretch(1)
        v.addLayout(btns)

        self.preview = PreviewView()
        v.addWidget(self.preview, 6)
        self.log_view = RunLogView()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status_label = QLabel("Pick a dataset folder, then Measure or Stack.")
        v.addWidget(self.log_view, 3)
        v.addWidget(self.progress)
        v.addWidget(self.status_label)
        return w

    # -------------------------------------------------------------- params

    def _collect_params(self) -> LazyStackParams:
        p = LazyStackParams()
        for attr, fs in self.dials.items():
            setattr(p, attr, fs.value())
        for attr, cb in self.checks.items():
            setattr(p, attr, cb.isChecked())
        return p

    def _apply_params(self, p: LazyStackParams):
        for attr, fs in self.dials.items():
            fs.set_value(float(getattr(p, attr)))
        for attr, cb in self.checks.items():
            cb.setChecked(bool(getattr(p, attr)))

    # -------------------------------------------------------------- actions

    def _choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose dataset folder")
        if not d:
            return
        self._folder = d
        sets = lsrun.find_sets(d)
        self.folder_label.setText(
            f"{Path(d).name} — {len(sets['lights'])} lights, {len(sets['darks'])} darks, "
            f"{len(sets['flats'])} flats, {len(sets['biases'])} biases")
        self.folder_label.setStyleSheet("")
        self.status_label.setText("Folder ready.")

    def _do_measure(self):
        if self._folder is None:
            self.status_label.setText("Pick a dataset folder first.")
            return
        folder, params = self._folder, self._collect_params()

        def fn(log, _p):
            res = lsrun.measure_only(folder, params, log=log)
            if res is None:
                return None
            c = res["cull"]
            log(f"Advisor: keep {len(c['keep'])}/{len(res['measures'])}, "
                f"reference frame index {c['reference']}.")
            for i, why in c.get("rejected", {}).items():
                log(f"  reject [{i}]: {why}")
            return {"image": None, "measure_only": True}

        self._start(fn, "measure")

    def _do_stack(self):
        if self._folder is None:
            self.status_label.setText("Pick a dataset folder first.")
            return
        folder, params = self._folder, self._collect_params()

        def fn(log, _p):
            res = lsrun.stack(folder, params, log=log)
            if res is None:
                return None
            out = dict(res)
            out["image"] = res["master"]
            return out

        self._start(fn, "stack")

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
        self.log_view.finish()
        self._set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        if not result:
            self.status_label.setText("Done — no output (see the log).")
            return
        if result.get("measure_only"):
            self.status_label.setText("Advisor done — see the log.")
            return
        img = result.get("image")
        if img is None:
            self.status_label.setText("Done — no master produced (see the log).")
            return
        self.result_image = np.asarray(img, dtype=np.float64)
        self.preview.set_image(self.result_image, keep_view=False)
        self.save_btn.setEnabled(True)
        via = result.get("registered_with", "?")
        self.status_label.setText(
            f"Stacked {result.get('n_stacked', '?')} frames via {via}. "
            "Open the master in LazyStretch to finish.")

    def _on_failed(self, msg: str):
        self.log_view.finish()
        self._set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label.setText("Failed.")
        QMessageBox.critical(self, "LazyStack error", msg)

    def _set_busy(self, busy: bool):
        for b in (self.measure_btn, self.stack_btn):
            b.setEnabled(not busy)

    def _save_result(self):
        if self.result_image is None:
            return
        stem = Path(self._folder).name if self._folder else "lazystack"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save master", f"{stem}_master.fits",
            "FITS (*.fits *.fit);;TIFF (*.tif *.tiff)")
        if not path:
            return
        try:
            save_image(path, self.result_image, bit_depth=16)
            self.status_label.setText(f"Saved {Path(path).name}.")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
