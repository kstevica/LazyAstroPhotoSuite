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
    ("fix_walking_noise", "Fix walking noise (static hot-pixel repair; no darks/dither needed)"),
    ("fix_banding", "Suppress column/row banding (fixed-pattern noise; no darks needed)"),
    ("do_register", "Register frames"),
    ("normalize", "Normalize to reference (background + scale)"),
    ("local_normalize", "Local normalization (spatially-varying gradient match)"),
    ("edge_crop", "Crop to common overlap (remove ragged registration edges)"),
    ("emit_snr_map", "Write noise/SNR map (feeds the stretch's SNR-protect mask)"),
    ("preserve_meteors", "Preserve meteor trails (detect + save a meteor layer for the stretch)"),
    ("reuse_cache", "Reuse cached intermediates"),
    ("stage_to_disk", "Stage to disk (low memory; off = in-RAM, no work files)"),
]


class LazyStackPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LazyStack")
        self._folder: Optional[str] = None
        self._nightscape_fg_path: Optional[str] = None
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

        # --- Nightscape (foreground-locked Milky Way) ---
        g_ns = QGroupBox("Nightscape (foreground-locked MW)")
        nv = QVBoxLayout(g_ns)
        self.ns_check = QCheckBox("Nightscape mode — stack the sky, keep a sharp foreground")
        self.ns_check.toggled.connect(self._on_nightscape_toggled)
        nv.addWidget(self.ns_check)
        fgrow = QHBoxLayout()
        self.ns_fg_btn = QPushButton("Foreground image…")
        self.ns_fg_btn.clicked.connect(self._choose_foreground)
        self.ns_fg_label = QLabel("auto: sharpest frame")
        self.ns_fg_label.setStyleSheet("color: gray;")
        fgrow.addWidget(self.ns_fg_btn)
        fgrow.addWidget(self.ns_fg_label, 1)
        nv.addLayout(fgrow)
        self.ns_bias = FloatSlider("Sky ↔ foreground bias", -0.30, 0.30, 0.0, decimals=2)
        nv.addWidget(self.ns_bias)
        self.ns_preview_btn = QPushButton("Preview segmentation")
        self.ns_preview_btn.clicked.connect(self._preview_segmentation)
        nv.addWidget(self.ns_preview_btn)
        ns_note = QLabel("Fixed-tripod MW + landscape. Registers on sky stars only; the sharp "
                         "foreground (or your own image) is composited over the deep sky in LazyStretch. "
                         "Preview shows the sky/foreground split — adjust the bias to refine.")
        ns_note.setWordWrap(True)
        ns_note.setStyleSheet("color: gray; font-size: 11px;")
        nv.addWidget(ns_note)
        col.addWidget(g_ns)

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
        p.nightscape = self.ns_check.isChecked()
        p.nightscape_foreground = self._nightscape_fg_path or ""
        p.nightscape_bias = float(self.ns_bias.value())
        return p

    def _on_nightscape_toggled(self, on: bool):
        # A moonlit star can look like a static hot pixel, and there are rarely darks/flats — so
        # calibration/cosmetic/walking-noise are off by default for a nightscape (re-enable if wanted).
        if on:
            for attr in ("do_calibrate", "do_cosmetic", "fix_walking_noise"):
                if attr in self.checks:
                    self.checks[attr].setChecked(False)
            self.status_label.setText("Nightscape mode on — Preview segmentation, then Stack.")

    def _choose_foreground(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose foreground image (optional)", self._folder or "",
            "Images (*.fits *.fit *.tif *.tiff *.png *.jpg *.jpeg *.raf *.cr2 *.cr3 *.nef *.arw *.dng)")
        if path:
            self._nightscape_fg_path = path
            self.ns_fg_label.setText(f"user: {Path(path).name}")
            self.ns_fg_label.setStyleSheet("")
            self.ns_check.setChecked(True)

    def _preview_segmentation(self):
        if self._folder is None:
            self.status_label.setText("Pick a dataset folder first.")
            return
        folder, bias, fg_path = self._folder, float(self.ns_bias.value()), self._nightscape_fg_path

        def fn(log, _p):
            from ..io.image_io import load_image
            from ..lazystack import nightscape as ns
            lights = lsrun.find_sets(folder)["lights"]
            if len(lights) < 2:
                log("Need at least 2 lights to segment.")
                return None
            step = max(1, len(lights) // 6)
            sample = lights[::step][:8]
            log(f"Segmenting from {len(sample)} sampled frames (bias {bias:+.2f})…")
            cube = np.stack([np.asarray(load_image(p).data, dtype=np.float32)[::6, ::6] for p in sample])
            ufg = (np.asarray(load_image(fg_path).data, dtype=np.float32)[::6, ::6]
                   if fg_path else None)
            _, mask, info = ns.plan(cube, cube.shape[1:3], user_fg_small=ufg, bias=bias)
            med = np.median(cube.mean(3), axis=0)
            base = np.clip(np.arcsinh(np.clip(med, 0, 1) * 25.0) / np.arcsinh(25.0) * 1.3, 0, 1)
            ov = np.stack([base] * 3, axis=-1)
            ov[..., 0] = np.clip(ov[..., 0] + (1.0 - mask) * 0.4, 0, 1)   # red-tint the foreground
            log(f"Sky/foreground split: {info['axis']} horizon, sky on the {info['sky_side']}, "
                f"foreground {100 * info['foreground_fraction']:.0f}%.")
            return {"image": ov, "segmentation_preview": True}

        self._start(fn, "segment")

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
        if result.get("segmentation_preview"):
            self.preview.set_image(np.asarray(result["image"], dtype=np.float64), keep_view=False)
            self.status_label.setText("Segmentation preview (red = foreground). Adjust the bias and "
                                      "re-preview, or Stack.")
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
        for b in (self.measure_btn, self.stack_btn, self.ns_preview_btn):
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
