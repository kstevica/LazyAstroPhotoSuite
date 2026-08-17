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
    QButtonGroup,
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
    ("demaze", "Remove X-Trans demosaic mesh (6×6 grid; Fuji, optional)"),
    ("do_register", "Register frames"),
    ("normalize", "Normalize to reference (background + scale)"),
    ("local_normalize", "Local normalization (spatially-varying gradient match)"),
    ("edge_crop", "Crop to common overlap (remove ragged registration edges)"),
    ("emit_snr_map", "Write noise/SNR map (feeds the stretch's SNR-protect mask)"),
    ("preserve_meteors", "Preserve meteor trails (detect + save a meteor layer for the stretch)"),
    ("amplified", "EXPERIMENTAL — Amplified signal (keep soft frames via frequency-split "
     "weights, e\u207b noise model, pattern removal, 2\u00d7 fine grid when undersampled)"),
    ("reuse_cache", "Reuse cached intermediates"),
    ("stage_to_disk", "Stage to disk (low memory; off = in-RAM, no work files)"),
]
# The nightscape window shows only the options that matter for a fixed-tripod sky stack.
# Calibration/cosmetic/walking-noise are omitted (rarely darks/flats, and a moonlit star
# reads as a hot pixel); registration, normalization and crop are what count.
_NIGHTSCAPE_CHECKS = [
    ("do_register", "Register on sky stars"),
    ("normalize", "Normalize to reference (background + scale)"),
    ("local_normalize", "Local normalization (spatially-varying gradient match)"),
    ("edge_crop", "Crop to common overlap (remove ragged registration edges)"),
    ("do_calibrate", "Calibrate (bias/dark/flat — only if you shot them)"),
    ("reuse_cache", "Reuse cached intermediates"),
    ("stage_to_disk", "Stage to disk (low memory; off = in-RAM, no work files)"),
]


class LazyStackPanel(QWidget):
    def __init__(self, nightscape: bool = False):
        super().__init__()
        self._nightscape = bool(nightscape)
        self.setWindowTitle("LazyNightscape" if self._nightscape else "LazyStack")
        self._folder: Optional[str] = None
        self._nightscape_fg_path: Optional[str] = None
        self._nightscape_manual_mask: Optional[np.ndarray] = None  # painted mask (preview-res)
        self._seg_median: Optional[np.ndarray] = None              # last segmentation-preview median
        self._seg_auto: Optional[np.ndarray] = None                # last auto sky mask (for watershed seeds)
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
        if self._nightscape:                                 # nightscape defaults: no calibration
            for attr in ("do_calibrate",):
                if attr in self.checks:
                    self.checks[attr].setChecked(False)

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
        for attr, label in (_NIGHTSCAPE_CHECKS if self._nightscape else _CHECKS):
            cb = QCheckBox(label)
            self.checks[attr] = cb
            ov.addWidget(cb)
        for attr, label, lo, hi, dec in _DIALS:
            fs = FloatSlider(label, lo, hi, lo, decimals=dec)
            self.dials[attr] = fs
            ov.addWidget(fs)
        col.addWidget(g_opt)

        if self._nightscape:
            col.addWidget(self._build_nightscape_group())

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

    def _build_nightscape_group(self) -> QWidget:
        # Foreground-locked Milky Way: this whole window IS nightscape mode (no on/off toggle).
        g_ns = QGroupBox("Foreground / sky segmentation")
        nv = QVBoxLayout(g_ns)
        fgrow = QHBoxLayout()
        self.ns_fg_btn = QPushButton("Foreground image…")
        self.ns_fg_btn.clicked.connect(self._choose_foreground)
        self.ns_fg_label = QLabel("auto: sharpest frame")
        self.ns_fg_label.setStyleSheet("color: gray;")
        fgrow.addWidget(self.ns_fg_btn)
        fgrow.addWidget(self.ns_fg_label, 1)
        nv.addLayout(fgrow)
        self.ns_bias = FloatSlider("Sky ↔ foreground bias", -0.10, 0.10, 0.0, decimals=3)
        nv.addWidget(self.ns_bias)
        self.ns_preview_btn = QPushButton("Preview segmentation")
        self.ns_preview_btn.clicked.connect(self._preview_segmentation)
        nv.addWidget(self.ns_preview_btn)
        # paint tools (rough scribbles → watershed snaps to the horizon)
        paintrow = QHBoxLayout()
        self.paint_group = QButtonGroup(self)
        self.paint_btns: Dict[object, QPushButton] = {}
        for mode, label in ((None, "Pan"), ("sky", "Paint Sky"), ("earth", "Paint Earth"), ("erase", "Erase")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, m=mode: self.preview.set_paint_mode(m))
            self.paint_group.addButton(b)
            paintrow.addWidget(b)
            self.paint_btns[mode] = b
        self.paint_btns[None].setChecked(True)
        nv.addLayout(paintrow)
        self.brush_slider = FloatSlider("Brush size", 4, 120, 14, decimals=0)
        self.brush_slider.valueChanged.connect(lambda v: self.preview.set_brush(int(v)))
        nv.addWidget(self.brush_slider)
        self.brush_falloff = FloatSlider("Brush falloff", 0.0, 1.0, 0.5, decimals=2)
        self.brush_falloff.valueChanged.connect(lambda v: self.preview.set_falloff(v))
        nv.addWidget(self.brush_falloff)
        self.brush_strength = FloatSlider("Brush strength", 0.1, 1.0, 1.0, decimals=2)
        self.brush_strength.valueChanged.connect(lambda v: self.preview.set_strength(v))
        nv.addWidget(self.brush_strength)
        self.fill_opposite_btn = QPushButton("Fill rest (opposite)")
        self.fill_opposite_btn.clicked.connect(self._fill_opposite)
        nv.addWidget(self.fill_opposite_btn)
        applyrow = QHBoxLayout()
        self.paint_apply_btn = QPushButton("Apply paint")
        self.paint_clear_btn = QPushButton("Clear paint")
        self.paint_apply_btn.clicked.connect(self._apply_paint)
        self.paint_clear_btn.clicked.connect(self._clear_paint)
        applyrow.addWidget(self.paint_apply_btn)
        applyrow.addWidget(self.paint_clear_btn)
        nv.addLayout(applyrow)
        self.ns_mask_status = QLabel("Mask: auto — adjust with the bias slider, or paint to refine")
        self.ns_mask_status.setStyleSheet("color: gray;")
        self.ns_mask_status.setWordWrap(True)
        nv.addWidget(self.ns_mask_status)
        ns_note = QLabel("Fixed-tripod MW + landscape. Registers on sky stars only; the sharp "
                         "foreground (or your own image) is composited over the deep sky in LazyStretch. "
                         "Preview shows the sky/foreground split — adjust the bias to refine.")
        ns_note.setWordWrap(True)
        ns_note.setStyleSheet("color: gray; font-size: 11px;")
        nv.addWidget(ns_note)
        return g_ns

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
        if self._nightscape:
            p.nightscape = True
            p.fix_walking_noise = False          # a moonlit star reads as a static hot pixel
            p.fix_banding = False
            p.nightscape_foreground = self._nightscape_fg_path or ""
            p.nightscape_bias = float(self.ns_bias.value())
            p.nightscape_mask_override = self._nightscape_manual_mask   # painted mask wins over auto
        return p

    def _apply_paint(self):
        if self._seg_median is None:
            self.status_label.setText("Run Preview segmentation first, then paint and Apply.")
            return
        scr = self.preview.scribbles()
        if scr is None or not np.any(scr):
            self.status_label.setText("Paint some Sky and Earth strokes on the preview first.")
            return
        from ..lazystack import nightscape as ns
        mask = ns.refine_mask(self._seg_median, scr, auto_mask=self._seg_auto)
        self._nightscape_manual_mask = np.asarray(mask, dtype=np.float32)
        med = self._seg_median
        base = np.clip(np.arcsinh(np.clip(med, 0, 1) * 25.0) / np.arcsinh(25.0) * 1.3, 0, 1)
        ov = np.stack([base] * 3, axis=-1)
        ov[..., 0] = np.clip(ov[..., 0] + (1.0 - mask) * 0.4, 0, 1)
        self.preview.set_image(ov, keep_view=True)               # show refined mask, keep strokes + view
        fg = 100 * (1 - float(mask.mean()))
        self.ns_mask_status.setText(f"✓ Mask: PAINTED — will be used for the stack (foreground {fg:.0f}%)")
        self.ns_mask_status.setStyleSheet("color: #1b8a3a; font-weight: bold;")
        self.status_label.setText(f"Painted mask applied (foreground {fg:.0f}%). Stack to use it.")

    def _clear_paint(self):
        self.preview.clear_scribbles()
        self._nightscape_manual_mask = None
        self.ns_mask_status.setText("Mask: auto — adjust with the bias slider, or paint to refine")
        self.ns_mask_status.setStyleSheet("color: gray;")
        self.status_label.setText("Paint cleared — auto/bias segmentation will be used.")

    def _fill_opposite(self):
        if self.preview.fill_opposite():
            self.status_label.setText("Filled the unpainted area with the opposite class — Apply paint to use it.")
        else:
            self.status_label.setText("Paint ONE class first (only Sky or only Earth), then Fill rest (opposite).")

    def _choose_foreground(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose foreground image (optional)", self._folder or "",
            "Images (*.fits *.fit *.tif *.tiff *.png *.jpg *.jpeg *.raf *.cr2 *.cr3 *.nef *.arw *.dng)")
        if path:
            self._nightscape_fg_path = path
            self.ns_fg_label.setText(f"user: {Path(path).name}")
            self.ns_fg_label.setStyleSheet("")

    def _preview_segmentation(self):
        if self._folder is None:
            self.status_label.setText("Pick a dataset folder first.")
            return
        folder, bias, fg_path = self._folder, float(self.ns_bias.value()), self._nightscape_fg_path

        def fn(log, _p):
            from ..io.image_io import load_preview
            from ..lazystack import nightscape as ns
            lights = lsrun.find_sets(folder)["lights"]
            if len(lights) < 2:
                log("Need at least 2 lights to segment.")
                return None
            step = max(1, len(lights) // 4)
            sample = lights[::step][:4]                           # a few frames suffice for the horizon
            log(f"Segmenting from {len(sample)} half-size frames (bias {bias:+.2f})…")
            cube = np.stack([np.asarray(load_preview(p, max_dim=1000), dtype=np.float32) for p in sample])
            if fg_path:                                          # Mode 2: segment (and paint on) the user foreground
                ufg = np.asarray(load_preview(fg_path, max_dim=1000), dtype=np.float32)
                mask, info = ns.segment_sky(ufg, bias=bias)
                med = ufg.mean(2) if ufg.ndim == 3 else ufg
            else:
                mask, info = ns.segment_sky(cube, bias=bias)
                med = np.median(cube.mean(3), axis=0)
            base = np.clip(np.arcsinh(np.clip(med, 0, 1) * 25.0) / np.arcsinh(25.0) * 1.3, 0, 1)
            ov = np.stack([base] * 3, axis=-1)
            ov[..., 0] = np.clip(ov[..., 0] + (1.0 - mask) * 0.4, 0, 1)   # red-tint the foreground
            log(f"Sky/foreground split: {info['axis']} horizon, sky on the {info['sky_side']}, "
                f"foreground {100 * info['foreground_fraction']:.0f}%.")
            return {"image": ov, "segmentation_preview": True, "median": med, "auto_mask": mask}

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
            self._seg_median = np.asarray(result["median"], dtype=np.float32)
            self._seg_auto = np.asarray(result["auto_mask"], dtype=np.float32)
            self._nightscape_manual_mask = None      # a fresh segmentation clears any prior paint
            self.ns_mask_status.setText("Mask: auto — adjust with the bias slider, or paint to refine")
            self.ns_mask_status.setStyleSheet("color: gray;")
            self.preview.set_image(np.asarray(result["image"], dtype=np.float64), keep_view=False)
            self.status_label.setText("Segmentation preview (red = foreground). Paint Sky/Earth to "
                                      "refine, then Apply — or adjust the bias and re-preview.")
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
        btns = [self.measure_btn, self.stack_btn]
        if hasattr(self, "ns_preview_btn"):
            btns.append(self.ns_preview_btn)
        for b in btns:
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
