"""The LazyStretch desktop main window (PySide6) — wraps the headless core."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
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

from ..data.loader import get_data
from ..external import Tools
from ..identify.catalog import get_catalog, object_label
from ..identify.classify import (
    detect_data_type,
    object_class as classify_object,
    palettes_for,
)
from ..identify.solver import solve
from ..io.image_io import LoadedImage, load_image, save_image
from ..io.recipes import apply_recipe, load_recipe, save_recipe
from ..objects.model import Parameters
from ..objects.presets import curated_for
from .preview import PreviewView
from .widgets import FilePicker, FloatSlider
from .worker import PipelineWorker

# (attribute, label) for the option checkboxes. The first six are profile-driven.
_PROFILE_TOGGLES = ["doSCNR", "doLocalContrast", "doBXT", "doNR", "doHDR", "doStarReduce"]
_OPTIONS = [
    ("doBgExtract", "Background / gradient extraction"),
    ("useGradientCorrection", "…use GradientCorrection (stronger)"),
    ("darkLaneGC", "…dark-lane gradient (no empty sky)"),
    ("doColorCal", "Color calibration"),
    ("preferSPCC", "…prefer SPCC (needs a solve)"),
    ("doBXT", "Deconvolution / sharpening (BlurX)"),
    ("useClassicalDeconv", "…classical RL deconvolution (no BlurX)"),
    ("doNR", "Noise reduction"),
    ("useNRMask", "…background-only (masked)"),
    ("doSCNR", "Remove green (SCNR)"),
    ("doHDR", "HDR core compression"),
    ("doStarReduce", "Star reduction"),
    ("removeStars", "Remove stars (starless + stars layer)"),
    ("useMask", "Protect faint signal with mask"),
    ("adaptiveFloor", "Adaptive shadow floor"),
    ("starProtect", "Star-protected saturation"),
    ("doLocalContrast", "Local contrast (LHE)"),
    ("protectCores", "Protect bright cores"),
    ("enhanceEmission", "Enhance emission (Ha)"),
    ("reduceCast", "Reduce background color cast"),
    ("inputStretched", "Input already stretched (polish only)"),
]

# The adjustment dials: (attr, label, lo, hi, decimals, default). The ± nudge dials default
# to 0 (class-neutral); "Highlights" defaults 0.20 (dialed down) — LOW dials bright cores /
# blown stars down and reveals core structure, HIGH leaves highlights bright (0 = dimmest,
# 1 = untouched, photographic-slider convention).
_DIALS = [
    ("satAdj", "Saturation ±", -0.5, 0.5, 2, 0.0),
    ("brightAdj", "Brightness ±", -0.10, 0.10, 3, 0.0),
    ("bgAdj", "Background ±", -0.25, 0.15, 2, 0.0),
    ("blackAdj", "Black point ±", -0.5, 0.5, 2, 0.0),
    ("contrastAdj", "Contrast ±", -0.15, 0.15, 2, 0.0),
    ("gradientCleanup", "Gradient cleanup", 0.0, 1.0, 2, 0.0),
    ("dehaze", "Dehaze", 0.0, 1.0, 2, 0.0),
    ("smallStars", "Small stars", 0.0, 0.5, 2, 0.0),
    ("starsAdj", "Stars ±", -0.3, 0.5, 2, 0.0),
    ("chromaNR", "Chroma NR", 0.0, 1.0, 2, 0.0),
    ("deepen", "Deepen (2nd pass)", 0.0, 1.0, 2, 0.0),
    ("highlights", "Highlights (0 dim → 1 bright)", 0.0, 1.0, 2, 0.20),
]


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LazyStretch")
        self.data = get_data()
        self.tools = Tools.resolve()
        self.loaded: Optional[LoadedImage] = None
        self.result_image: Optional[np.ndarray] = None
        self.result_stars: Optional[np.ndarray] = None
        self.solve_result = None
        self._recommendations: list = []
        self.worker: Optional[PipelineWorker] = None

        self.checks: Dict[str, QCheckBox] = {}
        self.dials: Dict[str, FloatSlider] = {}

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([760, 1000])

        root = QHBoxLayout(self)
        root.addWidget(splitter)
        self.resize(1760, 1060)
        self._apply_class_profile(self.class_combo.currentText())

    # ---------------------------------------------------------------- left

    def _build_left(self) -> QWidget:
        # Two columns: [Target/Identify/Output/Narrowband/Recipe] | [Adjustments/Options].
        col1 = QVBoxLayout()
        col2 = QVBoxLayout()

        # --- column 1 ---
        g_target = QGroupBox("Target")
        v = QVBoxLayout(g_target)
        self.target_picker = FilePicker("Open master…")
        self.target_picker.fileChosen.connect(self._load_image)
        v.addWidget(self.target_picker)
        orient = QHBoxLayout()
        for label, op, tip in [("↺", "rot_ccw", "Rotate 90° left"),
                               ("↻", "rot_cw", "Rotate 90° right"),
                               ("⇆", "flip_h", "Flip horizontal"),
                               ("⇅", "flip_v", "Flip vertical")]:
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setMaximumWidth(44)
            b.clicked.connect(lambda _=False, o=op: self._orient(o))
            orient.addWidget(b)
        v.addLayout(orient)
        col1.addWidget(g_target)

        g_id = QGroupBox("Identify")
        v = QVBoxLayout(g_id)
        self.identify_btn = QPushButton("Identify Target")
        self.identify_btn.clicked.connect(self._identify)
        self.analyze_btn = QPushButton("Analyze Frame")
        self.analyze_btn.clicked.connect(self._analyze)
        self.apply_reco_btn = QPushButton("Apply recommendations")
        self.apply_reco_btn.setEnabled(False)
        self.apply_reco_btn.clicked.connect(self._apply_recommendations)
        self.object_combo = QComboBox()
        self.object_combo.setEditable(True)
        v.addWidget(self.identify_btn)
        v.addWidget(self.analyze_btn)
        v.addWidget(self.apply_reco_btn)
        v.addWidget(QLabel("Object:"))
        v.addWidget(self.object_combo)
        col1.addWidget(g_id)

        g_out = QGroupBox("Output")
        grid = QGridLayout(g_out)
        self.type_label = QLabel("(load an image)")
        self.class_combo = QComboBox()
        self.class_combo.addItems(list(self.data.class_list))
        self.class_combo.setCurrentText("generic")
        self.class_combo.currentTextChanged.connect(self._apply_class_profile)
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(palettes_for("auto"))
        self.crop_slider = FloatSlider("Crop %", 0.0, 10.0, 3.0, decimals=1)
        self.auto_check = QCheckBox("Auto crop % + gradient tool")
        self.auto_check.setChecked(True)
        grid.addWidget(QLabel("Data type:"), 0, 0)
        grid.addWidget(self.type_label, 0, 1)
        grid.addWidget(QLabel("Object class:"), 1, 0)
        grid.addWidget(self.class_combo, 1, 1)
        grid.addWidget(QLabel("Palette:"), 2, 0)
        grid.addWidget(self.palette_combo, 2, 1)
        grid.addWidget(self.crop_slider, 3, 0, 1, 2)
        grid.addWidget(self.auto_check, 4, 0, 1, 2)
        col1.addWidget(g_out)

        g_nb = QGroupBox("Narrowband channels")
        grid = QGridLayout(g_nb)
        self.ha_picker = FilePicker("Ha…")
        self.oiii_picker = FilePicker("OIII…")
        self.sii_picker = FilePicker("SII…")
        for i, (lbl, pk) in enumerate([("Ha", self.ha_picker), ("OIII", self.oiii_picker),
                                       ("SII", self.sii_picker)]):
            grid.addWidget(QLabel(lbl), i, 0)
            grid.addWidget(pk, i, 1)
        col1.addWidget(g_nb)

        g_rec = QGroupBox("Recipe")
        rv = QHBoxLayout(g_rec)
        save_rec = QPushButton("Save recipe…")
        load_rec = QPushButton("Load recipe…")
        save_rec.clicked.connect(self._save_recipe)
        load_rec.clicked.connect(self._load_recipe)
        rv.addWidget(save_rec)
        rv.addWidget(load_rec)
        col1.addWidget(g_rec)
        col1.addStretch(1)

        # --- column 2 ---
        g_adj = QGroupBox("Adjustments")
        v = QVBoxLayout(g_adj)
        self._dial_defaults: Dict[str, float] = {}
        for attr, label, lo, hi, dec, dflt in _DIALS:
            fs = FloatSlider(label, lo, hi, dflt, decimals=dec)
            self.dials[attr] = fs
            self._dial_defaults[attr] = dflt
            v.addWidget(fs)
        reset = QPushButton("Reset to class defaults")
        reset.clicked.connect(self._reset_dials)
        v.addWidget(reset)
        col2.addWidget(g_adj)

        g_opt = QGroupBox("Options")
        ov = QVBoxLayout(g_opt)
        defaults = Parameters()
        for attr, label in _OPTIONS:
            cb = QCheckBox(label)
            cb.setChecked(bool(getattr(defaults, attr)))
            self.checks[attr] = cb
            ov.addWidget(cb)
        col2.addWidget(g_opt)
        col2.addStretch(1)

        cols = QHBoxLayout()
        w1, w2 = QWidget(), QWidget()
        w1.setLayout(col1)
        w2.setLayout(col2)
        cols.addWidget(w1)
        cols.addWidget(w2)
        holder = QWidget()
        holder.setLayout(cols)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        scroll.setMinimumWidth(720)
        return scroll

    # --------------------------------------------------------------- right

    def _build_right(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.preview = PreviewView()
        v.addWidget(self.preview, 7)

        info = QWidget()
        iv = QVBoxLayout(info)
        self.detected_label = QLabel("Open a linear master, pick a class, then Preview.")
        self.detected_label.setWordWrap(True)
        active = [n.split(" (")[0] for n, ok in self.tools.status().items() if ok]
        self.tools_label = QLabel("External tools: " + (", ".join(active) if active
                                  else "none detected (GraXpert→MMT, StarNet→skip, SPCC→BN+CC)"))
        self.tools_label.setStyleSheet("color: gray;")
        self.tools_label.setWordWrap(True)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status_label = QLabel("Ready.")
        iv.addWidget(self.detected_label)
        iv.addWidget(self.tools_label)
        iv.addWidget(self.log_view, 1)
        iv.addWidget(self.progress)
        iv.addWidget(self.status_label)

        btns = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.execute_btn = QPushButton("Execute")
        self.save_btn = QPushButton("Save Result…")
        self.save_btn.setEnabled(False)
        self.close_btn = QPushButton("Close")
        self.preview_btn.clicked.connect(lambda: self._run(preview=True, mode="preview"))
        self.execute_btn.clicked.connect(lambda: self._run(preview=False, mode="execute"))
        self.save_btn.clicked.connect(self._save_result)
        self.close_btn.clicked.connect(self.close)
        for b in (self.preview_btn, self.execute_btn, self.save_btn, self.close_btn):
            btns.addWidget(b)
        iv.addLayout(btns)

        v.addWidget(info, 3)
        return w

    # ------------------------------------------------------------- actions

    def _apply_class_profile(self, cls: str):
        """Class selection overwrites the six profile-driven toggles (PI behavior)."""
        prof = self.data.profile_for(cls)
        mapping = {"doSCNR": prof.scnr, "doLocalContrast": prof.localContrast,
                   "doBXT": prof.bxt, "doNR": prof.nr, "doHDR": prof.hdr,
                   "doStarReduce": prof.starReduce}
        for attr in _PROFILE_TOGGLES:
            cb = self.checks[attr]
            cb.blockSignals(True)
            cb.setChecked(bool(mapping[attr]))
            cb.blockSignals(False)

    def _reset_dials(self):
        for attr, fs in self.dials.items():
            fs.set_value(self._dial_defaults.get(attr, 0.0))
        self.crop_slider.set_value(3.0)

    def _orient(self, op: str):
        """Rotate/flip the loaded master in place (so the processed output is oriented)."""
        if self.loaded is None:
            self.status_label.setText("Load an image first.")
            return
        d = self.loaded.data
        if op == "rot_ccw":
            d = np.rot90(d, 1)
        elif op == "rot_cw":
            d = np.rot90(d, -1)
        elif op == "flip_h":
            d = d[:, ::-1]
        elif op == "flip_v":
            d = d[::-1, :]
        self.loaded.data = np.ascontiguousarray(d)
        self._show_quick_display()
        self.status_label.setText(f"Applied {op.replace('_', ' ')} ({self.loaded.data.shape[1]}×{self.loaded.data.shape[0]}).")

    def _load_image(self, path: str):
        try:
            self.loaded = load_image(path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        dtype = detect_data_type(self.loaded.keyword("FILTER"), self.loaded.is_color,
                                 self.loaded.keyword("BAYERPAT"))
        self.type_label.setText(dtype)
        self.palette_combo.clear()
        self.palette_combo.addItems(palettes_for(dtype))
        self.detected_label.setText(
            f"Loaded {Path(path).name}  —  {self.loaded.data.shape}, "
            f"{'color' if self.loaded.is_color else 'mono'}, type '{dtype}'.")
        self.status_label.setText("Loaded. Showing a quick display-stretch…")
        self._show_quick_display()

    def _show_quick_display(self):
        """Fast downsampled auto-stretch just so the loaded master is visible."""
        a = self.loaded.data
        step = max(1, max(a.shape[0], a.shape[1]) // 1400)
        small = a[::step, ::step]
        try:
            from ..stretch.autostretch import apply_auto_stretch
            disp, _ = apply_auto_stretch(small, -1.25, 0.25)
            self.preview.set_image(disp)
            self.status_label.setText("Ready. Click Preview for the full look.")
        except Exception:
            self.preview.set_image(small)

    def _identify(self):
        if self.loaded is None:
            self.status_label.setText("Load an image first.")
            return
        self.status_label.setText("Solving…")
        res = solve(self.loaded)
        self.solve_result = res
        if not res.solved:
            self.detected_label.setText("Plate solve did not succeed — set the class manually "
                                        "(or type the object name).")
            self.status_label.setText("No solve.")
            return
        cands = get_catalog().find_in_field(res.ra, res.dec, res.fov_radius_deg)
        if not cands:
            self.detected_label.setText("Solved, but no catalog object in field.")
            return
        self.object_combo.clear()
        self.object_combo.addItems([object_label(c.obj) for c in cands[:12]])
        top = cands[0].obj
        cls = classify_object(top, self.data)
        self.class_combo.setCurrentText(cls)
        msg = (f"Identified: {object_label(top)}  ->  class '{cls}'  "
               f"({'header WCS' if res.fromExisting else 'ASTAP'}).")
        rec = curated_for(top.id)
        if rec:
            self._apply_preset_to_controls(rec["settings"])
            msg += f"  ·  applied recipe: {rec['name']}"
        self.detected_label.setText(msg)
        self.status_label.setText("Identified.")

    def _apply_preset_to_controls(self, settings: dict):
        """Set the widgets from a curated recipe's / recommendation's settings dict."""
        for k, v in settings.items():
            if k in self.dials:
                self.dials[k].set_value(float(v))
            elif k in self.checks:
                self.checks[k].setChecked(bool(v))
            elif k == "cropPercent":
                self.crop_slider.set_value(float(v))
            elif k == "autoAssess":
                self.auto_check.setChecked(bool(v))

    def _analyze(self):
        """Measure the frame and show analysis + recommendations (Advisor)."""
        if self.loaded is None:
            self.status_label.setText("Load an image first.")
            return
        from ..objects.analyze import analyze_view

        res = analyze_view(self.loaded.data, self._collect_params(), self.data)
        self.log_view.clear()
        for line in res.lines:
            self.log_view.appendPlainText(line)
        self._recommendations = res.recommendations
        if res.recommendations:
            self.log_view.appendPlainText("\nRecommendations (click 'Apply recommendations'):")
            for r in res.recommendations:
                self.log_view.appendPlainText(f"  • {r['label']}")
            self.apply_reco_btn.setEnabled(True)
            self.status_label.setText(f"Analyzed — {len(res.recommendations)} recommendation(s).")
        else:
            self.apply_reco_btn.setEnabled(False)
            self.status_label.setText("Analyzed — no changes recommended.")

    def _apply_recommendations(self):
        if not self._recommendations:
            return
        self._apply_preset_to_controls({r["key"]: r["val"] for r in self._recommendations})
        self.apply_reco_btn.setEnabled(False)
        self.status_label.setText(f"Applied {len(self._recommendations)} recommendation(s).")

    # ---- recipes -------------------------------------------------------------

    def _save_recipe(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save recipe", "recipe.lsrecipe", "LazyStretch recipe (*.lsrecipe *.json)")
        if not path:
            return
        try:
            save_recipe(path, self._collect_params(), self.data)
            self.status_label.setText(f"Saved recipe {Path(path).name}.")
        except Exception as e:
            QMessageBox.critical(self, "Save recipe failed", str(e))

    def _load_recipe(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load recipe", "", "LazyStretch recipe (*.lsrecipe *.json)")
        if not path:
            return
        try:
            recipe = load_recipe(path)
        except Exception as e:
            QMessageBox.critical(self, "Load recipe failed", str(e))
            return
        p = Parameters.for_object(recipe.get("objectClass", self.class_combo.currentText()),
                                  data=self.data)
        n = apply_recipe(p, recipe, self.data)
        self._load_params_into_controls(p)
        self.status_label.setText(f"Loaded recipe {Path(path).name} ({n} settings).")

    def _load_params_into_controls(self, p: Parameters):
        """Reverse-sync: push a Parameters object out to every control."""
        self.class_combo.setCurrentText(p.object_class)   # fires profile toggles; overridden below
        if p.palette:
            idx = self.palette_combo.findText(p.palette)
            if idx < 0:
                self.palette_combo.addItem(p.palette)
                idx = self.palette_combo.findText(p.palette)
            self.palette_combo.setCurrentIndex(idx)
        for attr, fs in self.dials.items():
            fs.set_value(float(getattr(p, attr)))
        self.crop_slider.set_value(float(p.cropPercent))
        self.auto_check.setChecked(bool(p.autoAssess))
        for attr, cb in self.checks.items():
            cb.setChecked(bool(getattr(p, attr)))

    def _collect_params(self) -> Parameters:
        cls = self.class_combo.currentText()
        p = Parameters(object_class=cls, palette=self.palette_combo.currentText())
        for attr in self.checks:
            setattr(p, attr, self.checks[attr].isChecked())
        for attr, fs in self.dials.items():
            setattr(p, attr, fs.value())
        p.cropPercent = self.crop_slider.value()
        p.autoAssess = self.auto_check.isChecked()
        p.object_id = self.object_combo.currentText()
        if self.ha_picker.path() and self.oiii_picker.path():
            p.ha = self._mono(self.ha_picker.path())
            p.oiii = self._mono(self.oiii_picker.path())
            p.sii = self._mono(self.sii_picker.path()) if self.sii_picker.path() else None
        return p

    @staticmethod
    def _mono(path: str) -> np.ndarray:
        d = load_image(path).data
        return d[..., 0] if d.ndim == 3 else d

    def _run(self, *, preview: bool, mode: str):
        if self.worker is not None and self.worker.isRunning():
            return
        params = self._collect_params()
        image = self.loaded.data if self.loaded is not None else None
        if image is None and params.ha is None:
            self.status_label.setText("Load an image or assign Ha + OIII channels.")
            return
        self.log_view.clear()
        self._set_busy(True)
        self.progress.setRange(0, 0)  # indeterminate until first step
        self.status_label.setText("Preview…" if preview else "Executing…")

        self.worker = PipelineWorker(image, params, preview=preview, mode=mode,
                                     tools=self.tools, solve_result=self.solve_result)
        self.worker.progress.connect(self._on_progress)
        self.worker.logline.connect(self._on_log)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, i: int, total: int, name: str):
        self.progress.setRange(0, total)
        self.progress.setValue(i)
        self.status_label.setText(f"[{i}/{total}] {name}")

    def _on_log(self, line: str):
        self.log_view.appendPlainText(line)

    def _on_finished(self, result):
        self.result_image = result.image
        self.result_stars = result.stars_layer
        self.preview.set_image(result.image)
        self.save_btn.setEnabled(True)
        self._set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        skipped = f", {len(result.steps_skipped)} skipped" if result.steps_skipped else ""
        self.status_label.setText(
            f"Done — class '{result.object_class}', {len(result.steps_run)} steps{skipped}.")
        if self.worker is not None and self.worker.mode == "execute":
            self._save_result()

    def _on_failed(self, msg: str):
        self._set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label.setText("Failed.")
        QMessageBox.critical(self, "Pipeline error", msg)

    def _set_busy(self, busy: bool):
        for b in (self.preview_btn, self.execute_btn, self.identify_btn):
            b.setEnabled(not busy)

    def _save_result(self):
        if self.result_image is None:
            return
        stem = Path(self.target_picker.path()).stem if self.target_picker.path() else "lazystretch"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", f"{stem}_LazyStretch.tif",
            "TIFF (*.tif *.tiff);;PNG (*.png);;FITS (*.fits *.fit)")
        if not path:
            return
        try:
            save_image(path, self.result_image, bit_depth=16)
            msg = f"Saved {Path(path).name}."
            if self.result_stars is not None:
                stars_path = Path(path).with_name(Path(path).stem + "_stars" + Path(path).suffix)
                save_image(str(stars_path), self.result_stars, bit_depth=16)
                msg += f" Stars layer -> {stars_path.name}."
            self.status_label.setText(msg)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
