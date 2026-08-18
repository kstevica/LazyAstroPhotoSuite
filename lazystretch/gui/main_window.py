"""The LazyStretch desktop main window (PySide6) — wraps the headless core."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
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
from ..io.history import (
    MAX_ITEMS,
    STATUS_DISCARDED,
    STATUS_NONE,
    STATUS_SELECTED,
    HistoryStore,
)
from ..io.image_io import LoadedImage, load_image, save_image
from ..io.pins import load_pins, save_pins
from ..io.recipes import apply_recipe, load_recipe, recipe_from_params, save_recipe
from ..objects.model import Parameters
from ..objects.presets import curated_for
from ..pipeline.ledger import DECIDED as _L_DECIDED
from .preview import FullScreenViewer, PreviewView
from .widgets import FilePicker, FloatSlider, RunLogView
from .worker import PipelineWorker

# (attribute, label) for the option checkboxes. The first six are profile-driven.
_PROFILE_TOGGLES = ["doSCNR", "doLocalContrast", "doBXT", "doNR", "doHDR",
                    "doStarReduce", "haloTamer", "doBgExtract", "darkLaneGC", "reduceCast"]
_OPTIONS = [
    ("doBgExtract", "Background / gradient extraction"),
    ("useGradientCorrection", "…use GradientCorrection (stronger)"),
    ("darkLaneGC", "…dark-lane gradient (no empty sky)"),
    ("doColorCal", "Color calibration"),
    ("preferSPCC", "…prefer SPCC (needs a solve)"),
    ("fixChromatic", "Fix chromatic aberration (align R/G to B, before deconv)"),
    ("doBXT", "Deconvolution / sharpening (BlurX)"),
    ("useClassicalDeconv", "…classical RL deconvolution (no BlurX)"),
    ("doNR", "Noise reduction"),
    ("useNRMask", "…background-only (masked)"),
    ("doSCNR", "Remove green (SCNR)"),
    ("doHDR", "HDR core compression"),
    ("doStarReduce", "Star reduction"),
    ("haloTamer", "Halo Tamer (reflection filter-ring suppression)"),
    ("removeStars", "Remove stars (starless + stars layer)"),
    ("useMask", "Protect faint signal with mask"),
    ("adaptiveFloor", "Adaptive shadow floor"),
    ("starProtect", "Star-protected saturation"),
    ("doLocalContrast", "Local contrast (LHE)"),
    ("protectCores", "Protect bright cores"),
    ("enhanceEmission", "Enhance emission (Ha)"),
    ("reduceCast", "Reduce background color cast"),
    ("developForeground", "Develop nightscape foreground (off = already developed)"),
    ("inputStretched", "Input already stretched (polish only)"),
    ("debugBackground", "Show estimated background (debug: saves the model)"),
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
    ("structure", "Mid-scale structure (galaxy)", 0.0, 1.0, 2, 0.0),
    ("deepen", "Deepen (2nd pass)", 0.0, 1.0, 2, 0.0),
    ("highlights", "Highlights (0 dim → 1 bright)", 0.0, 1.0, 2, 0.20),
    ("transparency", "Core transparency (0 = off)", 0.0, 1.0, 2, 0.0),
    ("dimCore", "Dim core (0 = off)", 0.0, 1.0, 2, 0.0),
    ("starShrink", "Reduce stars (no StarNet, 0 = off)", 0.0, 1.0, 2, 0.0),
    ("deepenBackground", "De-veil background (0 = off)", 0.0, 1.0, 2, 0.0),
    ("snrProtect", "SNR protect (needs stack)", 0.0, 1.0, 2, 0.0),
    ("significance", "Significance (needs stack)", 0.0, 1.0, 2, 0.0),
    ("meteorStrength", "Meteor trails (needs stack)", 0.0, 1.0, 2, 1.0),
    ("nightscapeBrightness", "Nightscape foreground (needs stack)", 0.0, 1.0, 2, 0.5),
]


class LazyStretchPanel(QWidget):
    """The LazyStretch tool as an embeddable panel (hosted by gui/shell.py).

    Self-contained ``QWidget`` — the only top-level chrome it sets (title/size) is
    harmless when embedded in the shell's ``QStackedWidget`` and useful if shown alone.
    """

    # Emitted with a TIFF path when the user clicks "Develop" on a history item; the shell
    # opens LazyDevelop and loads it. Keeps the panels decoupled (no direct shell handle).
    openInDevelop = Signal(str)

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
        self._last_params: Optional[Parameters] = None
        self._hist_store: Optional[HistoryStore] = None   # disk-backed, per-master
        self._refreshing_history = False                  # guard reentrant selection signals
        self._fs_viewer: Optional[FullScreenViewer] = None  # kept alive while full-screen
        self._work_image: Optional[np.ndarray] = None     # base override: a history image to
        #                                                 continue from (else the loaded master)
        self._snr_noise_map: Optional[np.ndarray] = None  # LazyStack noise map for the loaded master
        self._snr_coverage_map: Optional[np.ndarray] = None  # LazyStack frame-support map
        self._meteor_layer: Optional[np.ndarray] = None      # LazyStack meteor layer for the master
        self._meteor_labels: Optional[np.ndarray] = None     # per-meteor id map
        self._nightscape_fg: Optional[np.ndarray] = None     # LazyStack nightscape foreground layer
        self._nightscape_mask: Optional[np.ndarray] = None   # nightscape feathered sky mask
        self._pins: Dict[str, object] = {}                # PROC pins for the loaded master

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

    @staticmethod
    def _scrolled(inner: QWidget) -> QScrollArea:
        """Wrap a panel in a resizable scroll area (each tab scrolls on its own)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    def _build_left(self) -> QWidget:
        # Two tabs: "Setup" (target/identify/output/narrowband) and "Adjust"
        # (adjustments + options side by side, plus the recipe buttons).
        tabs = QTabWidget()
        tabs.addTab(self._scrolled(self._build_setup_tab()), "Setup")
        tabs.addTab(self._scrolled(self._build_adjust_tab()), "Adjust")
        tabs.addTab(self._build_process_tab(), "Process")
        tabs.setMinimumWidth(720)
        return tabs

    def _build_process_tab(self) -> QWidget:
        """Tab 3: the PROC ledger — every computed pipeline value, live, each pinnable."""
        w = QWidget()
        v = QVBoxLayout(w)
        intro = QLabel("Every value the stretch engine decided this run. Select a row and "
                       "Pin to force a value; the next run re-derives from it. Measured "
                       "facts (italic) can't be pinned. Pins are saved per-master.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray;")
        v.addWidget(intro)

        self.proc_tree = QTreeWidget()
        self.proc_tree.setColumnCount(3)
        self.proc_tree.setHeaderLabels(["Step / value", "Value", "Pinned"])
        self.proc_tree.setRootIsDecorated(True)
        self.proc_tree.setAlternatingRowColors(True)
        self.proc_tree.currentItemChanged.connect(self._on_proc_row)
        v.addWidget(self.proc_tree, 1)

        row = QHBoxLayout()
        self.pin_edit = QLineEdit()
        self.pin_edit.setPlaceholderText("select a decided row…")
        self.pin_btn = QPushButton("Pin")
        self.unpin_btn = QPushButton("Unpin")
        self.unpin_all_btn = QPushButton("Unpin all")
        self.pin_btn.clicked.connect(self._pin_selected)
        self.unpin_btn.clicked.connect(self._unpin_selected)
        self.unpin_all_btn.clicked.connect(self._unpin_all)
        for wdg in (self.pin_btn, self.unpin_btn, self.unpin_all_btn):
            wdg.setFocusPolicy(Qt.NoFocus)
        row.addWidget(self.pin_edit, 1)
        row.addWidget(self.pin_btn)
        row.addWidget(self.unpin_btn)
        row.addWidget(self.unpin_all_btn)
        v.addLayout(row)
        self._set_pin_controls_enabled(False)
        return w

    def _build_setup_tab(self) -> QWidget:
        """Tab 1: Target, Identify, Output, Narrowband — the pre-processing setup."""
        w = QWidget()
        col1 = QVBoxLayout(w)

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
        grid.addWidget(QLabel("Data type:"), 0, 0)
        grid.addWidget(self.type_label, 0, 1)
        grid.addWidget(QLabel("Object class:"), 1, 0)
        grid.addWidget(self.class_combo, 1, 1)
        grid.addWidget(QLabel("Palette:"), 2, 0)
        grid.addWidget(self.palette_combo, 2, 1)
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
        col1.addStretch(1)
        return w

    def _build_adjust_tab(self) -> QWidget:
        """Tab 2: Adjustments and Options side by side, with the recipe buttons below."""
        w = QWidget()
        outer = QVBoxLayout(w)

        g_adj = QGroupBox("Adjustments")
        v = QVBoxLayout(g_adj)
        self.crop_slider = FloatSlider("Crop %", 0.0, 10.0, 3.0, decimals=1)
        self.auto_check = QCheckBox("Auto crop % + gradient tool")
        self.auto_check.setChecked(True)
        v.addWidget(self.crop_slider)
        v.addWidget(self.auto_check)
        self._dial_defaults: Dict[str, float] = {}
        for attr, label, lo, hi, dec, dflt in _DIALS:
            fs = FloatSlider(label, lo, hi, dflt, decimals=dec)
            self.dials[attr] = fs
            self._dial_defaults[attr] = dflt
            v.addWidget(fs)
        reset = QPushButton("Reset to class defaults")
        reset.clicked.connect(self._reset_dials)
        v.addWidget(reset)
        v.addStretch(1)

        g_opt = QGroupBox("Options")
        ov = QVBoxLayout(g_opt)
        defaults = Parameters()
        for attr, label in _OPTIONS:
            cb = QCheckBox(label)
            cb.setChecked(bool(getattr(defaults, attr)))
            self.checks[attr] = cb
            ov.addWidget(cb)
        ov.addStretch(1)

        cols = QHBoxLayout()
        cols.addWidget(g_adj)
        cols.addWidget(g_opt)
        outer.addLayout(cols, 1)

        self.meteor_group = QGroupBox("Meteor trails")
        mv = QVBoxLayout(self.meteor_group)
        mv.addWidget(QLabel("Detected trails — uncheck to hide from the image:"))
        self.meteor_list = QListWidget()
        self.meteor_list.setMaximumHeight(96)
        mv.addWidget(self.meteor_list)
        self.meteor_group.setVisible(False)          # shown only when a master has meteor trails
        outer.addWidget(self.meteor_group)

        g_rec = QGroupBox("Recipe")
        rv = QHBoxLayout(g_rec)
        save_rec = QPushButton("Save recipe…")
        load_rec = QPushButton("Load recipe…")
        save_rec.clicked.connect(self._save_recipe)
        load_rec.clicked.connect(self._load_recipe)
        rv.addWidget(save_rec)
        rv.addWidget(load_rec)
        outer.addWidget(g_rec)
        return w

    # --------------------------------------------------------------- right

    def _build_right(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # --- action buttons, above the image ---
        top_btns = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.execute_btn = QPushButton("Execute")
        self.save_btn = QPushButton("Save Result…")
        self.save_btn.setEnabled(False)
        self.fullscreen_btn = QPushButton("⛶ Full screen")
        self.fullscreen_btn.setToolTip("Show the current preview full-screen (Esc to close).")
        self.preview_btn.clicked.connect(lambda: self._run(preview=True, mode="preview"))
        self.execute_btn.clicked.connect(lambda: self._run(preview=False, mode="execute"))
        self.save_btn.clicked.connect(self._save_result)
        self.fullscreen_btn.clicked.connect(self._show_fullscreen)
        for b in (self.preview_btn, self.execute_btn, self.save_btn):
            top_btns.addWidget(b)
        top_btns.addStretch(1)
        top_btns.addWidget(self.fullscreen_btn)
        v.addLayout(top_btns)

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

        # --- history: header (label + small Clear) over [ list | controls column ] ---
        self.history_group = QWidget()
        hgv = QVBoxLayout(self.history_group)
        hgv.setContentsMargins(0, 0, 0, 0)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel(f"History — select to preview (last {MAX_ITEMS}, saved on disk)"))
        hdr.addStretch(1)
        self.open_folder_btn = QPushButton("Open folder")
        self.open_folder_btn.setToolTip("Open the history folder in the file manager.")
        self.open_folder_btn.setFocusPolicy(Qt.NoFocus)
        self.open_folder_btn.clicked.connect(self._open_history_folder)
        hdr.addWidget(self.open_folder_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedWidth(64)
        self.clear_btn.setFocusPolicy(Qt.NoFocus)
        self.clear_btn.clicked.connect(self._clear_history)
        hdr.addWidget(self.clear_btn)
        hgv.addLayout(hdr)

        body = QHBoxLayout()
        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(160)
        self.history_list.setToolTip("Select a past run to preview it and restore its settings.")
        self.history_list.currentItemChanged.connect(self._on_history_selected)
        body.addWidget(self.history_list, 1)                # list takes the width; grows tall

        ctrl = QVBoxLayout()
        star_row = QHBoxLayout()
        self.star_btns = []
        for i in range(5):
            b = QPushButton("☆")
            b.setFixedWidth(26)
            b.setFlat(True)
            b.setFocusPolicy(Qt.NoFocus)      # don't steal the list's selection/focus
            b.clicked.connect(lambda _=False, n=i + 1: self._rate_selected(n))
            self.star_btns.append(b)
            star_row.addWidget(b)
        star_row.addStretch(1)
        ctrl.addLayout(star_row)
        self.comment_edit = QLineEdit()
        self.comment_edit.setPlaceholderText("one-line comment…")
        self.comment_edit.editingFinished.connect(self._comment_selected)
        ctrl.addWidget(self.comment_edit)
        self.keep_btn = QPushButton("Keep ✓")
        self.discard_btn = QPushButton("Discard ✕")
        self.remove_btn = QPushButton("Remove")
        self.keep_btn.clicked.connect(lambda: self._status_selected(STATUS_SELECTED))
        self.discard_btn.clicked.connect(lambda: self._status_selected(STATUS_DISCARDED))
        self.remove_btn.clicked.connect(self._remove_selected)
        for b in (self.keep_btn, self.discard_btn, self.remove_btn):
            b.setFocusPolicy(Qt.NoFocus)      # keep the list's selection while acting on it
            ctrl.addWidget(b)
        self.continue_btn = QPushButton("Continue from image ▸")
        self.continue_btn.setToolTip(
            "Use this rendered image (16-bit) as the base for further processing, "
            "instead of the master. Enables polish mode; re-open the master to return to it.")
        self.continue_btn.setFocusPolicy(Qt.NoFocus)
        self.continue_btn.clicked.connect(self._continue_from_selected)
        ctrl.addSpacing(6)
        ctrl.addWidget(self.continue_btn)
        self.develop_btn = QPushButton("Develop ▸")
        self.develop_btn.setToolTip(
            "Open this history image (saved 16-bit TIFF) in LazyDevelop for hand-finishing.")
        self.develop_btn.setFocusPolicy(Qt.NoFocus)
        self.develop_btn.clicked.connect(self._develop_selected)
        ctrl.addWidget(self.develop_btn)
        ctrl.addStretch(1)
        ctrl_w = QWidget()
        ctrl_w.setLayout(ctrl)
        ctrl_w.setFixedWidth(200)
        body.addWidget(ctrl_w)
        hgv.addLayout(body)
        self.history_group.setVisible(False)      # appears once there's a run to show

        # --- log (full width; the action buttons live above the image) ---
        self.log_view = RunLogView()
        log_hdr = QHBoxLayout()
        log_hdr.addWidget(QLabel("Run log"))
        log_hdr.addStretch(1)
        self.save_log_btn = QPushButton("Save log…")
        self.save_log_btn.setToolTip("Save the run log to a text file (or right-click the log)")
        self.save_log_btn.clicked.connect(lambda: self.log_view.save_log(self))
        log_hdr.addWidget(self.save_log_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status_label = QLabel("Ready.")
        iv.addWidget(self.detected_label)
        iv.addWidget(self.tools_label)
        iv.addWidget(self.history_group)
        iv.addLayout(log_hdr)
        iv.addWidget(self.log_view, 1)
        iv.addWidget(self.progress)
        iv.addWidget(self.status_label)

        v.addWidget(info, 3)
        return w

    # ------------------------------------------------------------- actions

    def _show_fullscreen(self):
        """Open the image currently in the preview in a full-screen viewer."""
        arr = self.preview.current_array()
        if arr is None:
            self.status_label.setText("Nothing to show full-screen yet — Preview first.")
            return
        self._fs_viewer = FullScreenViewer()   # kept on self so it isn't GC'd while open
        self._fs_viewer.setWindowTitle("LazyStretch — full screen (Esc to close)")
        self._fs_viewer.set_image(arr, keep_view=False)
        self._fs_viewer.showFullScreen()
        self._fs_viewer.raise_()
        self._fs_viewer.activateWindow()

    def _apply_class_profile(self, cls: str):
        """Class selection overwrites the profile-driven toggles + the chroma-NR default (PI behavior)."""
        prof = self.data.profile_for(cls)
        mapping = {"doSCNR": prof.scnr, "doLocalContrast": prof.localContrast,
                   "doBXT": prof.bxt, "doNR": prof.nr, "doHDR": prof.hdr,
                   "doStarReduce": prof.starReduce, "haloTamer": prof.haloTamer,
                   "doBgExtract": prof.bgExtract, "darkLaneGC": prof.darkLaneGC,
                   "reduceCast": prof.reduceCast}
        for attr in _PROFILE_TOGGLES:
            cb = self.checks[attr]
            cb.blockSignals(True)
            cb.setChecked(bool(mapping[attr]))
            cb.blockSignals(False)
        for dial, val in (("chromaNR", prof.chromaNR),   # profile-seeded dials (OSC chroma-NR; galaxy
                          ("structure", prof.structure)):  # mid-scale structure enhancement)
            if dial in self.dials:
                fs = self.dials[dial]
                fs.blockSignals(True)
                fs.set_value(val)
                fs.blockSignals(False)

    def _populate_meteor_list(self, meta):
        """Fill the checkable trail list from the master's meteor metadata (hidden if none)."""
        self.meteor_list.clear()
        for m in meta:
            mid = int(m.get("id", 0))
            src = m.get("source") or "?"
            ts = m.get("timestamp")
            when = f" @ {ts}" if ts else ""
            it = QListWidgetItem(f"Meteor {mid} — {src}{when}  (len {float(m.get('length', 0)):.0f}px)")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            it.setData(Qt.UserRole, mid)
            self.meteor_list.addItem(it)
        self.meteor_group.setVisible(bool(meta))

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
        self._work_image = None       # a fresh master supersedes any "continue from" base
        dtype = detect_data_type(self.loaded.keyword("FILTER"), self.loaded.is_color,
                                 self.loaded.keyword("BAYERPAT"))
        self.type_label.setText(dtype)
        self.palette_combo.clear()
        self.palette_combo.addItems(palettes_for(dtype))
        self.detected_label.setText(
            f"Loaded {Path(path).name}  —  {self.loaded.data.shape}, "
            f"{'color' if self.loaded.is_color else 'mono'}, type '{dtype}'.")
        self.status_label.setText("Loaded. Showing a quick display-stretch…")
        # Per-master history lives in a `history/` subfolder next to the master and
        # reloads across sessions.
        mpath = getattr(self.loaded, "path", None)
        self._hist_store = HistoryStore(mpath) if mpath else None
        self._pins = load_pins(mpath) if mpath else {}
        # SNR-protect: pick up a LazyStack noise map sitting next to the master (if any).
        from ..lazystack.meteors import load_meteor_labels, load_meteor_layer, load_meteor_meta
        from ..processes.snrmask import load_coverage_map, load_noise_map
        self._snr_noise_map = load_noise_map(mpath) if mpath else None
        self._snr_coverage_map = load_coverage_map(mpath) if mpath else None
        self._meteor_layer = load_meteor_layer(mpath) if mpath else None
        self._meteor_labels = load_meteor_labels(mpath) if mpath else None
        self._populate_meteor_list(load_meteor_meta(mpath) if mpath else [])
        from ..lazystack.nightscape import load_foreground as _ld_fg, load_sky_mask as _ld_mask
        self._nightscape_fg = _ld_fg(mpath) if mpath else None
        self._nightscape_mask = _ld_mask(mpath) if mpath else None
        if self._snr_noise_map is not None:
            self.status_label.setText("Loaded. LazyStack noise map found — SNR-protect available.")
        if self._meteor_layer is not None:
            self.status_label.setText("Loaded. LazyStack meteor trail(s) found — will composite.")
        if self._nightscape_fg is not None:
            self.status_label.setText("Loaded. Nightscape foreground found — will composite over the sky.")
        self._refresh_history_list()
        self._show_quick_display()

    def _show_quick_display(self):
        """Fast downsampled auto-stretch just so the loaded master is visible."""
        a = self.loaded.data
        step = max(1, max(a.shape[0], a.shape[1]) // 1400)
        small = a[::step, ::step]
        try:
            from ..stretch.autostretch import apply_auto_stretch
            disp, _ = apply_auto_stretch(small, -1.25, 0.25)
            self.preview.set_image(disp, keep_view=False)   # new frame -> fit to window
            self.status_label.setText("Ready. Click Preview for the full look.")
        except Exception:
            self.preview.set_image(small, keep_view=False)

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
        # A wide field is the Milky Way (widefield) class regardless of any catalog object
        # in frame (LazyStretch.js:2753-2757).
        fov_wide = math.isfinite(res.fovDeg) and res.fovDeg >= 14
        if not cands and not fov_wide:
            self.detected_label.setText("Solved, but no catalog object in field.")
            self.status_label.setText("Solved.")
            return

        top = None
        cls = "generic"
        if cands:
            self.object_combo.clear()
            self.object_combo.addItems([object_label(c.obj) for c in cands[:12]])
            top = cands[0].obj
            cls = classify_object(top, self.data)
        if fov_wide:
            cls = "milkyway"
        self.class_combo.setCurrentText(cls)

        src = "header WCS" if res.fromExisting else "ASTAP"
        if fov_wide:
            msg = f"Field {res.fovDeg:.1f}° wide -> Milky Way (widefield) class ({src})."
            if top is not None:
                msg += f"  ·  nearest: {object_label(top)}"
        else:
            msg = f"Identified: {object_label(top)}  ->  class '{cls}'  ({src})."
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
        self.log_view.finish()

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
        p.snr_noise_map = self._snr_noise_map          # SNR-protect maps for the loaded master (or None)
        p.snr_coverage_map = self._snr_coverage_map
        p.meteor_layer = self._meteor_layer            # LazyStack meteor layer for the master (or None)
        p.meteor_labels = self._meteor_labels
        p.nightscape_foreground_layer = self._nightscape_fg   # nightscape foreground + mask (or None)
        p.nightscape_sky_mask = self._nightscape_mask
        if self.meteor_list.count():                   # which trails the user chose to keep visible
            sel = [self.meteor_list.item(i).data(Qt.UserRole)
                   for i in range(self.meteor_list.count())
                   if self.meteor_list.item(i).checkState() == Qt.Checked]
            p.meteorSelect = None if len(sel) == self.meteor_list.count() else sel
        else:
            p.meteorSelect = None
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
        self._last_params = params
        # A history image the user chose to continue from overrides the master as the base.
        if self._work_image is not None:
            image = self._work_image
        else:
            image = self.loaded.data if self.loaded is not None else None
        if image is None and params.ha is None:
            self.status_label.setText("Load an image or assign Ha + OIII channels.")
            return
        self.log_view.clear()
        self._set_busy(True)
        self.progress.setRange(0, 0)  # indeterminate until first step
        self.status_label.setText("Preview…" if preview else "Executing…")

        self.worker = PipelineWorker(image, params, preview=preview, mode=mode,
                                     tools=self.tools, solve_result=self.solve_result,
                                     pins=dict(self._pins))
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
        self.log_view.finish()
        self.result_image = result.image
        self.result_stars = result.stars_layer
        bg = getattr(result, "background_model", None)     # debug: save the estimated background model
        if bg is not None:
            try:
                mpath = getattr(self.loaded, "path", None)
                if mpath:
                    norm = (bg - float(np.min(bg))) / (float(np.ptp(bg)) or 1.0)
                    bg_path = str(Path(mpath).with_name(Path(mpath).stem + "_background.tif"))
                    save_image(bg_path, norm, bit_depth=16)
                    self.status_label.setText(f"Background model saved: {Path(bg_path).name} "
                                              f"(should show NO galactic structure)")
            except Exception:
                pass
        self.preview.set_image(result.image)     # same size -> keeps zoom/pan for comparison
        self.save_btn.setEnabled(True)
        self._populate_process_tab(getattr(result, "ledger", None))
        self._set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        mode = self.worker.mode if self.worker is not None else "preview"
        self._push_history(result, mode)
        skipped = f", {len(result.steps_skipped)} skipped" if result.steps_skipped else ""
        self.status_label.setText(
            f"Done — class '{result.object_class}', {len(result.steps_run)} steps{skipped}.")
        if self.worker is not None and self.worker.mode == "execute":
            self._save_result()

    # --- run history (disk-backed, per-master: settings + image + rating/comment) ----

    @staticmethod
    def _run_summary(p: Parameters, mode: str, steps: int) -> str:
        """A compact one-line summary of the settings that distinguish a run."""
        return (f"{mode} · {p.object_class} · HL{p.highlights:.2f} DC{p.dimCore:.2f} "
                f"T{p.transparency:.2f} sat{p.satAdj:+.2f} blk{p.blackAdj:+.2f} deep{p.deepen:.2f}"
                f" · {steps} steps")

    def _push_history(self, result, mode: str):
        """Save the finished run (settings + rendered image) to the on-disk history."""
        if self._hist_store is None or self._last_params is None:
            return                                             # narrowband-only / no master path
        recipe = recipe_from_params(self._last_params, self.data, include_state=True)
        label = self._run_summary(self._last_params, mode, len(result.steps_run))
        self._hist_store.add(np.asarray(result.image, dtype=np.float32),
                             recipe, label, mode, len(result.steps_run))
        self._refresh_history_list()

    @staticmethod
    def _item_label(item: dict) -> str:
        rating = int(item.get("rating", 0))
        stars = "★" * rating + "☆" * (5 - rating)
        st = item.get("status", STATUS_NONE)
        badge = " ✓" if st == STATUS_SELECTED else (" ✕" if st == STATUS_DISCARDED else "")
        comment = str(item.get("comment", "")).strip()
        ctxt = f"  — {comment}" if comment else ""
        return f"{stars}{badge}  {item.get('label', '')}{ctxt}"

    def _history_items(self) -> list:
        """Store items newest-first (matching the list-widget order)."""
        return list(reversed(self._hist_store.items)) if self._hist_store else []

    def _selected_item(self) -> Optional[dict]:
        row = self.history_list.currentRow()
        items = self._history_items()
        return items[row] if 0 <= row < len(items) else None

    def _refresh_history_list(self):
        self._refreshing_history = True
        self.history_list.clear()
        for item in self._history_items():                     # newest first
            li = QListWidgetItem(self._item_label(item), self.history_list)
            if item.get("status") == STATUS_DISCARDED:
                li.setForeground(Qt.gray)
        self.history_group.setVisible(bool(self._history_items()))
        self._refreshing_history = False

    def _on_history_selected(self, current, _previous=None):
        """Selecting a past run previews it, restores its settings, and fills the rating box."""
        if self._refreshing_history or current is None or self._hist_store is None:
            self._update_star_display(0)
            return
        item = self._selected_item()
        if item is None:
            return
        img = self._hist_store.load_image_data(item)
        if img is not None:
            self.preview.set_image(img, keep_view=True)        # compare in place
            self.result_image = np.asarray(img, dtype=np.float64)
            self.save_btn.setEnabled(True)
        p = Parameters.for_object(item["params"].get("objectClass", "generic"), data=self.data)
        apply_recipe(p, item["params"], self.data)
        self._load_params_into_controls(p)
        self._update_star_display(int(item.get("rating", 0)))
        self.comment_edit.setText(str(item.get("comment", "")))
        self.status_label.setText(f"History: {item.get('label', '')}")

    def _update_star_display(self, rating: int):
        for i, b in enumerate(self.star_btns):
            b.setText("★" if i < rating else "☆")

    def _rate_selected(self, n: int):
        item = self._selected_item()
        if item is None or self._hist_store is None:
            return
        new = 0 if int(item.get("rating", 0)) == n else n      # click current star -> clear
        self._hist_store.set_rating(item, new)
        self._update_star_display(new)
        self._relabel_current(item)

    def _comment_selected(self):
        item = self._selected_item()
        if item is None or self._hist_store is None:
            return
        self._hist_store.set_comment(item, self.comment_edit.text())
        self._relabel_current(item)

    def _status_selected(self, status: str):
        item = self._selected_item()
        if item is None or self._hist_store is None:
            return
        new = STATUS_NONE if item.get("status") == status else status   # toggle
        self._hist_store.set_status(item, new)
        self._relabel_current(item)

    def _remove_selected(self):
        item = self._selected_item()
        if item is None or self._hist_store is None:
            return
        self._hist_store.remove(item)
        self._refresh_history_list()

    def _continue_from_selected(self):
        """Make the selected history image the base for further processing.

        Loads its full-precision (16-bit) pixels as the working image so Preview/Execute
        build on it instead of the master, switches to polish mode (no re-stretch or
        re-calibration of an already-finished frame), and drops crop (the base is already
        cropped). Re-opening the master returns to it.
        """
        item = self._selected_item()
        if item is None or self._hist_store is None:
            return
        img = self._hist_store.load_image_data(item)
        if img is None:
            self.status_label.setText("Can't continue — the history image is missing on disk.")
            return
        self._work_image = np.asarray(img, dtype=np.float64)
        self.checks["inputStretched"].setChecked(True)   # finished frame -> polish only
        self.crop_slider.set_value(0.0)                  # already cropped; don't re-crop
        self.auto_check.setChecked(False)
        self.result_image = self._work_image
        self.result_stars = None
        self.save_btn.setEnabled(True)
        self.preview.set_image(self._work_image, keep_view=True)
        label = item.get("label", "")
        self.detected_label.setText(
            f"Base = history image “{label}” (polish mode). Preview/Execute build on it; "
            "re-open the master to go back.")
        self.status_label.setText("Continuing from the selected history image.")

    def _develop_selected(self):
        """Open the selected history image in LazyDevelop (it is already a 16-bit TIFF on disk)."""
        item = self._selected_item()
        if item is None or self._hist_store is None:
            self.status_label.setText("Select a history run to develop first.")
            return
        path = self._hist_store.image_path(item)
        if not Path(path).exists():
            self.status_label.setText("Can't develop — the history image is missing on disk.")
            return
        self.openInDevelop.emit(path)                    # shell opens LazyDevelop + loads it
        self.status_label.setText(f"Opening “{item.get('label', '')}” in LazyDevelop…")

    def _open_history_folder(self):
        """Open the per-master history folder in the OS file manager (Win/mac/Linux).

        ``QDesktopServices.openUrl`` on a ``file://`` URL is Qt's cross-platform shell-open:
        Explorer on Windows, Finder on macOS, the default file manager (xdg-open) on Linux.
        """
        if self._hist_store is None:
            self.status_label.setText("Load a master first — history is saved next to it.")
            return
        d = self._hist_store.dir
        d.mkdir(parents=True, exist_ok=True)             # may not exist until the first run
        if QDesktopServices.openUrl(QUrl.fromLocalFile(str(d))):
            self.status_label.setText(f"Opened {d}")
        else:
            self.status_label.setText(f"Couldn't open the file manager — folder is: {d}")

    def _clear_history(self):
        if self._hist_store is None or not self._hist_store.items:
            return
        if QMessageBox.question(self, "Clear history",
                                "Delete all history items for this master (files on disk too)?") \
                != QMessageBox.Yes:
            return
        self._hist_store.clear()
        self._refresh_history_list()

    def _relabel_current(self, item: dict):
        """Update ONLY the selected row's text/colour after a rating/comment/status edit.
        Touches neither the selection nor the preview (no re-select, no signal)."""
        li = self.history_list.item(self.history_list.currentRow())
        if li is None:
            return
        li.setText(self._item_label(item))
        if item.get("status") == STATUS_DISCARDED:
            li.setForeground(Qt.gray)
        else:
            li.setData(Qt.ForegroundRole, None)     # reset to the theme's default text colour

    # --- PROC ledger / pins (Stretch Process tab) ----------------------------

    @staticmethod
    def _fmt_val(v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    def _set_pin_controls_enabled(self, on: bool):
        for wdg in (self.pin_edit, self.pin_btn, self.unpin_btn):
            wdg.setEnabled(on)

    def _populate_process_tab(self, ledger):
        self.proc_tree.clear()
        self._proc_items: Dict[str, QTreeWidgetItem] = {}
        if ledger is None:
            return
        groups: Dict[str, QTreeWidgetItem] = {}
        for e in ledger.entries:
            node = groups.get(e.group)
            if node is None:
                node = QTreeWidgetItem(self.proc_tree, [e.group, "", ""])
                groups[e.group] = node
            item = QTreeWidgetItem(node, [e.name, self._fmt_val(e.computed), ""])
            item.setData(0, Qt.UserRole, {"key": e.key, "kind": e.kind, "computed": e.computed})
            self._style_proc_item(item, e.kind, self._pins.get(e.key))
            self._proc_items[e.key] = item
        self.proc_tree.expandAll()
        self.proc_tree.resizeColumnToContents(0)
        self.proc_tree.resizeColumnToContents(1)

    def _style_proc_item(self, item, kind: str, pin_val):
        """Set a value row's Pinned column + colour from its kind and any active pin."""
        pinnable = kind == _L_DECIDED
        if not pinnable:
            item.setForeground(0, Qt.gray)
            font = item.font(0)
            font.setItalic(True)
            item.setFont(0, font)
            return
        if pin_val is not None:
            item.setText(2, self._fmt_val(pin_val))
            for col in range(3):
                item.setForeground(col, Qt.darkYellow)
        else:
            item.setText(2, "")
            for col in range(3):
                item.setData(col, Qt.ForegroundRole, None)

    def _on_proc_row(self, current, _prev=None):
        meta = current.data(0, Qt.UserRole) if current is not None else None
        if not meta or meta.get("kind") != _L_DECIDED:
            self._set_pin_controls_enabled(False)
            self.pin_edit.clear()
            return
        self._set_pin_controls_enabled(True)
        key = meta["key"]
        cur = self._pins.get(key, meta["computed"])
        self.pin_edit.setText(self._fmt_val(cur))

    def _selected_proc_meta(self):
        item = self.proc_tree.currentItem()
        meta = item.data(0, Qt.UserRole) if item is not None else None
        if not meta or meta.get("kind") != _L_DECIDED:
            return None, None
        return item, meta

    @staticmethod
    def _parse_pin(text: str, computed):
        t = text.strip()
        if isinstance(computed, bool):
            return t.lower() in ("1", "true", "yes", "on")
        try:
            f = float(t)
        except ValueError:
            return t
        return int(round(f)) if isinstance(computed, int) and not isinstance(computed, bool) else f

    def _pin_selected(self):
        item, meta = self._selected_proc_meta()
        if item is None:
            return
        self._pins[meta["key"]] = self._parse_pin(self.pin_edit.text(), meta["computed"])
        self._save_pins()
        self._style_proc_item(item, meta["kind"], self._pins[meta["key"]])
        self.status_label.setText(f"Pinned {meta['key']} — takes effect on the next Preview.")

    def _unpin_selected(self):
        item, meta = self._selected_proc_meta()
        if item is None or meta["key"] not in self._pins:
            return
        del self._pins[meta["key"]]
        self._save_pins()
        self._style_proc_item(item, meta["kind"], None)
        self.pin_edit.setText(self._fmt_val(meta["computed"]))
        self.status_label.setText(f"Unpinned {meta['key']}.")

    def _unpin_all(self):
        if not self._pins:
            return
        self._pins = {}
        self._save_pins()
        for key, item in getattr(self, "_proc_items", {}).items():
            meta = item.data(0, Qt.UserRole)
            if meta:
                self._style_proc_item(item, meta["kind"], None)
        self.status_label.setText("All pins cleared.")

    def _save_pins(self):
        mpath = getattr(self.loaded, "path", None) if self.loaded is not None else None
        if mpath:
            save_pins(mpath, self._pins)

    def _on_failed(self, msg: str):
        self.log_view.finish()
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


# Backward-compatible alias: the LazyStretch panel used to be the top-level window.
MainWindow = LazyStretchPanel
