"""The LazyDevelop desktop panel — interactive image finishing (hosted by gui/shell.py).

Left column: source + a categorised toolbox (built generically from the op registry) +
the active tool's controls (with a mask/opacity gate) + a mask panel + the op history.
Right: the image canvas (rubber-band crop, mask painting, before/after) + a log.

Heavy tools (wavelets, HDR, ABE, NR) run in a background worker; light tools preview
live on the full-resolution image.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, QRectF, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsRectItem,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..develop import DevelopDocument, ops as dev_ops
from ..develop import masks as dev_masks
from ..io.image_io import load_image, save_image
from .preview import PreviewView
from .widgets import FloatSlider, RunLogView
from .worker import CallableWorker

# Heavy tools (wavelets, HDR, ABE, NR, deconv, …) preview on a downscaled proxy of the
# current result so sliders stay responsive; Apply always recomputes at full resolution.
PROXY_MAX_DIM = 1400


def _downscale_rgb(a: np.ndarray, max_dim: int = PROXY_MAX_DIM) -> np.ndarray:
    """Anti-aliased area downscale so the long edge ≤ max_dim (no-op if already small)."""
    a = np.asarray(a, dtype=np.float32)
    h, w = a.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return np.ascontiguousarray(a)
    f = int(np.ceil(m / max_dim))
    from scipy.ndimage import uniform_filter
    size = (f, f, 1) if a.ndim == 3 else f
    return np.ascontiguousarray(uniform_filter(a, size=size, mode="nearest")[::f, ::f])


# =============================================================================
# Curve editor widget
# =============================================================================
class CurveEditor(QWidget):
    """A small interactive tone-curve editor. Points are [x, y] in [0, 1]."""

    curveChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(200)
        self._pts: List[List[float]] = [[0.0, 0.0], [1.0, 1.0]]
        self._drag = -1
        self._pad = 10
        self.setMouseTracking(True)

    def set_points(self, pts):
        self._pts = [[float(x), float(y)] for x, y in (pts or [[0, 0], [1, 1]])]
        self._pts.sort(key=lambda p: p[0])
        self.update()

    def points(self):
        return [[round(x, 4), round(y, 4)] for x, y in self._pts]

    # --- geometry helpers (data [0,1] <-> widget pixels) ---
    def _to_px(self, x, y):
        w = self.width() - 2 * self._pad
        h = self.height() - 2 * self._pad
        return self._pad + x * w, self._pad + (1.0 - y) * h

    def _to_data(self, px, py):
        w = self.width() - 2 * self._pad
        h = self.height() - 2 * self._pad
        x = (px - self._pad) / max(w, 1)
        y = 1.0 - (py - self._pad) / max(h, 1)
        return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))

    def _hit(self, px, py):
        for i, (x, y) in enumerate(self._pts):
            ex, ey = self._to_px(x, y)
            if abs(ex - px) < 9 and abs(ey - py) < 9:
                return i
        return -1

    def paintEvent(self, _):
        from ..develop.curves import curve_lut
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.fillRect(r, QColor(30, 30, 34))
        p.setPen(QPen(QColor(70, 70, 78), 1))
        for k in range(1, 4):                       # grid quarters
            gx = self._pad + k / 4 * (self.width() - 2 * self._pad)
            gy = self._pad + k / 4 * (self.height() - 2 * self._pad)
            p.drawLine(int(gx), self._pad, int(gx), self.height() - self._pad)
            p.drawLine(self._pad, int(gy), self.width() - self._pad, int(gy))
        # curve
        lut = curve_lut(self._pts, n=128)
        p.setPen(QPen(QColor(120, 190, 255), 2))
        prev = None
        for i, yv in enumerate(lut):
            x = i / (len(lut) - 1)
            px, py = self._to_px(x, float(yv))
            if prev is not None:
                p.drawLine(int(prev[0]), int(prev[1]), int(px), int(py))
            prev = (px, py)
        # control points
        p.setPen(QPen(QColor(255, 255, 255), 1))
        for x, y in self._pts:
            ex, ey = self._to_px(x, y)
            p.setBrush(QColor(255, 210, 90))
            p.drawEllipse(int(ex) - 4, int(ey) - 4, 8, 8)

    def mousePressEvent(self, e):
        px, py = e.position().x(), e.position().y()
        i = self._hit(px, py)
        if e.button() == Qt.LeftButton:
            if i < 0:                                # add a new point
                x, y = self._to_data(px, py)
                self._pts.append([x, y])
                self._pts.sort(key=lambda p: p[0])
                i = self._hit(*self._to_px(x, y))
            self._drag = i
            self.update()

    def mouseMoveEvent(self, e):
        if self._drag < 0:
            return
        x, y = self._to_data(e.position().x(), e.position().y())
        endpoint = self._drag == 0 or self._drag == len(self._pts) - 1
        if endpoint:                                 # endpoints move only vertically
            self._pts[self._drag][1] = y
        else:
            lo = self._pts[self._drag - 1][0] + 1e-3
            hi = self._pts[self._drag + 1][0] - 1e-3
            self._pts[self._drag][0] = min(hi, max(lo, x))
            self._pts[self._drag][1] = y
        self.update()
        self.curveChanged.emit(self.points())

    def mouseReleaseEvent(self, _):
        self._drag = -1
        self.curveChanged.emit(self.points())

    def mouseDoubleClickEvent(self, e):
        i = self._hit(e.position().x(), e.position().y())
        if 0 < i < len(self._pts) - 1:               # remove an interior point
            del self._pts[i]
            self.update()
            self.curveChanged.emit(self.points())


# =============================================================================
# Canvas: PreviewView + rubber-band crop rectangle
# =============================================================================
class DevelopCanvas(PreviewView):
    """PreviewView plus a rubber-band rectangle for the Crop tool."""

    rectSelected = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rect_mode = False
        self._rect_item: Optional[QGraphicsRectItem] = None
        self._rect_start = None

    def set_rect_mode(self, on: bool):
        self._rect_mode = bool(on)
        if on:
            self.setDragMode(PreviewView.NoDrag)
        else:
            self.setDragMode(PreviewView.ScrollHandDrag)
            self._clear_rect_item()

    def _clear_rect_item(self):
        if self._rect_item is not None and self._rect_item.scene() is not None:
            self.scene().removeItem(self._rect_item)
        self._rect_item = None

    def show_rect(self, rect: dict):
        arr = self.current_array()
        if arr is None:
            return
        h, w = arr.shape[:2]
        x0 = rect.get("x0", 0.0) * w; x1 = rect.get("x1", 1.0) * w
        y0 = rect.get("y0", 0.0) * h; y1 = rect.get("y1", 1.0) * h
        if self._rect_item is None:
            self._rect_item = self.scene().addRect(QRectF(x0, y0, x1 - x0, y1 - y0),
                                                   QPen(QColor(255, 210, 90), 0))
            self._rect_item.setZValue(5)
        self._rect_item.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))

    def mousePressEvent(self, e):
        if self._rect_mode and e.button() == Qt.LeftButton and self.current_array() is not None:
            self._rect_start = self.mapToScene(e.position().toPoint())
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._rect_mode and self._rect_start is not None and (e.buttons() & Qt.LeftButton):
            self._emit_rect(self.mapToScene(e.position().toPoint()))
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._rect_mode and self._rect_start is not None:
            self._emit_rect(self.mapToScene(e.position().toPoint()))
            self._rect_start = None
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def _emit_rect(self, pt):
        arr = self.current_array()
        if arr is None:
            return
        h, w = arr.shape[:2]
        x0 = min(self._rect_start.x(), pt.x()); x1 = max(self._rect_start.x(), pt.x())
        y0 = min(self._rect_start.y(), pt.y()); y1 = max(self._rect_start.y(), pt.y())
        rect = {"x0": max(0.0, min(1.0, x0 / w)), "y0": max(0.0, min(1.0, y0 / h)),
                "x1": max(0.0, min(1.0, x1 / w)), "y1": max(0.0, min(1.0, y1 / h))}
        self.show_rect(rect)
        self.rectSelected.emit(rect)


# =============================================================================
# Generic tool parameter panel
# =============================================================================
class ToolPanel(QWidget):
    """Builds controls for one Op from its ParamSpecs, plus a mask/opacity gate."""

    changed = Signal()                     # a param changed → re-preview
    applied = Signal()                     # Apply (new step) / Update (edit) pressed
    cancelled = Signal()
    deleted = Signal()                     # Delete step pressed (edit mode only)
    rectModeChanged = Signal(bool)

    def __init__(self, op: "dev_ops.Op", mask_names: List[str], parent=None,
                 *, edit_index: Optional[int] = None, init: Optional[dict] = None):
        super().__init__(parent)
        self.op = op
        self.edit_index = edit_index       # None = new adjustment; int = editing that step
        self._values: Dict[str, object] = op.defaults()
        self._widgets: Dict[str, QWidget] = {}
        self._curve: Optional[CurveEditor] = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        title = QLabel(op.label)
        f = title.font(); f.setBold(True); f.setPointSize(f.pointSize() + 1)
        title.setFont(f)
        v.addWidget(title)
        if op.tooltip:
            hint = QLabel(op.tooltip); hint.setWordWrap(True)
            hint.setStyleSheet("color: gray;")
            v.addWidget(hint)

        for spec in op.params:
            w = self._build_control(spec)
            if w is not None:
                v.addWidget(w)

        self._has_rect = any(s.kind == "rect" for s in op.params)
        self._gateable = op.category != "Geometry"
        if self._gateable:
            v.addWidget(self._build_gate(mask_names))

        if init is not None:
            self._apply_init(init)

        editing = edit_index is not None
        row = QHBoxLayout()
        self.apply_btn = QPushButton("Update step" if editing else "Apply")
        self.cancel_btn = QPushButton("Cancel")
        self.apply_btn.clicked.connect(self.applied.emit)
        self.cancel_btn.clicked.connect(self.cancelled.emit)
        row.addWidget(self.apply_btn)
        if editing:
            self.delete_btn = QPushButton("Delete step")
            self.delete_btn.clicked.connect(self.deleted.emit)
            row.addWidget(self.delete_btn)
        row.addWidget(self.cancel_btn)
        v.addLayout(row)

    def _apply_init(self, init: dict):
        """Populate every control from a stored step's params + gate (edit mode)."""
        params = init.get("params", {}) or {}
        for spec in self.op.params:
            if spec.key not in params:
                continue
            val = params[spec.key]
            self._values[spec.key] = spec.coerce(val) if spec.kind not in ("curve", "rect") else val
            w = self._widgets.get(spec.key)
            if w is None:
                continue
            if spec.kind in ("float", "int"):
                w.set_value(float(val))
            elif spec.kind == "bool":
                w.setChecked(bool(val))
            elif spec.kind == "choice":
                w.setCurrentIndex(int(val))
            elif spec.kind == "curve":
                w.set_points(val)
        if self._gateable:
            mask = init.get("mask")
            idx = self.mask_combo.findText(mask) if mask else 0
            self.mask_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.invert_cb.setChecked(bool(init.get("mask_invert", False)))
            self.opacity.set_value(float(init.get("opacity", 1.0)))

    # ---- gate (mask + invert + opacity) ----
    def _build_gate(self, mask_names):
        box = QGroupBox("Apply through")
        gl = QVBoxLayout(box)
        self.mask_combo = QComboBox()
        self.mask_combo.addItem("Whole image")
        for n in mask_names:
            self.mask_combo.addItem(n)
        self.mask_combo.currentIndexChanged.connect(lambda _: self.changed.emit())
        gl.addWidget(self.mask_combo)
        self.invert_cb = QCheckBox("Invert mask")
        self.invert_cb.toggled.connect(lambda _: self.changed.emit())
        gl.addWidget(self.invert_cb)
        self.opacity = FloatSlider("Opacity", 0.0, 1.0, 1.0, 2)
        self.opacity.valueChanged.connect(lambda _: self.changed.emit())
        gl.addWidget(self.opacity)
        return box

    def refresh_masks(self, mask_names):
        if not self._gateable:
            return
        cur = self.mask_combo.currentText()
        self.mask_combo.blockSignals(True)
        self.mask_combo.clear()
        self.mask_combo.addItem("Whole image")
        for n in mask_names:
            self.mask_combo.addItem(n)
        idx = self.mask_combo.findText(cur)
        self.mask_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.mask_combo.blockSignals(False)

    def gate(self):
        if not self._gateable:
            return {"mask": None, "mask_invert": False, "opacity": 1.0}
        mask = self.mask_combo.currentText()
        return {"mask": None if mask == "Whole image" else mask,
                "mask_invert": self.invert_cb.isChecked(),
                "opacity": self.opacity.value()}

    # ---- param controls ----
    def _build_control(self, spec: "dev_ops.ParamSpec") -> Optional[QWidget]:
        if spec.kind in ("float", "int"):
            dec = 0 if spec.kind == "int" else spec.decimals
            sl = FloatSlider(spec.label, spec.lo, spec.hi, float(spec.default), dec)
            sl.valueChanged.connect(lambda val, k=spec.key: self._set(k, val))
            if spec.tooltip:
                sl.setToolTip(spec.tooltip)
            self._widgets[spec.key] = sl
            return sl
        if spec.kind == "bool":
            cb = QCheckBox(spec.label)
            cb.setChecked(bool(spec.default))
            cb.toggled.connect(lambda val, k=spec.key: self._set(k, val))
            self._widgets[spec.key] = cb
            return cb
        if spec.kind == "choice":
            box = QWidget(); h = QHBoxLayout(box); h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(spec.label))
            combo = QComboBox()
            combo.addItems(spec.choices or [])
            combo.setCurrentIndex(int(spec.default))
            combo.currentIndexChanged.connect(lambda val, k=spec.key: self._set(k, val))
            h.addWidget(combo, 1)
            self._widgets[spec.key] = combo
            return box
        if spec.kind == "curve":
            self._curve = CurveEditor()
            self._curve.set_points(spec.default)
            self._curve.curveChanged.connect(lambda pts, k=spec.key: self._set(k, pts))
            self._widgets[spec.key] = self._curve
            return self._curve
        if spec.kind == "rect":
            lbl = QLabel("Drag a rectangle on the image to set the crop.")
            lbl.setWordWrap(True); lbl.setStyleSheet("color: #d8b45a;")
            return lbl
        return None

    def _set(self, key, val):
        spec = next((s for s in self.op.params if s.key == key), None)
        self._values[key] = spec.coerce(val) if spec else val
        self.changed.emit()

    def set_rect(self, rect: dict):
        self._values["rect"] = rect
        self.changed.emit()

    def params(self):
        return dict(self._values)

    def wants_rect(self):
        return self._has_rect


# =============================================================================
# Main panel
# =============================================================================
class LazyDevelopPanel(QWidget):
    """The LazyDevelop tool as an embeddable panel."""

    def __init__(self):
        super().__init__()
        self.doc: Optional[DevelopDocument] = None
        self.worker: Optional[CallableWorker] = None
        self.tool_panel: Optional[ToolPanel] = None
        self._showing_mask: Optional[str] = None
        self._paint_target = "sky"
        self._proxy: Optional[np.ndarray] = None      # cached downscaled current result
        self._edit_proxy: Optional[np.ndarray] = None  # proxy of an edited step's input
        self._preview_kind: Optional[str] = None      # None | "full" | "proxy"

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._do_live_preview)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_tree_pane())      # panel 1: tool tree
        splitter.addWidget(self._build_detail_pane())    # panel 2: selected tool's options
        splitter.addWidget(self._build_view())           # panel 3: canvas + log
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([280, 360, 1060])
        root = QHBoxLayout(self)
        root.addWidget(splitter)
        self._set_enabled_tools(False)

        QShortcut(QKeySequence.Undo, self, activated=self._undo)
        QShortcut(QKeySequence.Redo, self, activated=self._redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self._redo)

    # ------------------------------------------------------------------ layout
    def _build_image_group(self) -> QWidget:
        src = QGroupBox("Image")
        sv = QVBoxLayout(src)
        row = QHBoxLayout()
        self.open_btn = QPushButton("Open…")
        self.save_btn = QPushButton("Save as…")
        self.open_btn.clicked.connect(self._open)
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)
        row.addWidget(self.open_btn); row.addWidget(self.save_btn)
        sv.addLayout(row)
        row2 = QHBoxLayout()
        self.save_recipe_btn = QPushButton("Save recipe…")
        self.load_recipe_btn = QPushButton("Load recipe…")
        self.save_recipe_btn.clicked.connect(self._save_recipe)
        self.load_recipe_btn.clicked.connect(self._load_recipe)
        self.save_recipe_btn.setEnabled(False)
        self.load_recipe_btn.setEnabled(False)
        row2.addWidget(self.save_recipe_btn); row2.addWidget(self.load_recipe_btn)
        sv.addLayout(row2)
        self.info_label = QLabel("No image loaded.")
        self.info_label.setStyleSheet("color: gray;")
        sv.addWidget(self.info_label)
        return src

    def _build_tree_pane(self) -> QWidget:
        """Panel 1: the Image group + a collapsible tree of every tool."""
        col = QVBoxLayout()
        col.setContentsMargins(6, 6, 3, 6)
        col.addWidget(self._build_image_group())

        heading = QLabel("Tools")
        hf = heading.font(); hf.setBold(True); heading.setFont(hf)
        col.addWidget(heading)

        self.toolbox = QTreeWidget()
        self.toolbox.setHeaderHidden(True)
        self.toolbox.setIndentation(12)
        self.toolbox.setAnimated(True)
        self._tool_items: List[QTreeWidgetItem] = []
        for cat, oplist in dev_ops.by_category().items():
            parent = QTreeWidgetItem([cat])
            pf = parent.font(0); pf.setBold(True); parent.setFont(0, pf)
            parent.setFlags(Qt.ItemIsEnabled)           # category row: not selectable
            self.toolbox.addTopLevelItem(parent)
            for op in oplist:
                child = QTreeWidgetItem([op.label])
                child.setToolTip(0, op.tooltip)
                child.setData(0, Qt.UserRole, op.name)
                parent.addChild(child)
                self._tool_items.append(child)
            parent.setExpanded(True)
        self.toolbox.itemClicked.connect(self._tree_item_clicked)
        col.addWidget(self.toolbox, 1)

        holder = QWidget(); holder.setLayout(col)
        holder.setMinimumWidth(240)
        return holder

    def _build_detail_pane(self) -> QWidget:
        """Panel 2: the selected tool's options, plus masks and history."""
        col = QVBoxLayout()
        col.setContentsMargins(3, 6, 6, 6)

        self.tool_holder = QGroupBox("Adjustment")
        self.tool_holder_layout = QVBoxLayout(self.tool_holder)
        self._placeholder = QLabel("Select a tool on the left to adjust it.")
        self._placeholder.setStyleSheet("color: gray;")
        self.tool_holder_layout.addWidget(self._placeholder)
        col.addWidget(self.tool_holder)

        col.addWidget(self._build_mask_group())
        col.addWidget(self._build_history_group())
        col.addStretch(1)

        holder = QWidget(); holder.setLayout(col)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(holder); scroll.setMinimumWidth(320)
        return scroll

    def _tree_item_clicked(self, item, _col=0):
        name = item.data(0, Qt.UserRole)
        if name:
            self._open_tool(dev_ops.get(name))

    def _build_mask_group(self) -> QWidget:
        box = QGroupBox("Masks")
        v = QVBoxLayout(box)
        self.mask_list = QListWidget()
        self.mask_list.setMaximumHeight(90)
        self.mask_list.itemClicked.connect(self._toggle_mask_view)
        v.addWidget(self.mask_list)
        r1 = QHBoxLayout()
        b_lights = QPushButton("Lum lights"); b_darks = QPushButton("Lum darks")
        b_lights.clicked.connect(lambda: self._make_lum_mask(dev_masks.LUM_LIGHTS))
        b_darks.clicked.connect(lambda: self._make_lum_mask(dev_masks.LUM_DARKS))
        r1.addWidget(b_lights); r1.addWidget(b_darks)
        v.addLayout(r1)
        r2 = QHBoxLayout()
        b_bright = QPushButton("Highlights"); b_range = QPushButton("Sky/range")
        b_bright.clicked.connect(self._make_highlights_mask)
        b_range.clicked.connect(self._make_range_mask)
        r2.addWidget(b_bright); r2.addWidget(b_range)
        v.addLayout(r2)
        r3 = QHBoxLayout()
        self.paint_btn = QPushButton("Paint mask…"); self.paint_btn.setCheckable(True)
        self.paint_btn.toggled.connect(self._toggle_paint)
        b_del = QPushButton("Delete")
        b_del.clicked.connect(self._delete_mask)
        r3.addWidget(self.paint_btn); r3.addWidget(b_del)
        v.addLayout(r3)
        self.paint_row = QWidget()
        pr = QHBoxLayout(self.paint_row); pr.setContentsMargins(0, 0, 0, 0)
        self.brush = FloatSlider("Brush", 4, 200, 40, 0)
        self.brush.valueChanged.connect(lambda v: self.canvas.set_brush(int(v)))
        pr.addWidget(self.brush)
        self.save_paint_btn = QPushButton("Save painted → mask")
        self.save_paint_btn.clicked.connect(self._save_painted_mask)
        pr.addWidget(self.save_paint_btn)
        self.paint_row.setVisible(False)
        v.addWidget(self.paint_row)
        return box

    def _build_history_group(self) -> QWidget:
        box = QGroupBox("History")
        v = QVBoxLayout(box)
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(150)
        self.history_list.itemClicked.connect(self._history_clicked)
        v.addWidget(self.history_list)
        hint = QLabel("Click a step to view & edit it; later steps are kept.")
        hint.setStyleSheet("color: gray;"); hint.setWordWrap(True)
        v.addWidget(hint)
        row = QHBoxLayout()
        self.undo_btn = QPushButton("Undo"); self.redo_btn = QPushButton("Redo")
        self.del_step_btn = QPushButton("Delete step")
        self.undo_btn.clicked.connect(self._undo); self.redo_btn.clicked.connect(self._redo)
        self.del_step_btn.clicked.connect(self._delete_selected_step)
        self.del_step_btn.setEnabled(False)
        row.addWidget(self.undo_btn); row.addWidget(self.redo_btn); row.addWidget(self.del_step_btn)
        v.addLayout(row)
        return box

    def _build_view(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.clicked.connect(lambda: self.canvas.fit())
        self.before_btn = QPushButton("Show original"); self.before_btn.setCheckable(True)
        self.before_btn.toggled.connect(self._toggle_before)
        bar.addWidget(self.fit_btn); bar.addWidget(self.before_btn)
        bar.addStretch(1)
        self.status_label = QLabel("")
        bar.addWidget(self.status_label)
        v.addLayout(bar)

        self.canvas = DevelopCanvas()
        self.canvas.rectSelected.connect(self._on_rect)
        v.addWidget(self.canvas, 8)

        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        v.addWidget(self.progress)
        self.log_view = RunLogView()
        v.addWidget(self.log_view, 2)
        return w

    # ----------------------------------------------------------------- loading
    def _open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "",
            "Images (*.tif *.tiff *.png *.fits *.fit *.fts *.xisf);;All files (*)")
        if not path:
            return
        try:
            loaded = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self.doc = DevelopDocument(loaded.data, path=path, header=loaded.header)
        self._cancel_tool()
        self.mask_list.clear()
        h, w = self.doc.base.shape[:2]
        kind = "colour" if self.doc.is_color else "mono"
        self.info_label.setText(f"{Path(path).name}\n{w} × {h}  ({kind})")
        self.save_btn.setEnabled(True)
        self.save_recipe_btn.setEnabled(True)
        self.load_recipe_btn.setEnabled(True)
        self._set_enabled_tools(True)
        self._refresh_canvas(fit=True)
        self._refresh_history()
        self.log_view.clear()
        self.log_view.append(f"Opened {Path(path).name}")

    def _save(self):
        if self.doc is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", "", "TIFF (*.tif);;PNG (*.png);;FITS (*.fits)")
        if not path:
            return
        try:
            save_image(path, np.asarray(self.doc.result(), dtype=np.float64), bit_depth=16)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.log_view.append(f"Saved {Path(path).name}")
        self.status_label.setText(f"Saved {Path(path).name}")

    def _save_recipe(self):
        if self.doc is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save recipe", "", "Recipe (*.ldrecipe *.json)")
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self.doc.to_recipe(), indent=2))
        except Exception as exc:
            QMessageBox.critical(self, "Save recipe failed", str(exc))
            return
        self.log_view.append(f"Saved recipe {Path(path).name} ({len(self.doc.ops)} steps)")

    def _load_recipe(self):
        if self.doc is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load recipe", "", "Recipe (*.ldrecipe *.json);;All files (*)")
        if not path:
            return
        try:
            recipe = json.loads(Path(path).read_text())
            self.doc.apply_recipe(recipe)
        except Exception as exc:
            QMessageBox.critical(self, "Load recipe failed", str(exc))
            return
        self._cancel_tool(refresh=False)
        self._refresh_canvas()
        self._refresh_history()
        self.log_view.append(f"Applied recipe {Path(path).name} ({len(recipe)} steps)")

    # ------------------------------------------------------------------- tools
    def _select_tree_tool(self, name: str):
        """Highlight a tool's leaf in the tree (does not fire itemClicked)."""
        for it in self._tool_items:
            if it.data(0, Qt.UserRole) == name:
                self.toolbox.setCurrentItem(it)
                return

    def _mount_tool_panel(self, panel: "ToolPanel", op: "dev_ops.Op", rect_default: dict):
        self.tool_panel = panel
        panel.changed.connect(self._schedule_preview)
        panel.applied.connect(self._apply_tool)
        panel.cancelled.connect(self._cancel_tool)
        panel.deleted.connect(self._delete_current_step)
        self.tool_holder_layout.addWidget(panel)
        self._placeholder.setVisible(False)
        self._select_tree_tool(op.name)
        if panel.wants_rect():
            # A crop's rectangle is fractions of its INPUT. Show that input (not a cropped
            # preview) so the rubber-band maps to the right pixels, both for a new crop
            # (current view) and when editing an existing crop step (its cached input).
            base = (self.doc.input_before(panel.edit_index)
                    if panel.edit_index is not None else self.doc.result())
            self.canvas.set_image(np.asarray(base, dtype=np.float32), keep_view=True)
            self._preview_kind = None
            self.canvas.set_rect_mode(True)
            self.canvas.show_rect(rect_default)
        self._schedule_preview()

    def _open_tool(self, op: "dev_ops.Op"):
        """Start a NEW adjustment (inserted at the current history position on Apply)."""
        if self.doc is None or self._busy():
            return
        if op.needs_color and not self.doc.is_color:
            self.status_label.setText(f"{op.label} needs a colour image.")
            return
        self._cancel_tool(refresh=False)
        rect = op.defaults().get("rect", {"x0": 0, "y0": 0, "x1": 1, "y1": 1})
        self._mount_tool_panel(ToolPanel(op, self.doc.mask_names()), op, rect)

    def _open_tool_for_edit(self, index: int):
        """Open the tool for an EXISTING step, populated with its stored values."""
        if self.doc is None or self._busy() or not (0 <= index < len(self.doc.ops)):
            return
        op = dev_ops.get(self.doc.ops[index].name)
        self._cancel_tool(refresh=False)
        init = self.doc.op_params(index)
        rect = init["params"].get("rect", {"x0": 0, "y0": 0, "x1": 1, "y1": 1})
        self._mount_tool_panel(
            ToolPanel(op, self.doc.mask_names(), edit_index=index, init=init), op, rect)

    def _cancel_tool(self, refresh=True):
        self.canvas.set_rect_mode(False)
        self._edit_proxy = None
        if self.tool_panel is not None:
            self.tool_panel.setParent(None)
            self.tool_panel.deleteLater()
            self.tool_panel = None
        self._placeholder.setVisible(True)
        if refresh and self.doc is not None:
            self._refresh_canvas()

    def _on_rect(self, rect: dict):
        if self.tool_panel is not None and self.tool_panel.wants_rect():
            self.tool_panel.set_rect(rect)

    def _schedule_preview(self):
        if self.tool_panel is None:
            return
        self._preview_timer.start()

    def _get_proxy(self) -> np.ndarray:
        """A cached, downscaled copy of the current committed result (for heavy previews)."""
        if self._proxy is None and self.doc is not None:
            self._proxy = _downscale_rgb(self.doc.result())
        return self._proxy

    def _do_live_preview(self):
        if self.tool_panel is None or self.doc is None or self._busy():
            return
        op = self.tool_panel.op
        params = self.tool_panel.params()
        gate = self.tool_panel.gate()
        idx = self.tool_panel.edit_index
        editing = idx is not None
        if self.tool_panel.wants_rect():
            # Crop: the canvas keeps showing the input + rubber-band overlay; the crop is
            # only applied on Apply/Update (previewing it would break the rect mapping).
            self.status_label.setText(f"{op.label}: drag a rectangle, then "
                                      f"{'Update' if editing else 'Apply'}.")
            return
        try:
            if editing:
                # Editing an existing step: preview the op re-applied to its own input
                # (heavy tools on a downscaled proxy of that input).
                if op.heavy:
                    if self._edit_proxy is None:
                        self._edit_proxy = _downscale_rgb(self.doc.input_before(idx))
                    out = self.doc.preview_replace(idx, op.name, params,
                                                   base=self._edit_proxy, **gate)
                    kind = "proxy"
                else:
                    out = self.doc.preview_replace(idx, op.name, params, **gate)
                    kind = "full"
            elif op.heavy:
                # New heavy adjustment: preview on a downscaled proxy so sliders stay
                # responsive; Apply still recomputes at full resolution (see _apply_tool).
                proxy = self._get_proxy()
                pdoc = DevelopDocument(proxy)
                pdoc.masks = self.doc.masks            # _blend resizes masks to proxy size
                out = pdoc.preview_op(op.name, params, **gate)
                kind = "proxy"
            else:
                out = self.doc.preview_op(op.name, params, **gate)
                kind = "full"
        except Exception as exc:
            self.status_label.setText(f"{op.label}: {exc}")
            return
        # Clear "Show original" without re-entering _refresh_canvas (which would reset the
        # preview resolution tracking and re-fit the view).
        self.before_btn.blockSignals(True)
        self.before_btn.setChecked(False)
        self.before_btn.blockSignals(False)
        # Fit the first time we switch preview resolution (proxy↔full); keep zoom after.
        keep = (kind == self._preview_kind)
        self._preview_kind = kind
        self.canvas.set_image(np.asarray(out, dtype=np.float32), keep_view=keep)
        verb = "Editing" if editing else "Previewing"
        suffix = "  (proxy — Apply computes full resolution)" if kind == "proxy" else ""
        self.status_label.setText(f"{verb} {op.label}{suffix}")

    def _apply_is_heavy(self, op: "dev_ops.Op", idx: Optional[int]) -> bool:
        """Whether committing this op should run in the worker (off the UI thread).

        True if the op itself is heavy OR the recompute it triggers re-runs any heavy
        KEPT downstream op: a replace re-runs ops[idx+1:]; an insert re-runs ops[cursor:].
        """
        start = idx + 1 if idx is not None else self.doc.cursor
        downstream_heavy = any(dev_ops.get(o.name).heavy for o in self.doc.ops[start:])
        return op.heavy or downstream_heavy

    def _apply_tool(self):
        if self.tool_panel is None or self.doc is None or self._busy():
            return
        op = self.tool_panel.op
        params = self.tool_panel.params()
        gate = self.tool_panel.gate()
        idx = self.tool_panel.edit_index

        def commit():
            if idx is None:
                self.doc.apply_op(op.name, params, **gate)      # insert at cursor
            else:
                self.doc.replace_op(idx, params, **gate)        # edit in place + recompute

        verb = "Updated" if idx is not None else "Applied"
        heavy = self._apply_is_heavy(op, idx)
        if not heavy:
            commit()
            self._finish_apply(verb, op.label)
            return
        self._set_busy(True)
        self.status_label.setText(f"{'Updating' if idx is not None else 'Applying'} {op.label}…")

        def fn(log, progress):
            log(f"{op.label}…")
            commit()
            return {"label": op.label, "verb": verb}

        self.worker = CallableWorker(fn, mode="apply")
        self.worker.logline.connect(self.log_view.appendPlainText)
        self.worker.finished_ok.connect(lambda r: self._finish_apply(r["verb"], r["label"]))
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _finish_apply(self, verb, label):
        self._set_busy(False)
        self.log_view.append(f"{verb} {label}")
        self._cancel_tool(refresh=False)
        self.toolbox.clearSelection()
        self._refresh_canvas()
        self._refresh_history()

    def _delete_current_step(self):
        if self.tool_panel is None or self.tool_panel.edit_index is None or self.doc is None \
                or self._busy():
            return
        idx = self.tool_panel.edit_index
        self.doc.delete_op(idx)
        self.log_view.append(f"Deleted step {idx + 1}")
        self._cancel_tool(refresh=False)
        self.toolbox.clearSelection()
        self._refresh_canvas()
        self._refresh_history()

    def _on_failed(self, msg):
        self._set_busy(False)
        if self.doc is not None:
            self._refresh_history()
        QMessageBox.critical(self, "Tool failed", msg)

    # ----------------------------------------------------------------- history
    def _refresh_history(self):
        self.history_list.blockSignals(True)
        self.history_list.clear()
        self.history_list.addItem(QListWidgetItem("• Base image"))
        for i, op in enumerate(self.doc.ops):
            it = QListWidgetItem(f"{i + 1}. {op.title()}")
            if (i + 1) > self.doc.cursor:              # a "future" step: kept, not applied now
                it.setForeground(QColor(150, 150, 150))
            self.history_list.addItem(it)
        self.history_list.setCurrentRow(self.doc.cursor)
        self.history_list.blockSignals(False)
        self.undo_btn.setEnabled(self.doc.can_undo())
        self.redo_btn.setEnabled(self.doc.can_redo())
        self.del_step_btn.setEnabled(len(self.doc.ops) > 0)

    def _history_clicked(self, item):
        """Navigate to a step (keeping the rest) and open it for editing."""
        if self.doc is None or self._busy():
            return
        row = self.history_list.row(item)              # 0 = base, k = after op k
        self.doc.goto(row)
        self._cancel_tool(refresh=False)
        self._refresh_canvas()                          # show the state at this position
        if row >= 1:
            self._open_tool_for_edit(row - 1)
        else:
            self.toolbox.clearSelection()
        self._refresh_history()

    def _delete_selected_step(self):
        if self.doc is None or self._busy():
            return
        row = self.history_list.currentRow()
        if row >= 1:
            self.doc.delete_op(row - 1)
            self.log_view.append(f"Deleted step {row}")
            self._cancel_tool(refresh=False)
            self.toolbox.clearSelection()
            self._refresh_canvas()
            self._refresh_history()

    def _undo(self):
        if self.doc and not self._busy() and self.doc.undo():
            self._cancel_tool(refresh=False)
            self._refresh_canvas(); self._refresh_history()

    def _redo(self):
        if self.doc and not self._busy() and self.doc.redo():
            self._cancel_tool(refresh=False)
            self._refresh_canvas(); self._refresh_history()

    # ------------------------------------------------------------------- masks
    def _make_lum_mask(self, kind):
        if self.doc is None:
            return
        m = dev_masks.luminosity_mask(self.doc.result(), kind, depth=2)
        name = self._unique_mask_name(f"Lum {kind}")
        self.doc.add_mask(name, m)
        self._refresh_masks()

    def _make_highlights_mask(self):
        if self.doc is None:
            return
        m = dev_masks.highlights_mask(self.doc.result(), 0.6)
        self.doc.add_mask(self._unique_mask_name("Highlights"), m)
        self._refresh_masks()

    def _make_range_mask(self):
        if self.doc is None:
            return
        m = dev_masks.range_mask(self.doc.result(), 0.0, 0.4)
        self.doc.add_mask(self._unique_mask_name("Sky"), m)
        self._refresh_masks()

    def _unique_mask_name(self, base):
        name = base; i = 2
        while name in self.doc.masks:
            name = f"{base} {i}"; i += 1
        return name

    def _refresh_masks(self):
        self.mask_list.clear()
        for n in self.doc.mask_names():
            self.mask_list.addItem(QListWidgetItem(n))
        if self.tool_panel is not None:
            self.tool_panel.refresh_masks(self.doc.mask_names())

    def _toggle_mask_view(self, item):
        name = item.text()
        if self._showing_mask == name:
            self._showing_mask = None
            self._refresh_canvas()
            self.status_label.setText("")
        else:
            self._showing_mask = name
            m = self.doc.masks.get(name)
            if m is not None:
                self.canvas.set_image(np.asarray(m, dtype=np.float32), keep_view=True)
                self.status_label.setText(f"Viewing mask: {name}")

    def _delete_mask(self):
        item = self.mask_list.currentItem()
        if item is None or self.doc is None:
            return
        self.doc.remove_mask(item.text())
        if self._showing_mask == item.text():
            self._showing_mask = None
            self._refresh_canvas()
        self._refresh_masks()

    def _toggle_paint(self, on):
        self.paint_row.setVisible(on)
        if on:
            self.canvas.set_brush(int(self.brush.value()))
            self.canvas.set_paint_mode("sky")
            self.status_label.setText("Paint the region to select, then Save painted → mask.")
        else:
            self.canvas.set_paint_mode(None)
            self._refresh_canvas()

    def _save_painted_mask(self):
        if self.doc is None:
            return
        sc = self.canvas.scribbles()
        if sc is None or not np.any(sc):
            QMessageBox.information(self, "Paint mask", "Paint on the image first.")
            return
        m = dev_masks.scribble_to_mask(sc, positive="sky")
        self.doc.add_mask(self._unique_mask_name("Painted"), m)
        self.canvas.clear_scribbles()
        self.paint_btn.setChecked(False)
        self._refresh_masks()

    # ------------------------------------------------------------------ canvas
    def _refresh_canvas(self, fit=False):
        if self.doc is None:
            return
        self._showing_mask = None
        self._proxy = None                 # committed result changed → rebuild proxy lazily
        self._preview_kind = None
        self.canvas.set_image(np.asarray(self.doc.result(), dtype=np.float32),
                              keep_view=not fit)

    def _toggle_before(self, on):
        if self.doc is None:
            return
        if on:
            self.canvas.set_image(np.asarray(self.doc.base, dtype=np.float32), keep_view=True)
        else:
            self._refresh_canvas()

    # ------------------------------------------------------------------ enable
    def _set_enabled_tools(self, on):
        self.toolbox.setEnabled(on)
        self.tool_holder.setEnabled(on)

    def _busy(self):
        return self.worker is not None and self.worker.isRunning()

    def _set_busy(self, busy):
        self.progress.setRange(0, 0) if busy else self.progress.setRange(0, 100)
        if not busy:
            self.progress.setValue(0)
        # While the worker mutates the document off-thread, lock every control that could
        # also read/mutate it on the main thread (avoids a cross-thread cache race).
        for b in (self.open_btn, self.save_btn, self.save_recipe_btn, self.load_recipe_btn,
                  self.toolbox, self.tool_holder, self.history_list):
            b.setEnabled(not busy)
        if busy:
            self.undo_btn.setEnabled(False)
            self.redo_btn.setEnabled(False)
            self.del_step_btn.setEnabled(False)
        elif self.doc is not None:
            self.undo_btn.setEnabled(self.doc.can_undo())
            self.redo_btn.setEnabled(self.doc.can_redo())
            self.del_step_btn.setEnabled(len(self.doc.ops) > 0)
