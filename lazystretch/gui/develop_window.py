"""The LazyDevelop desktop panel — interactive image finishing (hosted by gui/shell.py).

Left column: source + a categorised toolbox (built generically from the op registry) +
the active tool's controls (with a mask/opacity gate) + a mask panel + the op history.
Right: the image canvas (rubber-band crop, mask painting, before/after) + a log.

Heavy tools (wavelets, HDR, ABE, NR) run in a background worker; light tools preview
live on the full-resolution image.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, QRectF, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
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
from .preview import FullScreenViewer, PreviewView
from .widgets import FloatSlider, LogExportMixin
from .worker import CallableWorker


class DevelopLog(LogExportMixin, QTreeWidget):
    """A simple 3-column edit log: absolute Time · Event · Value (the applied values).

    Unlike the run-log used by the batch tools, Develop logs discrete edits, so the
    third column carries each step's actual values rather than a duration.
    """

    log_title = "LazyDevelop log"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHeaderLabels(["Time", "Event", "Value"])
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self.header().setStretchLastSection(True)
        self.setColumnWidth(0, 74)
        self.setColumnWidth(1, 190)

    def add(self, event: str, value: str = "") -> None:
        item = QTreeWidgetItem([datetime.now().strftime("%H:%M:%S"), str(event), str(value)])
        self.addTopLevelItem(item)
        self.scrollToBottom()

    # QPlainTextEdit / RunLogView-compatible aliases used by existing call sites.
    def append(self, msg: str) -> None:
        self.add(msg)

    def appendPlainText(self, msg: str) -> None:
        self.add(msg)

    def finish(self) -> None:
        pass

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
        self._rect_label: Optional[QGraphicsSimpleTextItem] = None
        self._rect_start = None
        self._rect: Optional[dict] = None          # current selection (normalised)
        self._drag_mode = None                     # 'new' | 'move' | 'resize'
        self._drag_edges: set = set()              # subset of {'l','r','t','b'} for resize
        self._move_anchor = None                   # (px, py) where a move drag began
        self._move_rect0: Optional[dict] = None    # the rect at move-drag start
        self._stars_mode = False                   # star-spike marker editing
        self._stars: list = []                     # marked stars (normalised dicts)
        self._active = -1                          # active star (length slider edits it)
        self._star_drag = -1                       # index being dragged, or -1
        self._star_items: list = []                # QGraphics overlay items to clear/redraw
        self._new_len = 0.06                       # length assigned to newly-added stars
        self._preview_count = 4                    # schematic spike count/angle for the overlay
        self._preview_angle = 0.0
        self._preview_jitter = 0.0
        self.setMouseTracking(True)                # hover cursors without a pressed button

    def set_rect_mode(self, on: bool):
        self._rect_mode = bool(on)
        if on:
            self.setDragMode(PreviewView.NoDrag)
        else:
            self.setDragMode(PreviewView.ScrollHandDrag)
            self._clear_rect_item()
            self.unsetCursor()

    def _clear_rect_item(self):
        for attr in ("_rect_item", "_rect_label"):
            item = getattr(self, attr)
            if item is not None and item.scene() is not None:
                self.scene().removeItem(item)
            setattr(self, attr, None)
        self._rect = None
        self._drag_mode = None
        self._rect_start = None

    def show_rect(self, rect: dict):
        arr = self.current_array()
        if arr is None:
            return
        self._rect = dict(rect)                     # remember it so it can be dragged/refined
        h, w = arr.shape[:2]
        x0 = rect.get("x0", 0.0) * w; x1 = rect.get("x1", 1.0) * w
        y0 = rect.get("y0", 0.0) * h; y1 = rect.get("y1", 1.0) * h
        if self._rect_item is None:
            self._rect_item = self.scene().addRect(QRectF(x0, y0, x1 - x0, y1 - y0),
                                                   QPen(QColor(255, 210, 90), 0))
            self._rect_item.setZValue(5)
        self._rect_item.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))

        # Live pixel size of the selection, drawn at the rect's top-left at a constant
        # on-screen size (ignores zoom), like Lightroom's crop overlay.
        px_w = max(0, int(round((x1 - x0))))
        px_h = max(0, int(round((y1 - y0))))
        if self._rect_label is None:
            self._rect_label = self.scene().addSimpleText("")
            self._rect_label.setBrush(QBrush(QColor(255, 210, 90)))
            self._rect_label.setPen(QPen(QColor(0, 0, 0, 200), 0))   # thin dark outline
            font = QFont(); font.setPointSize(11); font.setBold(True)
            self._rect_label.setFont(font)
            self._rect_label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            self._rect_label.setZValue(6)
        self._rect_label.setText(f"{px_w} × {px_h} px")
        self._rect_label.setPos(min(x0, x1) + 6 / max(self.transform().m11(), 1e-6),
                                min(y0, y1) + 6 / max(self.transform().m22(), 1e-6))

    # --- crop rectangle: draw a new one, or drag an existing one to refine it -----
    def _px_bounds(self):
        """Current selection in pixel coords (x0, y0, x1, y1), or None."""
        arr = self.current_array()
        if arr is None or self._rect is None:
            return None
        h, w = arr.shape[:2]
        return (self._rect["x0"] * w, self._rect["y0"] * h,
                self._rect["x1"] * w, self._rect["y1"] * h)

    def _hit_test(self, px, py):
        """Classify a click at pixel (px, py): ('resize', edges) on a handle, ('move', ..)
        inside a partial rect, else ('new', ..) to start a fresh rubber-band."""
        b = self._px_bounds()
        if b is None:
            return "new", set()
        x0, y0, x1, y1 = b
        tol = 10.0 / max(self.transform().m11(), 1e-6)      # ~10 screen px, zoom-independent
        edges = set()
        if abs(px - x0) <= tol and y0 - tol <= py <= y1 + tol:
            edges.add("l")
        if abs(px - x1) <= tol and y0 - tol <= py <= y1 + tol:
            edges.add("r")
        if abs(py - y0) <= tol and x0 - tol <= px <= x1 + tol:
            edges.add("t")
        if abs(py - y1) <= tol and x0 - tol <= px <= x1 + tol:
            edges.add("b")
        if edges:
            return "resize", edges
        arr = self.current_array()
        h, w = arr.shape[:2]
        full = (self._rect["x0"] <= 1e-3 and self._rect["y0"] <= 1e-3
                and self._rect["x1"] >= 1 - 1e-3 and self._rect["y1"] >= 1 - 1e-3)
        if not full and x0 <= px <= x1 and y0 <= py <= y1:
            return "move", set()
        return "new", set()                                 # empty area (or full frame) → draw new

    def _set_rect_px(self, x0, y0, x1, y1, w, h):
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        rect = {"x0": max(0.0, min(1.0, x0 / w)), "y0": max(0.0, min(1.0, y0 / h)),
                "x1": max(0.0, min(1.0, x1 / w)), "y1": max(0.0, min(1.0, y1 / h))}
        self.show_rect(rect)
        self.rectSelected.emit(rect)

    def _apply_resize(self, px, py, w, h):
        x0, y0, x1, y1 = self._rect["x0"] * w, self._rect["y0"] * h, \
            self._rect["x1"] * w, self._rect["y1"] * h
        if "l" in self._drag_edges: x0 = px
        if "r" in self._drag_edges: x1 = px
        if "t" in self._drag_edges: y0 = py
        if "b" in self._drag_edges: y1 = py
        self._set_rect_px(x0, y0, x1, y1, w, h)

    def _apply_move(self, px, py, w, h):
        dx = px - self._move_anchor[0]
        dy = py - self._move_anchor[1]
        r0 = self._move_rect0
        x0 = r0["x0"] * w + dx; x1 = r0["x1"] * w + dx
        y0 = r0["y0"] * h + dy; y1 = r0["y1"] * h + dy
        rw, rh = x1 - x0, y1 - y0                            # translate-clamp: keep the size
        if x0 < 0: x0, x1 = 0, rw
        if x1 > w: x0, x1 = w - rw, w
        if y0 < 0: y0, y1 = 0, rh
        if y1 > h: y0, y1 = h - rh, h
        self._set_rect_px(x0, y0, x1, y1, w, h)

    def _cursor_for(self, mode, edges):
        if mode == "resize":
            if edges in ({"l", "t"}, {"r", "b"}):
                return Qt.SizeFDiagCursor
            if edges in ({"r", "t"}, {"l", "b"}):
                return Qt.SizeBDiagCursor
            if edges & {"l", "r"}:
                return Qt.SizeHorCursor
            return Qt.SizeVerCursor
        if mode == "move":
            return Qt.SizeAllCursor
        return Qt.CrossCursor

    def mousePressEvent(self, e):
        if self._stars_mode and self.current_array() is not None:
            if self._stars_press(e):
                e.accept(); return
        if self._rect_mode and e.button() == Qt.LeftButton and self.current_array() is not None:
            pt = self.mapToScene(e.position().toPoint())
            mode, edges = self._hit_test(pt.x(), pt.y())
            if mode == "resize":
                self._drag_mode = "resize"; self._drag_edges = edges
            elif mode == "move":
                self._drag_mode = "move"
                self._move_anchor = (pt.x(), pt.y()); self._move_rect0 = dict(self._rect)
            else:
                self._drag_mode = "new"; self._rect_start = pt
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._stars_mode and self.current_array() is not None:
            if self._stars_move(e):
                e.accept(); return
            super().mouseMoveEvent(e); return
        if self._rect_mode and self.current_array() is not None:
            pt = self.mapToScene(e.position().toPoint())
            if self._drag_mode and (e.buttons() & Qt.LeftButton):
                h, w = self.current_array().shape[:2]
                if self._drag_mode == "new":
                    self._set_rect_px(self._rect_start.x(), self._rect_start.y(),
                                      pt.x(), pt.y(), w, h)
                elif self._drag_mode == "resize":
                    self._apply_resize(pt.x(), pt.y(), w, h)
                else:
                    self._apply_move(pt.x(), pt.y(), w, h)
                e.accept()
                return
            if not (e.buttons() & Qt.LeftButton):           # hover: show the affordance
                self.setCursor(self._cursor_for(*self._hit_test(pt.x(), pt.y())))
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._stars_mode:
            if self._stars_release(e):
                e.accept(); return
        if self._rect_mode and self._drag_mode:
            pt = self.mapToScene(e.position().toPoint())
            h, w = self.current_array().shape[:2]
            if self._drag_mode == "new":
                self._set_rect_px(self._rect_start.x(), self._rect_start.y(),
                                  pt.x(), pt.y(), w, h)
            elif self._drag_mode == "resize":
                self._apply_resize(pt.x(), pt.y(), w, h)
            else:
                self._apply_move(pt.x(), pt.y(), w, h)
            self._drag_mode = None; self._rect_start = None
            self._drag_edges = set(); self._move_anchor = None
            e.accept()
            return
        super().mouseReleaseEvent(e)

    # --- star-spike markers: auto-marked stars the user can select/add/move/remove ----
    starsChanged = Signal(list)                 # the star list changed (add/move/remove/len)
    starActivated = Signal(int)                 # a star became active (-1 = none)

    def set_stars_mode(self, on: bool):
        self._stars_mode = bool(on)
        if on:
            self.setDragMode(PreviewView.NoDrag)
        else:
            self.setDragMode(PreviewView.ScrollHandDrag)
            self._clear_star_items()
            self._stars = []
            self._active = -1
            self._star_drag = -1
            self.unsetCursor()

    def _clear_star_items(self):
        for it in getattr(self, "_star_items", []):
            if it.scene() is not None:
                self.scene().removeItem(it)
        self._star_items = []

    def set_stars(self, stars: list, active: int = -1):
        self._stars = [dict(s) for s in stars]
        self._active = active if -1 <= active < len(self._stars) else -1
        self._redraw_stars()

    def set_new_len(self, v: float):
        self._new_len = float(v)

    def set_preview(self, count: int, angle_deg: float, jitter: float = 0.0):
        self._preview_count = int(count)
        self._preview_angle = float(angle_deg)
        self._preview_jitter = float(jitter)
        self._redraw_stars()

    def set_active_len(self, v: float):
        if 0 <= self._active < len(self._stars):
            self._stars[self._active]["len"] = float(v)
            self._redraw_stars()
            self.starsChanged.emit([dict(s) for s in self._stars])
        self._new_len = float(v)

    def _redraw_stars(self):
        self._clear_star_items()
        arr = self.current_array()
        if arr is None or not self._stars_mode:
            return
        h, w = arr.shape[:2]
        diag = float(np.hypot(h, w))
        r = 7.0 / max(self.transform().m11(), 1e-6)        # marker radius, zoom-independent
        n = max(int(getattr(self, "_preview_count", 4)), 3)
        a0 = np.radians(getattr(self, "_preview_angle", 0.0))
        jit = float(getattr(self, "_preview_jitter", 0.0))
        for i, s in enumerate(self._stars):
            cx = s["x"] * (w - 1); cy = s["y"] * (h - 1)
            active = (i == self._active)
            col = QColor(255, 210, 90) if active else QColor(120, 200, 255)
            length = float(s.get("len", getattr(self, "_new_len", 0.06))) * diag
            for k in range(n):                             # schematic spike overlay (live)
                ang = a0 + k * 2.0 * np.pi / n
                lk = length
                if jit > 0:                                # match render's deterministic jitter
                    hsh = (np.sin(s["x"] * 127.1 + s["y"] * 311.7 + k * 74.7)
                           * 43758.5453) % 1.0
                    lk = length * (1.0 + jit * (2.0 * hsh - 1.0) * 0.5)
                line = self.scene().addLine(cx, cy, cx + np.cos(ang) * lk,
                                            cy + np.sin(ang) * lk,
                                            QPen(col, 0))
                line.setOpacity(0.9 if active else 0.55)
                line.setZValue(5)
                self._star_items.append(line)
            dot = self.scene().addEllipse(cx - r, cy - r, 2 * r, 2 * r, QPen(col, 0))
            dot.setZValue(6)
            self._star_items.append(dot)

    def _nearest_star(self, px, py):
        arr = self.current_array()
        if arr is None or not self._stars:
            return -1
        h, w = arr.shape[:2]
        tol = 12.0 / max(self.transform().m11(), 1e-6)
        best, bd = -1, tol * tol
        for i, s in enumerate(self._stars):
            dx = s["x"] * (w - 1) - px; dy = s["y"] * (h - 1) - py
            d = dx * dx + dy * dy
            if d <= bd:
                best, bd = i, d
        return best

    def _add_star(self, px, py, w, h):
        arr = self.current_array()
        col = arr[int(np.clip(py, 0, h - 1)), int(np.clip(px, 0, w - 1))]
        col = [float(col[0]), float(col[1]), float(col[2])] if arr.ndim == 3 else [1.0, 1.0, 1.0]
        star = {"x": float(np.clip(px / max(w - 1, 1), 0, 1)),
                "y": float(np.clip(py / max(h - 1, 1), 0, 1)),
                "len": float(getattr(self, "_new_len", 0.06)),
                "flux": 0.8, "col": col}
        self._stars.append(star)
        self._active = len(self._stars) - 1
        self._redraw_stars()
        self.starsChanged.emit([dict(s) for s in self._stars])
        self.starActivated.emit(self._active)

    def _stars_press(self, e) -> bool:
        pt = self.mapToScene(e.position().toPoint())
        h, w = self.current_array().shape[:2]
        i = self._nearest_star(pt.x(), pt.y())
        if e.button() == Qt.RightButton:                   # right-click removes a star
            if i >= 0:
                del self._stars[i]
                self._active = -1
                self._redraw_stars()
                self.starsChanged.emit([dict(s) for s in self._stars])
                self.starActivated.emit(-1)
            return True
        if e.button() == Qt.LeftButton:
            if i >= 0:                                     # select + start moving it
                self._active = i; self._star_drag = i
                self._redraw_stars()
                self.starActivated.emit(i)
            else:                                          # empty → add a star here
                self._add_star(pt.x(), pt.y(), w, h)
                self._star_drag = self._active
            return True
        return False

    def _stars_move(self, e) -> bool:
        pt = self.mapToScene(e.position().toPoint())
        h, w = self.current_array().shape[:2]
        if self._star_drag >= 0 and (e.buttons() & Qt.LeftButton):
            self._stars[self._star_drag]["x"] = float(np.clip(pt.x() / max(w - 1, 1), 0, 1))
            self._stars[self._star_drag]["y"] = float(np.clip(pt.y() / max(h - 1, 1), 0, 1))
            self._redraw_stars()
            self.starsChanged.emit([dict(s) for s in self._stars])
            return True
        if not (e.buttons() & Qt.LeftButton):              # hover cursor
            over = self._nearest_star(pt.x(), pt.y()) >= 0
            self.setCursor(Qt.PointingHandCursor if over else Qt.CrossCursor)
        return False

    def _stars_release(self, e) -> bool:
        if self._star_drag >= 0:
            self._star_drag = -1
            return True
        return False


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
    starsRedetect = Signal()               # "Auto-detect" pressed (star-spikes tool)
    starsClear = Signal()                  # "Clear all" pressed (star-spikes tool)

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
        self._has_stars = any(s.kind == "stars" for s in op.params)
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
            self._values[spec.key] = (spec.coerce(val)
                                      if spec.kind not in ("curve", "rect", "stars") else val)
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
        if spec.kind == "stars":
            box = QWidget(); bl = QVBoxLayout(box); bl.setContentsMargins(0, 0, 0, 0)
            hint = QLabel("Click a star to select, click empty to add, drag to move, "
                          "right-click to remove.")
            hint.setWordWrap(True); hint.setStyleSheet("color: #d8b45a;")
            bl.addWidget(hint)
            r = QHBoxLayout()
            b1 = QPushButton("Auto-detect"); b1.clicked.connect(self.starsRedetect.emit)
            b2 = QPushButton("Clear all"); b2.clicked.connect(self.starsClear.emit)
            r.addWidget(b1); r.addWidget(b2)
            bl.addLayout(r)
            return box
        return None

    def _set(self, key, val):
        spec = next((s for s in self.op.params if s.key == key), None)
        self._values[key] = spec.coerce(val) if spec else val
        self.changed.emit()

    def set_rect(self, rect: dict):
        self._values["rect"] = rect
        self.changed.emit()

    def set_stars(self, stars: list):
        """Store the star list without a re-preview (the canvas overlay is the preview)."""
        self._values["stars"] = [dict(s) for s in stars]

    def control(self, key: str):
        return self._widgets.get(key)

    def params(self):
        return dict(self._values)

    def wants_rect(self):
        return self._has_rect

    def wants_stars(self):
        return self._has_stars


# =============================================================================
# Auto-develop suggestion panel
# =============================================================================
class AutoPanel(QWidget):
    """A checklist of analyser-suggested steps, in finishing order, with reasons."""

    changed = Signal()
    applied = Signal()
    cancelled = Signal()

    def __init__(self, steps: List[dict], parent=None):
        super().__init__(parent)
        self.steps = steps
        self._checks: List = []
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Auto-develop")
        f = title.font(); f.setBold(True); f.setPointSize(f.pointSize() + 1)
        title.setFont(f)
        v.addWidget(title)
        hint = QLabel("Analysed the full image. Untick anything you don't want, then Apply all.")
        hint.setWordWrap(True); hint.setStyleSheet("color: gray;")
        v.addWidget(hint)

        for s in steps:
            op = dev_ops.get(s["name"])
            gate = ""
            if s.get("mask"):
                gate = f"  ·  {'except ' if s.get('mask_invert') else 'in '}{s['mask']}"
            cb = QCheckBox(f"{op.label} — {s['reason']}{gate}")
            cb.setChecked(True)
            cb.toggled.connect(lambda _=False: self.changed.emit())
            self._checks.append((cb, s))
            v.addWidget(cb)

        row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply all")
        self.cancel_btn = QPushButton("Cancel")
        self.apply_btn.clicked.connect(self.applied.emit)
        self.cancel_btn.clicked.connect(self.cancelled.emit)
        row.addWidget(self.apply_btn); row.addWidget(self.cancel_btn)
        v.addLayout(row)

    def selected_steps(self) -> List[dict]:
        return [s for cb, s in self._checks if cb.isChecked()]


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
        self.auto_panel: Optional["AutoPanel"] = None
        self._showing_mask: Optional[str] = None
        self._paint_target = "sky"
        self._proxy: Optional[np.ndarray] = None      # cached downscaled current result
        self._edit_proxy: Optional[np.ndarray] = None  # proxy of an edited step's input
        self._auto_mask_map: dict = {}                # canonical semantic name → installed name
        self._preview_kind: Optional[str] = None      # None | "full" | "proxy"
        self._fs_viewer: Optional[FullScreenViewer] = None  # kept alive while full-screen

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._do_live_preview)

        self._auto_timer = QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.setInterval(180)
        self._auto_timer.timeout.connect(self._auto_preview)

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
        self.auto_btn = QPushButton("✨  Auto-develop…")
        self.auto_btn.setToolTip("Analyse the whole image and suggest a finishing recipe.")
        self.auto_btn.clicked.connect(self._open_auto)
        self.auto_btn.setEnabled(False)
        sv.addWidget(self.auto_btn)
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
        self.mask_group = box
        v = QVBoxLayout(box)
        hint = QLabel("A library any tool can gate on. Double-click to rename.")
        hint.setStyleSheet("color: gray;"); hint.setWordWrap(True)
        v.addWidget(hint)
        self.mask_list = QListWidget()
        self.mask_list.setMaximumHeight(90)
        self.mask_list.itemClicked.connect(self._toggle_mask_view)
        self.mask_list.itemDoubleClicked.connect(self._rename_mask)
        v.addWidget(self.mask_list)
        self.auto_masks_btn = QPushButton("✨  Auto masks (sky · stars · nebula · dust · cores)")
        self.auto_masks_btn.setToolTip("Analyse the image and generate a set of semantic masks.")
        self.auto_masks_btn.clicked.connect(self._auto_masks)
        v.addWidget(self.auto_masks_btn)
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
        # Combine masks into a reusable composite (e.g. Nebula ∩ ¬Bright stars).
        cr = QHBoxLayout()
        self.combine_a = QComboBox(); self.combine_b = QComboBox()
        self.combine_op = QComboBox()
        self.combine_op.addItems(["∩ intersect", "∪ union", "− subtract", "¬ invert A"])
        self.combine_op.currentIndexChanged.connect(self._update_combine_enabled)
        cr.addWidget(self.combine_a, 1); cr.addWidget(self.combine_op)
        cr.addWidget(self.combine_b, 1)
        v.addLayout(cr)
        cr2 = QHBoxLayout()
        self.combine_btn = QPushButton("Combine →")
        self.combine_btn.clicked.connect(self._combine_masks)
        cr2.addWidget(self.combine_btn)
        v.addLayout(cr2)
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
        self.fullscreen_btn = QPushButton("⛶ Full screen")
        self.fullscreen_btn.setToolTip("Show the current image full-screen (Esc to close).")
        self.fullscreen_btn.clicked.connect(self._show_fullscreen)
        bar.addWidget(self.fit_btn); bar.addWidget(self.before_btn)
        bar.addWidget(self.fullscreen_btn)
        bar.addStretch(1)
        self.status_label = QLabel("")
        bar.addWidget(self.status_label)
        v.addLayout(bar)

        self.canvas = DevelopCanvas()
        self.canvas.rectSelected.connect(self._on_rect)
        v.addWidget(self.canvas, 8)

        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        v.addWidget(self.progress)
        self.log_view = DevelopLog()
        log_hdr = QHBoxLayout()
        log_hdr.addWidget(QLabel("Edit log"))
        log_hdr.addStretch(1)
        self.save_log_btn = QPushButton("Save log…")
        self.save_log_btn.setToolTip("Save the edit log to a text file (or right-click the log)")
        self.save_log_btn.clicked.connect(lambda: self.log_view.save_log(self))
        log_hdr.addWidget(self.save_log_btn)
        v.addLayout(log_hdr)
        v.addWidget(self.log_view, 2)
        return w

    # ----------------------------------------------------------------- loading
    def _open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.fits *.fit *.fts *.xisf);;All files (*)")
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
        self.auto_btn.setEnabled(True)
        self._set_enabled_tools(True)
        self._refresh_canvas(fit=True)
        self._refresh_history()
        self.log_view.clear()
        self.log_view.append(f"Opened {Path(path).name}")

    def _save(self):
        if self.doc is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", "",
            "TIFF (*.tif);;JPEG (*.jpg);;PNG (*.png);;FITS (*.fits)")
        if not path:
            return
        quality = 92
        if Path(path).suffix.lower() in (".jpg", ".jpeg"):
            q, ok = QInputDialog.getInt(self, "JPEG quality",
                                        "Quality (1–100):", 92, 1, 100, 1)
            if not ok:
                return
            quality = q
        try:
            save_image(path, np.asarray(self.doc.result(), dtype=np.float64),
                       bit_depth=16, quality=quality)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        qtxt = f" (q{quality})" if Path(path).suffix.lower() in (".jpg", ".jpeg") else ""
        self.log_view.add(f"Saved {Path(path).name}{qtxt}")
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
        elif panel.wants_stars():
            self._mount_star_editor(panel)
        self._schedule_preview()

    def _mount_star_editor(self, panel: "ToolPanel"):
        """Show the current image with editable star markers (auto-detected for a new tool)."""
        from ..develop.spikes import detect_stars
        base = (self.doc.input_before(panel.edit_index)
                if panel.edit_index is not None else self.doc.result())
        self.canvas.set_image(np.asarray(base, dtype=np.float32), keep_view=True)
        self._preview_kind = None
        self.canvas.set_stars_mode(True)
        p = panel.params()
        length = float(p.get("length", 0.06))
        self.canvas.set_new_len(length)
        self.canvas.set_preview(int(p.get("count", 4)), float(p.get("angle", 0.0)),
                                float(p.get("jitter", 0.0)))
        stars = list(p.get("stars") or [])
        if not stars and panel.edit_index is None:          # new tool → auto-detect
            stars = detect_stars(base, max_stars=int(p.get("max_stars", 30)))
            for s in stars:
                s["len"] = length * (0.6 + 0.8 * float(s.get("flux", 0.5)))
        self.canvas.set_stars(stars)
        panel.set_stars(stars)
        lc = panel.control("length")
        if lc is not None:
            lc.valueChanged.connect(self._on_star_length)
        panel.changed.connect(self._on_star_globals)
        self.canvas.starsChanged.connect(self._on_stars_changed)
        self.canvas.starActivated.connect(self._on_star_activated)
        panel.starsRedetect.connect(self._redetect_stars)
        panel.starsClear.connect(self._clear_stars)
        self.status_label.setText(f"Star spikes: {len(stars)} stars — edit, then Apply.")

    def _on_star_length(self, v):
        if self.canvas._stars_mode:
            self.canvas.set_active_len(float(v))

    def _on_star_globals(self):
        if self.tool_panel is not None and self.tool_panel.wants_stars():
            p = self.tool_panel.params()
            self.canvas.set_preview(int(p.get("count", 4)), float(p.get("angle", 0.0)),
                                float(p.get("jitter", 0.0)))

    def _on_stars_changed(self, stars):
        if self.tool_panel is not None and self.tool_panel.wants_stars():
            self.tool_panel.set_stars(stars)

    def _on_star_activated(self, i):
        if self.tool_panel is None or not self.tool_panel.wants_stars():
            return
        stars = self.tool_panel.params().get("stars") or []
        lc = self.tool_panel.control("length")
        if lc is not None and 0 <= i < len(stars):
            lc.blockSignals(True)
            lc.set_value(float(stars[i].get("len", 0.06)))
            lc.blockSignals(False)

    def _redetect_stars(self):
        from ..develop.spikes import detect_stars
        if self.doc is None or self.tool_panel is None:
            return
        base = self.canvas.current_array()
        p = self.tool_panel.params()
        length = float(p.get("length", 0.06))
        stars = detect_stars(base, max_stars=int(p.get("max_stars", 30)))
        for s in stars:
            s["len"] = length * (0.6 + 0.8 * float(s.get("flux", 0.5)))
        self.canvas.set_stars(stars)
        self.tool_panel.set_stars(stars)
        self.status_label.setText(f"Star spikes: {len(stars)} stars detected.")

    def _clear_stars(self):
        self.canvas.set_stars([])
        if self.tool_panel is not None:
            self.tool_panel.set_stars([])

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
        self.canvas.set_stars_mode(False)
        self._edit_proxy = None
        for attr in ("tool_panel", "auto_panel"):
            panel = getattr(self, attr)
            if panel is not None:
                panel.setParent(None)
                panel.deleteLater()
                setattr(self, attr, None)
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
        if self.tool_panel.wants_stars():
            # Star spikes: the canvas shows the image + editable markers with a live
            # schematic spike overlay; the real spikes are composited on Apply/Update.
            n = len(params.get("stars") or [])
            self.status_label.setText(f"{op.label}: {n} stars — edit, then "
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

    def _op_value_str(self, op: "dev_ops.Op", params: dict, gate: dict) -> str:
        """A concise 'what was applied' string for the log Value column."""
        parts: List[str] = []
        for spec in op.params:
            if spec.key not in params or params[spec.key] is None:
                continue
            v = params[spec.key]
            if spec.kind == "choice":
                ch = spec.choices or []
                parts.append(str(ch[int(v)]) if 0 <= int(v) < len(ch) else str(v))
            elif spec.kind == "bool":
                if bool(v) != bool(spec.default):        # only note a changed toggle
                    parts.append(spec.label if v else f"no {spec.label.lower()}")
            elif spec.kind == "curve":
                n = len(v) if v else 0
                if n > 2:
                    parts.append(f"{n} pts")
            elif spec.kind == "rect":
                r = v or {}
                w = int(round((r.get("x1", 1) - r.get("x0", 0)) * 100))
                h = int(round((r.get("y1", 1) - r.get("y0", 0)) * 100))
                parts.append(f"{w}%×{h}%")
            elif spec.kind == "stars":
                parts.append(f"{len(v or [])} stars")
            else:                                        # float / int
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if abs(fv - float(spec.default)) < 1e-9:  # unchanged from default → skip
                    continue
                name = spec.label.replace(" ±", "").replace("…", "").strip()
                parts.append(f"{name} {fv:.{spec.decimals}f}" if spec.kind == "float"
                             else f"{name} {int(fv)}")
        if gate.get("mask"):
            parts.append(("¬" if gate.get("mask_invert") else "") + str(gate["mask"]))
        opac = float(gate.get("opacity", 1.0))
        if opac < 0.999:
            parts.append(f"{int(round(opac * 100))}%")
        return ", ".join(parts)

    # -------------------------------------------------------------- auto-develop
    def _open_auto(self):
        """Analyse the full image (in a worker), then show suggested steps."""
        if self.doc is None or self._busy():
            return
        self._cancel_tool(refresh=False)
        self.toolbox.clearSelection()
        self._set_busy(True)
        self.status_label.setText("Analysing the whole image…")
        img = self.doc.result()                       # read-only in the worker
        existing = set(self.doc.mask_names())          # reuse masks already in the library

        def fn(log, progress):
            log("Analysing the whole image + generating semantic masks…")
            from ..develop.auto import auto_develop_plan
            return auto_develop_plan(img, existing=existing)

        self.worker = CallableWorker(fn, mode="auto")
        self.worker.logline.connect(self.log_view.appendPlainText)
        self.worker.finished_ok.connect(self._show_auto)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _show_auto(self, plan):
        self._set_busy(False)
        steps = plan.get("steps", [])
        if not steps:
            self.status_label.setText("Auto-develop: nothing to fix — the image already looks clean.")
            self.log_view.add("Auto-develop", "no changes suggested")
            return
        # Add the semantic masks the recipe gates on (remapping step names if a mask had
        # to be renamed to avoid clobbering a user mask), so its steps resolve by name.
        self._install_auto_masks(plan.get("masks", {}), steps)
        self._refresh_masks()
        self.auto_panel = AutoPanel(steps)
        self.auto_panel.changed.connect(self._schedule_auto_preview)
        self.auto_panel.applied.connect(self._apply_auto)
        self.auto_panel.cancelled.connect(self._cancel_tool)
        self.tool_holder_layout.addWidget(self.auto_panel)
        self._placeholder.setVisible(False)
        self._auto_preview()

    def _schedule_auto_preview(self):
        if self.auto_panel is not None:
            self._auto_timer.start()

    def _auto_preview(self):
        if self.auto_panel is None or self.doc is None or self._busy():
            return
        steps = self.auto_panel.selected_steps()
        pdoc = DevelopDocument(self._get_proxy())     # preview the recipe on a proxy
        pdoc.masks = self.doc.masks                    # so semantic-mask gates resolve
        try:
            for s in steps:
                pdoc.apply_op(s["name"], s["params"],
                              mask=s.get("mask"), mask_invert=s.get("mask_invert", False))
        except Exception as exc:
            self.status_label.setText(f"Auto preview: {exc}")
            return
        self.before_btn.blockSignals(True)
        self.before_btn.setChecked(False)
        self.before_btn.blockSignals(False)
        keep = (self._preview_kind == "auto")
        self._preview_kind = "auto"
        self.canvas.set_image(np.asarray(pdoc.result(), dtype=np.float32), keep_view=keep)
        self.status_label.setText(
            f"Auto-develop preview: {len(steps)} step(s) (proxy — Apply computes full resolution)")

    def _apply_auto(self):
        if self.auto_panel is None or self.doc is None or self._busy():
            return
        steps = list(self.auto_panel.selected_steps())
        if not steps:
            self._cancel_tool()
            return
        self._set_busy(True)
        self.status_label.setText(f"Applying auto-develop ({len(steps)} steps)…")

        def fn(log, progress):
            for i, s in enumerate(steps):
                log(f"{dev_ops.get(s['name']).label}…")
                progress(i + 1, len(steps), s["name"])
                self.doc.apply_op(s["name"], s["params"],
                                  mask=s.get("mask"), mask_invert=s.get("mask_invert", False))
            return {"steps": steps}

        self.worker = CallableWorker(fn, mode="auto-apply")
        self.worker.logline.connect(self.log_view.appendPlainText)
        self.worker.finished_ok.connect(lambda r: self._finish_auto(r["steps"]))
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _finish_auto(self, steps):
        self._set_busy(False)
        for s in steps:
            op = dev_ops.get(s["name"])
            gate = {"mask": s.get("mask"), "mask_invert": s.get("mask_invert", False),
                    "opacity": 1.0}
            self.log_view.add(f"Auto: {op.label}",
                              self._op_value_str(op, op.merged(s["params"]), gate))
        self._cancel_tool(refresh=False)
        self.toolbox.clearSelection()
        self._refresh_canvas()
        self._refresh_history()

    def _apply_tool(self):
        if self.tool_panel is None or self.doc is None or self._busy():
            return
        op = self.tool_panel.op
        params = self.tool_panel.params()
        gate = self.tool_panel.gate()
        idx = self.tool_panel.edit_index
        value = self._op_value_str(op, params, gate)

        def commit():
            if idx is None:
                self.doc.apply_op(op.name, params, **gate)      # insert at cursor
            else:
                self.doc.replace_op(idx, params, **gate)        # edit in place + recompute

        verb = "Updated" if idx is not None else "Applied"
        heavy = self._apply_is_heavy(op, idx)
        if not heavy:
            commit()
            self._finish_apply(verb, op.label, value)
            return
        self._set_busy(True)
        self.status_label.setText(f"{'Updating' if idx is not None else 'Applying'} {op.label}…")

        def fn(log, progress):
            log(f"{op.label}…")
            commit()
            return {"label": op.label, "verb": verb, "value": value}

        self.worker = CallableWorker(fn, mode="apply")
        self.worker.logline.connect(self.log_view.appendPlainText)
        self.worker.finished_ok.connect(
            lambda r: self._finish_apply(r["verb"], r["label"], r["value"]))
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _finish_apply(self, verb, label, value=""):
        self._set_busy(False)
        self.log_view.add(f"{verb} {label}", value)
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
        if self.doc is None or self._busy():
            return
        m = dev_masks.luminosity_mask(self.doc.result(), kind, depth=2)
        name = self._unique_mask_name(f"Lum {kind}")
        self.doc.add_mask(name, m)
        self._refresh_masks()

    def _make_highlights_mask(self):
        if self.doc is None or self._busy():
            return
        m = dev_masks.highlights_mask(self.doc.result(), 0.6)
        self.doc.add_mask(self._unique_mask_name("Highlights"), m)
        self._refresh_masks()

    def _make_range_mask(self):
        if self.doc is None or self._busy():
            return
        m = dev_masks.range_mask(self.doc.result(), 0.0, 0.4)
        self.doc.add_mask(self._unique_mask_name("Sky"), m)
        self._refresh_masks()

    def _auto_masks(self):
        """Generate semantic masks from the current image (in a worker)."""
        if self.doc is None or self._busy():
            return
        self._set_busy(True)
        self.status_label.setText("Generating semantic masks…")
        img = self.doc.result()

        def fn(log, progress):
            log("Analysing the image for semantic masks…")
            from ..develop.semantic import segment
            return {"masks": segment(img)}

        self.worker = CallableWorker(fn, mode="auto-masks")
        self.worker.logline.connect(self.log_view.appendPlainText)
        self.worker.finished_ok.connect(lambda r: self._add_auto_masks(r["masks"]))
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _install_auto_masks(self, masks, steps=None):
        """Add auto-generated masks non-destructively; remap step gates to installed names.

        Never removes anything (so running Auto masks and Auto-develop can't delete each
        other's masks or a user's library). A mask we generated before is refreshed in
        place (idempotent re-runs, via the canonical→installed map); a collision with a
        *user* mask bumps to 'Name 2' rather than overwriting it. ``steps`` gate names are
        remapped to the installed names.
        """
        name_map = {}
        for name, m in masks.items():
            prev = self._auto_mask_map.get(name)
            if prev and prev in self.doc.masks:
                n = prev                              # refresh our own mask in place
            else:
                n = self._unique_mask_name(name)      # first time / deleted → don't clobber user
            self.doc.add_mask(n, m)
            self._auto_mask_map[name] = n
            name_map[name] = n
        if steps:
            for s in steps:
                if s.get("mask") in name_map:
                    s["mask"] = name_map[s["mask"]]
        return name_map

    def _add_auto_masks(self, masks):
        self._set_busy(False)
        name_map = self._install_auto_masks(masks)
        self._refresh_masks()
        self.log_view.add("Auto masks", ", ".join(name_map.values()))
        self.status_label.setText(f"Generated {len(masks)} semantic masks.")

    def _unique_mask_name(self, base):
        name = base; i = 2
        while name in self.doc.masks:
            name = f"{base} {i}"; i += 1
        return name

    def _refresh_masks(self):
        names = self.doc.mask_names()
        self.mask_list.clear()
        for n in names:
            self.mask_list.addItem(QListWidgetItem(n))
        for combo in (self.combine_a, self.combine_b):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear(); combo.addItems(names)
            idx = combo.findText(cur)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
        self._update_combine_enabled()
        if self.tool_panel is not None:
            self.tool_panel.refresh_masks(names)

    def _update_combine_enabled(self):
        n = len(self.doc.mask_names()) if self.doc else 0
        invert = self.combine_op.currentIndex() == 3        # ¬ invert A
        self.combine_b.setEnabled(n >= 1 and not invert)
        self.combine_btn.setEnabled(n >= (1 if invert else 2))

    def _rename_mask(self, item):
        if self.doc is None or self._busy():
            return
        old = item.text()
        new, ok = QInputDialog.getText(self, "Rename mask", "New name:", text=old)
        if not ok:
            return
        if self.doc.rename_mask(old, new.strip()):
            if self._showing_mask == old:
                self._showing_mask = new.strip()
            self._refresh_masks()
        elif new.strip() and new.strip() != old:
            QMessageBox.information(self, "Rename mask", "That name is already in use.")

    def _combine_masks(self):
        if self.doc is None or self._busy():
            return
        names = self.doc.mask_names()
        op = ("intersect", "union", "subtract", "invert")[self.combine_op.currentIndex()]
        na = self.combine_a.currentText()
        if na not in self.doc.masks:
            return
        a = self.doc.masks[na]
        nb = self.combine_b.currentText()
        b = self.doc.masks.get(nb) if op != "invert" else None
        if op != "invert" and b is None:
            return
        result = dev_masks.combine_masks(a, b, op)
        name = self._unique_mask_name(dev_masks.combined_name(na, op, None if op == "invert" else nb))
        self.doc.add_mask(name, result)
        self._refresh_masks()
        self.status_label.setText(f"Created mask: {name}")

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
        if item is None or self.doc is None or self._busy():
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

    def _show_fullscreen(self):
        """Open the image currently on the canvas in a full-screen viewer (Esc to close)."""
        arr = self.canvas.current_array()
        if arr is None:
            self.status_label.setText("Nothing to show full-screen yet — open an image first.")
            return
        self._fs_viewer = FullScreenViewer()   # kept on self so it isn't GC'd while open
        self._fs_viewer.setWindowTitle("LazyDevelop — full screen (Esc to close)")
        self._fs_viewer.set_image(arr, keep_view=False)
        self._fs_viewer.showFullScreen()
        self._fs_viewer.raise_()
        self._fs_viewer.activateWindow()

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
                  self.auto_btn, self.auto_masks_btn, self.toolbox, self.tool_holder,
                  self.history_list, self.mask_group, self.before_btn, self.fit_btn):
            b.setEnabled(not busy)
        if busy:
            self.undo_btn.setEnabled(False)
            self.redo_btn.setEnabled(False)
            self.del_step_btn.setEnabled(False)
        elif self.doc is not None:
            self.undo_btn.setEnabled(self.doc.can_undo())
            self.redo_btn.setEnabled(self.doc.can_redo())
            self.del_step_btn.setEnabled(len(self.doc.ops) > 0)
