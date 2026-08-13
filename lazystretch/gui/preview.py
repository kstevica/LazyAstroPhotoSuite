"""Preview pane — a zoom/pan QGraphicsView showing a numpy image."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
)


def ndarray_to_qimage(a: np.ndarray) -> QImage:
    """Convert a float [0,1] mono/RGB array to an 8-bit QImage (copied, self-owned)."""
    arr = np.clip(np.asarray(a, dtype=np.float64), 0.0, 1.0)
    u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    if u8.ndim == 2:
        h, w = u8.shape
        u8 = np.ascontiguousarray(u8)
        return QImage(u8.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, _ = u8.shape
    u8 = np.ascontiguousarray(u8[..., :3])
    return QImage(u8.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# Gentle per-wheel-step zoom base. angleDelta is 120 per mouse notch and small on
# trackpads, so 1.0011**delta is smooth and continuous: ~1.14x per notch (vs the old
# abrupt 1.25x), and fine-grained on a trackpad.
_ZOOM_BASE = 1.0011
_MIN_SCALE = 0.02
_MAX_SCALE = 40.0


class PreviewView(QGraphicsView):
    """Fit-to-window on set; smooth wheel zoom anchored under the cursor; drag to pan.

    Double-click resets to fit.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(self.renderHints())
        self.setBackgroundBrush(Qt.black)
        self.setAlignment(Qt.AlignCenter)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)   # zoom toward cursor
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self._has_image = False
        self._user_zoomed = False
        self._img_size = None            # (w, h) of the current pixmap
        self._array = None               # last numpy array set (for the full-screen viewer)
        # --- painting (nightscape sky/foreground scribbles) ---
        self._scribble_item = QGraphicsPixmapItem()      # translucent overlay above the image
        self._scribble_item.setZValue(1)
        self._scene.addItem(self._scribble_item)
        self._scribbles = None           # int8 (H, W): +1 sky, -1 earth, 0 none
        self._paint_mode = None          # None | 'sky' | 'earth' | 'erase'
        self._brush = 14                 # brush radius in image pixels

    # -------------------------------------------------------------- painting

    def set_paint_mode(self, mode):
        """Enable a brush ('sky'|'earth'|'erase') or None to pan. Allocates a scribble map."""
        self._paint_mode = mode
        self.setDragMode(QGraphicsView.NoDrag if mode else QGraphicsView.ScrollHandDrag)
        if mode and self._scribbles is None and self._array is not None:
            self._scribbles = np.zeros(self._array.shape[:2], dtype=np.int8)

    def set_brush(self, radius: int):
        self._brush = max(1, int(radius))

    def scribbles(self):
        """The current scribble map (+1 sky / −1 earth / 0), or None if nothing painted."""
        return None if self._scribbles is None else self._scribbles.copy()

    def clear_scribbles(self):
        if self._scribbles is not None:
            self._scribbles[:] = 0
            self._refresh_scribble_overlay()

    def _paint_at(self, scene_pt):
        if self._scribbles is None or self._paint_mode is None:
            return
        h, w = self._scribbles.shape
        ix, iy, r = int(round(scene_pt.x())), int(round(scene_pt.y())), self._brush
        y0, y1 = max(0, iy - r), min(h, iy + r + 1)
        x0, x1 = max(0, ix - r), min(w, ix + r + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disc = (yy - iy) ** 2 + (xx - ix) ** 2 <= r * r
        self._scribbles[y0:y1, x0:x1][disc] = {"sky": 1, "earth": -1, "erase": 0}[self._paint_mode]
        self._refresh_scribble_overlay()

    def _refresh_scribble_overlay(self):
        if self._scribbles is None:
            return
        h, w = self._scribbles.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[self._scribbles > 0] = (60, 170, 255, 130)     # sky = cyan
        rgba[self._scribbles < 0] = (255, 70, 70, 130)      # earth = red
        rgba = np.ascontiguousarray(rgba)
        img = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        self._scribble_item.setPixmap(QPixmap.fromImage(img))

    def current_array(self) -> "np.ndarray | None":
        """The numpy image currently displayed, or ``None`` if nothing is shown yet."""
        return self._array

    def set_image(self, a: np.ndarray, keep_view: bool | None = None):
        """Show ``a``. When the new image is the SAME size as the current one (a
        re-process of the same frame), the zoom + pan are preserved so changes can be
        compared in place; a different size (a new frame) fits to window. ``keep_view``
        forces the choice: False always fits, True keeps the view when the size matches.
        """
        a = np.asarray(a)
        self._array = a
        new_size = (a.shape[1], a.shape[0])
        keep = (self._has_image and self._img_size == new_size and keep_view is not False)
        saved = None
        if keep:
            saved = (self.transform(),
                     self.horizontalScrollBar().value(),
                     self.verticalScrollBar().value())
        pix = QPixmap.fromImage(ndarray_to_qimage(a))
        self._item.setPixmap(pix)
        self._scene.setSceneRect(self._item.boundingRect())
        self._has_image = True
        if new_size != self._img_size and self._scribbles is not None:   # new frame → drop stale scribbles
            self._scribbles = None
            self._scribble_item.setPixmap(QPixmap())
        self._img_size = new_size
        if saved is not None:
            transform, hs, vs = saved
            self.setTransform(transform)         # restore zoom
            self.horizontalScrollBar().setValue(hs)
            self.verticalScrollBar().setValue(vs)  # restore pan (scene rect is unchanged)
        else:
            self._user_zoomed = False
            self.fit()

    def fit(self):
        if self._has_image:
            self.fitInView(self._item, Qt.KeepAspectRatio)
            self._user_zoomed = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._user_zoomed:      # only auto-refit while the user hasn't zoomed
            self.fit()

    def wheelEvent(self, event):
        if not self._has_image:
            return
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta == 0:
            return
        factor = _ZOOM_BASE ** delta
        cur = self.transform().m11()
        target = cur * factor
        if target < _MIN_SCALE:
            factor = _MIN_SCALE / cur
        elif target > _MAX_SCALE:
            factor = _MAX_SCALE / cur
        self._user_zoomed = True
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if self._paint_mode and event.button() == Qt.LeftButton and self._has_image:
            self._paint_at(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._paint_mode and (event.buttons() & Qt.LeftButton) and self._has_image:
            self._paint_at(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.fit()
        super().mouseDoubleClickEvent(event)


class FullScreenViewer(PreviewView):
    """A borderless full-screen image viewer with the same zoom/pan behaviour.

    Opened from the main window's *Full screen* button on whatever image is on the
    preview. Esc closes it; scroll zooms, drag pans, double-click refits.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hint = QLabel(
            "Esc to close   ·   scroll to zoom   ·   drag to pan   ·   double-click to fit",
            self)
        self._hint.setStyleSheet(
            "color: white; background: rgba(0, 0, 0, 150); padding: 5px 10px; "
            "border-radius: 5px;")
        self._hint.move(14, 14)
        self._hint.adjustSize()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)
