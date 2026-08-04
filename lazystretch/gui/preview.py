"""Preview pane — a zoom/pan QGraphicsView showing a numpy image."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


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

    def set_image(self, a: np.ndarray):
        pix = QPixmap.fromImage(ndarray_to_qimage(a))
        self._item.setPixmap(pix)
        self._scene.setSceneRect(self._item.boundingRect())
        self._has_image = True
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

    def mouseDoubleClickEvent(self, event):
        self.fit()
        super().mouseDoubleClickEvent(event)
