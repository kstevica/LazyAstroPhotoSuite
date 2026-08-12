"""Small reusable Qt widgets: a float slider, a file picker, and the run-log view."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)


class FloatSlider(QWidget):
    """A labelled horizontal slider mapping an int track to a float range."""

    valueChanged = Signal(float)

    def __init__(self, label: str, lo: float, hi: float, value: float = 0.0,
                 decimals: int = 2, steps: int = 1000, parent=None):
        super().__init__(parent)
        self._lo, self._hi, self._steps, self._dec = lo, hi, steps, decimals
        self._name = QLabel(label)
        self._name.setMinimumWidth(96)
        self._name.setMaximumWidth(210)          # never let a long label squeeze the slider away
        self._name.setToolTip(label)             # full text on hover if it ever elides
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimumWidth(80)         # keep the track draggable regardless of layout
        self._slider.setRange(0, steps)
        self._slider.setValue(self._to_int(value))
        self._value = QLabel()
        self._value.setMinimumWidth(48)
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._slider.valueChanged.connect(self._on_change)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._name)
        lay.addWidget(self._slider, 1)
        lay.addWidget(self._value)
        self._update_label()

    def _to_int(self, v: float) -> int:
        return int(round((v - self._lo) / (self._hi - self._lo) * self._steps))

    def _to_float(self, i: int) -> float:
        return self._lo + (i / self._steps) * (self._hi - self._lo)

    def _update_label(self):
        self._value.setText(f"{self.value():+.{self._dec}f}" if self._lo < 0
                            else f"{self.value():.{self._dec}f}")

    def _on_change(self, _i: int):
        self._update_label()
        self.valueChanged.emit(self.value())

    def value(self) -> float:
        return self._to_float(self._slider.value())

    def set_value(self, v: float):
        self._slider.blockSignals(True)
        self._slider.setValue(self._to_int(v))
        self._slider.blockSignals(False)
        self._update_label()


class FilePicker(QWidget):
    """A 'choose file' button + elided filename label."""

    fileChosen = Signal(str)

    def __init__(self, caption: str = "Choose…",
                 name_filter: str = ("Images (*.fits *.fit *.fts *.tif *.tiff *.png *.xisf "
                                     "*.cr2 *.cr3 *.nef *.arw *.raf *.dng *.orf *.rw2 "
                                     "*.pef *.srw *.raw)"),
                 parent=None):
        super().__init__(parent)
        self._caption = caption
        self._filter = name_filter
        self._path: Optional[str] = None
        self._button = QPushButton(caption)
        self._label = QLabel("(none)")
        self._label.setStyleSheet("color: gray;")
        self._button.clicked.connect(self._choose)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._button)
        lay.addWidget(self._label, 1)

    def _choose(self):
        path, _ = QFileDialog.getOpenFileName(self, self._caption, "", self._filter)
        if path:
            self.set_path(path)

    def set_path(self, path: str):
        self._path = path
        self._label.setText(Path(path).name)
        self._label.setStyleSheet("")
        self.fileChosen.emit(path)

    def path(self) -> Optional[str]:
        return self._path

    def clear(self):
        self._path = None
        self._label.setText("(none)")
        self._label.setStyleSheet("color: gray;")


class RunLogView(QTreeWidget):
    """A 3-column run log: time-since-start · event · how long the step took.

    Each ``append`` stamps the elapsed time since the run began; the step's duration is the
    gap until the next line, backfilled when that line arrives (``finish`` closes the last
    one). Drop-in for the old QPlainTextEdit — ``appendPlainText`` and ``clear`` still work.
    """

    _MAX_ROWS = 4000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHeaderLabels(["Time", "Event", "Took"])
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        h = self.header()
        h.setStretchLastSection(False)
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._t0: Optional[float] = None
        self._last_item: Optional[QTreeWidgetItem] = None
        self._last_time: Optional[float] = None

    def start(self):
        """Reset the log and the clock (safe to call at the start of a run)."""
        super().clear()
        self._t0 = None
        self._last_item = None
        self._last_time = None

    def clear(self):                                   # QPlainTextEdit-compatible
        self.start()

    def append(self, msg: str):
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        if self._last_item is not None and self._last_time is not None:
            self._last_item.setText(2, self._fmt_dur(now - self._last_time))
        item = QTreeWidgetItem([self._fmt_elapsed(now - self._t0), msg, ""])
        item.setTextAlignment(0, Qt.AlignRight | Qt.AlignVCenter)
        item.setTextAlignment(2, Qt.AlignRight | Qt.AlignVCenter)
        self.addTopLevelItem(item)
        self.scrollToBottom()
        self._last_item = item
        self._last_time = now
        if self.topLevelItemCount() > self._MAX_ROWS:
            self.takeTopLevelItem(0)

    def appendPlainText(self, msg: str):               # QPlainTextEdit-compatible
        self.append(msg)

    def finish(self):
        """Close the last line's duration when the run ends."""
        if self._last_item is not None and self._last_time is not None:
            self._last_item.setText(2, self._fmt_dur(time.monotonic() - self._last_time))

    @staticmethod
    def _fmt_elapsed(s: float) -> str:
        m = int(s // 60)
        return f"{m:d}:{s - 60 * m:04.1f}" if m else f"{s:.1f}s"

    @staticmethod
    def _fmt_dur(s: float) -> str:
        if s < 0.1:
            return f"{s * 1000:.0f}ms"
        if s < 60:
            return f"{s:.1f}s"
        m = int(s // 60)
        return f"{m:d}:{s - 60 * m:04.1f}"
