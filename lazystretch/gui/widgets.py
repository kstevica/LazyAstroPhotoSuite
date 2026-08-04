"""Small reusable Qt widgets: a float slider and a file picker."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
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
        self._slider = QSlider(Qt.Horizontal)
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
                 name_filter: str = "Images (*.fits *.fit *.fts *.tif *.tiff *.png *.xisf)",
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
