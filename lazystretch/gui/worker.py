"""Background pipeline worker — runs run_pipeline off the UI thread."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..objects.model import Parameters
from ..pipeline.runcore import PipelineResult, run_pipeline


class PipelineWorker(QThread):
    """Runs the pipeline and streams progress/log via signals; emits the result."""

    progress = Signal(int, int, str)   # i, total, step name
    logline = Signal(str)
    finished_ok = Signal(object)       # PipelineResult
    failed = Signal(str)

    def __init__(self, image: Optional[np.ndarray], params: Parameters,
                 preview: bool, mode: str = "preview", tools=None,
                 solve_result=None, parent=None):
        super().__init__(parent)
        self._image = image
        self._params = params
        self._preview = preview
        self._tools = tools
        self._solve = solve_result
        self.mode = mode          # "preview" | "execute" (used by the caller on finish)

    def run(self):
        try:
            result: PipelineResult = run_pipeline(
                self._image, self._params, preview=self._preview,
                tools=self._tools, solve_result=self._solve,
                progress=lambda i, t, n: self.progress.emit(i, t, n),
                log=lambda line: self.logline.emit(line),
            )
            self.finished_ok.emit(result)
        except Exception as e:  # pragma: no cover - surfaced in the UI
            self.failed.emit(f"{type(e).__name__}: {e}")
