"""Background workers — run the pipelines off the UI thread."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..objects.model import Parameters
from ..pipeline.runcore import PipelineResult, run_pipeline


class CallableWorker(QThread):
    """Run an arbitrary ``fn(log, progress) -> result`` off the UI thread.

    The generic sibling of :class:`PipelineWorker` — used by the LazyMoonSun (and future
    LazyStack) panels, which drive their own engines rather than ``run_pipeline``. The
    signal contract is identical, so panels wire up the same slots.
    """

    progress = Signal(int, int, str)
    logline = Signal(str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable, *, mode: str = "", parent=None):
        super().__init__(parent)
        self._fn = fn
        self.mode = mode

    def run(self):
        try:
            result = self._fn(lambda line: self.logline.emit(line),
                              lambda i, t, n: self.progress.emit(i, t, n))
            self.finished_ok.emit(result)
        except Exception as e:  # pragma: no cover - surfaced in the UI
            self.failed.emit(f"{type(e).__name__}: {e}")


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
