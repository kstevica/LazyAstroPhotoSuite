"""Base class for external-tool wrappers (the AI wall) — feature-detected, skippable."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class ExternalTool:
    """A non-bundled external tool, resolved at runtime with graceful degradation.

    Subclasses set ``name``, ``executables`` (candidate binary names on PATH) and
    optionally ``env_var`` (an environment variable holding the tool's path). ``path``
    (constructor arg) overrides everything. Resolution order: explicit path -> env var
    -> PATH lookup.
    """

    name: str = "external-tool"
    executables: List[str] = []
    env_var: Optional[str] = None

    def __init__(self, path: Optional[str] = None):
        self._path = path

    def resolve(self) -> Optional[str]:
        """Return the tool's executable path, or None if not installed."""
        if self._path and Path(self._path).exists():
            return self._path
        if self.env_var:
            env = os.environ.get(self.env_var)
            if env and Path(env).exists():
                return env
        for exe in self.executables:
            found = shutil.which(exe)
            if found:
                return found
        return None

    def is_available(self) -> bool:
        return self.resolve() is not None

    def run(self, img: np.ndarray, **kwargs) -> Tuple[np.ndarray, bool]:
        """Run the tool if available; otherwise return the image unchanged + False."""
        if not self.is_available():
            return np.asarray(img, dtype=np.float64).copy(), False
        return self._run(np.asarray(img, dtype=np.float64), **kwargs)

    def _run(self, img: np.ndarray, **kwargs) -> Tuple[np.ndarray, bool]:  # pragma: no cover
        raise NotImplementedError(
            f"{self.name}: integration not implemented yet (P5). Install the tool and "
            f"implement _run(); until then it is skipped gracefully."
        )
