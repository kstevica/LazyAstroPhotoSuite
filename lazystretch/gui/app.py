"""GUI entry point: ``lazystretch-gui`` (or ``python -m lazystretch.gui.app``)."""
from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from .shell import AppShell

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("LazyStretch")
    win = AppShell()          # launcher → LazyStretch / LazyStack / LazyMoonSun
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
