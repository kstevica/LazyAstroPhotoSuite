"""Application shell — one window hosting the launcher and the three tool panels.

The shell is a ``QMainWindow`` whose central widget is a ``QStackedWidget``: page 0 is a
launcher chooser (LazyStretch / LazyStack / LazyMoonSun cards) and the remaining pages are
the tool panels, built lazily on first open. A persistent "Tools" menu switches pages, so
every tool is one click away and the launcher is always reachable via Tools ▸ Home.

Only LazyStretch is built today; LazyStack and LazyMoonSun arrive in later phases and show a
disabled "coming soon" card until then. One QApplication, one window, shared widgets/io.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# key, brand, tagline, available-now
_TOOLS = [
    ("stretch", "LazyStretch",
     "Automated, statistics-driven stretching & finishing for deep-sky masters.", True),
    ("stack", "LazyStack",
     "Calibrate, register and integrate a folder of subs into a master.", True),
    ("moonsun", "LazyMoonSun",
     "Lucky-imaging burst stacking & finishing for the Sun and Moon.", True),
]

_TITLES = {
    "home": "LazyStretch Suite",
    "stretch": "LazyStretch",
    "stack": "LazyStack",
    "moonsun": "LazyMoonSun",
}


class ToolCard(QFrame):
    """A single launcher card: brand, tagline, and an Open (or 'Coming soon') button."""

    def __init__(self, key: str, brand: str, tagline: str, available: bool,
                 on_open: Callable[[str], None]):
        super().__init__()
        self.setObjectName("toolCard")
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(300)
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 18, 18, 18)
        v.setSpacing(10)

        title = QLabel(brand)
        f = title.font()
        f.setPointSize(f.pointSize() + 5)
        f.setBold(True)
        title.setFont(f)
        v.addWidget(title)

        desc = QLabel(tagline)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: gray;")
        desc.setMinimumHeight(64)
        desc.setAlignment(Qt.AlignTop)
        v.addWidget(desc, 1)

        btn = QPushButton("Open" if available else "Coming soon")
        btn.setEnabled(available)
        if available:
            btn.clicked.connect(lambda: on_open(key))
        else:
            btn.setToolTip("Arrives in a later phase of the port.")
        v.addWidget(btn)


class LauncherPage(QWidget):
    """The chooser shown on startup — a row of tool cards under a heading."""

    def __init__(self, on_open: Callable[[str], None]):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 40, 40, 40)
        outer.addStretch(1)

        heading = QLabel("LazyStretch Suite")
        hf = heading.font()
        hf.setPointSize(hf.pointSize() + 9)
        hf.setBold(True)
        heading.setFont(hf)
        heading.setAlignment(Qt.AlignCenter)
        outer.addWidget(heading)

        sub = QLabel("Choose a tool to get started.")
        sub.setStyleSheet("color: gray;")
        sub.setAlignment(Qt.AlignCenter)
        outer.addWidget(sub)
        outer.addSpacing(28)

        cards = QHBoxLayout()
        cards.addStretch(1)
        for key, brand, tagline, available in _TOOLS:
            cards.addWidget(ToolCard(key, brand, tagline, available, on_open))
        cards.addStretch(1)
        outer.addLayout(cards)
        outer.addStretch(2)


class AppShell(QMainWindow):
    """Top-level window: launcher (page 0) + lazily-built tool panels, with a Tools menu."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(_TITLES["home"])
        self.resize(1760, 1060)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.launcher = LauncherPage(on_open=self.open_tool)
        self.stack.addWidget(self.launcher)          # index 0 = home

        self._panels: Dict[str, QWidget] = {}
        self._build_menu()
        self.show_home()

    # ------------------------------------------------------------------ menu

    def _build_menu(self):
        menu = self.menuBar().addMenu("Tools")
        home = QAction("Home (launcher)", self)
        home.setShortcut("Ctrl+Shift+H")
        home.triggered.connect(self.show_home)
        menu.addAction(home)
        menu.addSeparator()
        self._tool_actions: Dict[str, QAction] = {}
        for key, brand, _tag, available in _TOOLS:
            act = QAction(brand, self)
            act.setEnabled(available)
            act.triggered.connect(lambda _=False, k=key: self.open_tool(k))
            menu.addAction(act)
            self._tool_actions[key] = act

        # A persistent back-to-launcher button, shown only while a tool is open (the
        # tools are stacked pages in one window, so there is no per-tool window to close).
        self.toolbar = QToolBar("Navigation")
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)
        self.home_btn = QAction("⌂  Launcher", self)
        self.home_btn.setToolTip("Back to the tool selector (Ctrl+Shift+H)")
        self.home_btn.triggered.connect(self.show_home)
        self.toolbar.addAction(self.home_btn)
        self._tool_name_label = QLabel("")
        self._tool_name_label.setStyleSheet("margin-left: 12px; font-weight: bold;")
        self.toolbar.addWidget(self._tool_name_label)
        self.toolbar.setVisible(False)

    # --------------------------------------------------------------- navigation

    def show_home(self):
        self.stack.setCurrentWidget(self.launcher)
        self.setWindowTitle(_TITLES["home"])
        self.toolbar.setVisible(False)

    def _make_panel(self, key: str) -> Optional[QWidget]:
        """Construct a tool panel on first open. Only LazyStretch exists so far."""
        if key == "stretch":
            from .main_window import LazyStretchPanel
            return LazyStretchPanel()
        if key == "moonsun":
            from .moonsun_window import LazyMoonSunPanel
            return LazyMoonSunPanel()
        if key == "stack":
            from .stack_window import LazyStackPanel
            return LazyStackPanel()
        return None

    def open_tool(self, key: str):
        if key not in self._panels:
            panel = self._make_panel(key)
            if panel is None:
                return
            self.stack.addWidget(panel)
            self._panels[key] = panel
        self.stack.setCurrentWidget(self._panels[key])
        self.setWindowTitle(_TITLES.get(key, _TITLES["home"]))
        self._tool_name_label.setText(_TITLES.get(key, ""))
        self.toolbar.setVisible(True)
