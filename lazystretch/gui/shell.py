"""Application shell — one window hosting the launcher and the tool panels.

The shell is a ``QMainWindow`` whose central widget is a ``QStackedWidget``: page 0 is a
poster-style launcher and the remaining pages are the tool panels, built lazily on first
open. A persistent "Tools" menu switches pages, so every tool is one click away and the
launcher is always reachable via Tools ▸ Home.

The launcher frames the suite as a pipeline: **Build the master** (LazyStack) → **Process
the image** (LazyStretch → LazyDevelop), with **Sun & Moon** (LazyMoonSun) as a separate
specialty. One QApplication, one window, shared widgets/io.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

_ASSETS = Path(__file__).parent / "assets"

# The tools, in pipeline order. group: "master" → "process" → "solar".
_TOOLS = [
    {"key": "stack", "brand": "LazyStack", "role": "Integrate", "group": "master",
     "tagline": "Calibrate, register and integrate a folder of subs into a clean master.",
     "thumb": "card_stack.jpg", "accent": (86, 170, 224)},
    {"key": "stretch", "brand": "LazyStretch", "role": "Stretch", "group": "process",
     "tagline": "Automated, statistics-driven stretching and finishing for deep-sky masters.",
     "thumb": "card_stretch.jpg", "accent": (150, 128, 232)},
    {"key": "develop", "brand": "LazyDevelop", "role": "Finish", "group": "process",
     "tagline": "Hand-finish a stretched master — curves, colour, detail, masks, semantic auto.",
     "thumb": "card_develop.jpg", "accent": (150, 128, 232)},
    {"key": "moonsun", "brand": "LazyMoonSun", "role": "Sun & Moon", "group": "solar",
     "tagline": "Lucky-imaging burst stacking and finishing for the Sun and Moon.",
     "thumb": "card_moonsun.jpg", "accent": (232, 176, 92)},
]
_TOOLS_BY_KEY = {t["key"]: t for t in _TOOLS}

_TITLES = {
    "home": "LazyStretch Suite",
    "stretch": "LazyStretch",
    "stack": "LazyStack",
    "moonsun": "LazyMoonSun",
    "develop": "LazyDevelop",
}


def _section_label(text: str, accent) -> QLabel:
    """A small uppercase section heading with an accent tint."""
    lbl = QLabel(text.upper())
    r, g, b = accent
    lbl.setStyleSheet(
        f"color: rgb({r},{g},{b}); font-size: 12px; font-weight: 800; "
        "letter-spacing: 3px; padding: 0 2px;")
    return lbl


class ToolCard(QFrame):
    """A poster tile: a thumbnail with a role chip, brand, tagline and Open — the whole
    card is clickable and glows in its accent colour on hover."""

    def __init__(self, spec: dict, on_open: Callable[[str], None]):
        super().__init__()
        self.setObjectName("toolCard")
        self.setFixedSize(300, 384)
        self.setCursor(Qt.PointingHandCursor)
        self._spec = spec
        self._on_open = on_open
        self._hover = False
        self._accent = QColor(*spec["accent"])
        self._pix = QPixmap(str(_ASSETS / spec["thumb"]))

        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(26)
        self._shadow.setOffset(0, 12)
        self._shadow.setColor(QColor(0, 0, 0, 170))
        self.setGraphicsEffect(self._shadow)

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 18)
        v.setSpacing(8)

        r, g, b = spec["accent"]
        chip = QLabel(spec["role"].upper())
        chip.setObjectName("roleChip")
        chip.setStyleSheet(
            f"#roleChip {{ background: rgba({r},{g},{b},0.90); color: white; "
            "border-radius: 9px; padding: 3px 10px; font-size: 10px; font-weight: 800; "
            "letter-spacing: 1px; }")
        top = QHBoxLayout()
        top.addWidget(chip)
        top.addStretch(1)
        v.addLayout(top)
        v.addStretch(1)

        title = QLabel(spec["brand"])
        tf = QFont(); tf.setPointSize(title.font().pointSize() + 9); tf.setBold(True)
        title.setFont(tf)
        title.setStyleSheet("color: white; background: transparent;")
        v.addWidget(title)

        desc = QLabel(spec["tagline"])
        desc.setWordWrap(True)
        desc.setStyleSheet("color: rgba(226,231,242,0.82); background: transparent; font-size: 12px;")
        desc.setMinimumHeight(56)
        v.addWidget(desc)

        btn = QPushButton("Open  →")
        btn.setObjectName("openBtn")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            "#openBtn { background: rgba(255,255,255,0.12); color: white; "
            "border: 1px solid rgba(255,255,255,0.30); border-radius: 9px; "
            "padding: 9px 0; font-weight: 700; }"
            f"#openBtn:hover {{ background: rgba({r},{g},{b},0.55); "
            f"border: 1px solid rgba({r},{g},{b},0.95); }}")
        btn.clicked.connect(lambda: on_open(spec["key"]))
        v.addWidget(btn)

    # -- interaction --
    def enterEvent(self, e):
        self._hover = True
        self._shadow.setBlurRadius(42)
        self._shadow.setColor(QColor(self._accent.red(), self._accent.green(),
                                     self._accent.blue(), 150))
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        self._shadow.setBlurRadius(26)
        self._shadow.setColor(QColor(0, 0, 0, 170))
        self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._on_open(self._spec["key"])
        super().mousePressEvent(e)

    # -- painting: rounded thumbnail + legibility gradient + accent border --
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0.5, 0.5, self.width() - 1, self.height() - 1, 16, 16)
        p.setClipPath(path)
        if not self._pix.isNull():
            scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatioByExpanding,
                                      Qt.SmoothTransformation)
            p.drawPixmap(-(scaled.width() - self.width()) // 2,
                         -(scaled.height() - self.height()) // 2, scaled)
        else:
            p.fillRect(self.rect(), QColor(18, 22, 34))
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QColor(6, 8, 16, 30))
        grad.setColorAt(0.42, QColor(6, 8, 16, 105))
        grad.setColorAt(1.0, QColor(3, 5, 11, 238))
        p.fillRect(self.rect(), grad)
        p.setClipping(False)
        if self._hover:
            p.setPen(QPen(self._accent, 2))
        else:
            p.setPen(QPen(QColor(255, 255, 255, 46), 1))
        p.drawPath(path)


class LauncherPage(QWidget):
    """The startup chooser — a pipeline of poster tiles over a deep-sky backdrop."""

    def __init__(self, on_open: Callable[[str], None]):
        super().__init__()
        self._bg = QPixmap(str(_ASSETS / "launcher_bg.jpg"))

        root = QVBoxLayout(self)
        root.setContentsMargins(56, 40, 56, 40)
        root.addStretch(2)

        title = QLabel("LazyStretch Suite")
        hf = QFont(); hf.setPointSize(title.font().pointSize() + 20); hf.setBold(True)
        title.setFont(hf)
        title.setStyleSheet("color: white; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        sub = QLabel("An astrophotography workflow — from raw subs to a finished image.")
        sub.setStyleSheet("color: rgba(206,214,232,0.80); background: transparent; font-size: 15px;")
        sub.setAlignment(Qt.AlignCenter)
        root.addWidget(sub)
        root.addSpacing(34)

        # --- pipeline: [Build the master]  →  [Process the image] ---
        pipe = QHBoxLayout()
        pipe.setSpacing(22)
        pipe.addStretch(1)

        pipe.addLayout(self._group("Build the master", _TOOLS_BY_KEY["stack"]["accent"],
                                   [ToolCard(_TOOLS_BY_KEY["stack"], on_open)]))
        pipe.addLayout(self._arrow())
        pipe.addLayout(self._group(
            "Process the image", _TOOLS_BY_KEY["stretch"]["accent"],
            [ToolCard(_TOOLS_BY_KEY["stretch"], on_open),
             ToolCard(_TOOLS_BY_KEY["develop"], on_open)]))
        pipe.addStretch(1)
        root.addLayout(pipe)

        root.addSpacing(30)

        # --- separate specialty: Sun & Moon ---
        solar = QHBoxLayout()
        solar.addStretch(1)
        solar.addLayout(self._group("Sun & Moon", _TOOLS_BY_KEY["moonsun"]["accent"],
                                    [ToolCard(_TOOLS_BY_KEY["moonsun"], on_open)]))
        solar.addStretch(1)
        root.addLayout(solar)
        root.addStretch(3)

    def _group(self, heading: str, accent, cards) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(12)
        col.addWidget(_section_label(heading, accent), 0, Qt.AlignLeft)
        row = QHBoxLayout()
        row.setSpacing(20)
        for c in cards:
            row.addWidget(c)
        col.addLayout(row)
        col.addStretch(1)
        return col

    def _arrow(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.addSpacing(30)                       # drop below the section heading row
        col.addStretch(1)
        arrow = QLabel("→")
        af = QFont(); af.setPointSize(34)
        arrow.setFont(af)
        arrow.setStyleSheet("color: rgba(190,200,225,0.55); background: transparent;")
        col.addWidget(arrow, 0, Qt.AlignVCenter)
        col.addStretch(1)
        return col

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        if not self._bg.isNull():
            scaled = self._bg.scaled(self.size(), Qt.KeepAspectRatioByExpanding,
                                     Qt.SmoothTransformation)
            p.drawPixmap(-(scaled.width() - self.width()) // 2,
                         -(scaled.height() - self.height()) // 2, scaled)
        else:
            p.fillRect(self.rect(), QColor(8, 10, 18))
        # darken so the tiles and text read clearly
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QColor(6, 8, 16, 205))
        grad.setColorAt(0.5, QColor(6, 8, 16, 150))
        grad.setColorAt(1.0, QColor(4, 6, 12, 225))
        p.fillRect(self.rect(), grad)


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
        for t in _TOOLS:
            act = QAction(t["brand"], self)
            act.triggered.connect(lambda _=False, k=t["key"]: self.open_tool(k))
            menu.addAction(act)
            self._tool_actions[t["key"]] = act

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
        """Construct a tool panel on first open."""
        if key == "stretch":
            from .main_window import LazyStretchPanel
            return LazyStretchPanel()
        if key == "moonsun":
            from .moonsun_window import LazyMoonSunPanel
            return LazyMoonSunPanel()
        if key == "stack":
            from .stack_window import LazyStackPanel
            return LazyStackPanel()
        if key == "develop":
            from .develop_window import LazyDevelopPanel
            return LazyDevelopPanel()
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
