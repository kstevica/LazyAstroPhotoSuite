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

from PySide6.QtCore import QEasingCurve, QSize, Qt, QVariantAnimation
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
    QGridLayout,
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

try:
    from .. import __version__ as _VERSION           # package source of truth
except Exception:
    _VERSION = "0.1.0"

# Background animation: a subtle clockwise rotation + zoom-in at full phase. BASE_SCALE
# over-scales enough that the rotated/zoomed image never reveals a frame edge.
_BG_MAX_ANGLE = 3.2      # degrees
_BG_BASE_SCALE = 1.22
_BG_ZOOM = 0.10

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
    {"key": "fly", "brand": "LazyFlight", "role": "Animate", "group": "process",
     "tagline": "Turn a finished still into a faithful 3D fly-through video — nothing invented.",
     "thumb": "card_fly.jpg", "accent": (96, 206, 200)},
    {"key": "moonsun", "brand": "LazyMoonSun", "role": "Sun & Moon", "group": "solar",
     "tagline": "Lucky-imaging burst stacking and finishing for the Sun and Moon.",
     "thumb": "card_moonsun.jpg", "accent": (232, 176, 92)},
]
_TOOLS_BY_KEY = {t["key"]: t for t in _TOOLS}

_TITLES = {
    "home": "LazyAstroPhotoSuite",
    "stretch": "LazyStretch",
    "stack": "LazyStack",
    "moonsun": "LazyMoonSun",
    "develop": "LazyDevelop",
    "fly": "LazyFlight",
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

        # Hover breathing zoom of the thumbnail. On leave the animation stops but the zoom
        # is NOT reset — the tile stays wherever the pulse left it.
        self._zoom = 1.0
        self._zoom_anim = QVariantAnimation(self)
        self._zoom_anim.setDuration(6400)                 # slow breathing zoom
        self._zoom_anim.setKeyValueAt(0.0, 1.0)
        self._zoom_anim.setKeyValueAt(0.5, 1.06)
        self._zoom_anim.setKeyValueAt(1.0, 1.0)
        self._zoom_anim.setEasingCurve(QEasingCurve.InOutSine)
        self._zoom_anim.setLoopCount(-1)
        self._zoom_anim.valueChanged.connect(self._set_zoom)

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
    def _set_zoom(self, v):
        self._zoom = float(v)
        self.update()

    def enterEvent(self, e):
        self._hover = True
        self._shadow.setBlurRadius(42)
        self._shadow.setColor(QColor(self._accent.red(), self._accent.green(),
                                     self._accent.blue(), 150))
        self._zoom_anim.start()                  # begin the breathing zoom
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        self._shadow.setBlurRadius(26)
        self._shadow.setColor(QColor(0, 0, 0, 170))
        self._zoom_anim.stop()                   # freeze where it is — do NOT reset the zoom
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
            zw = max(self.width(), int(self.width() * self._zoom))
            zh = max(self.height(), int(self.height() * self._zoom))
            scaled = self._pix.scaled(QSize(zw, zh), Qt.KeepAspectRatioByExpanding,
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
        self._bg_scaled: Optional[QPixmap] = None
        self._bg_key = None
        self._phase = 0.0                        # 0 … 1 … 0, drives a slow rotate + zoom-in

        # Very slow rotate (a few degrees, clockwise) + gentle zoom-in and back, so the
        # backdrop feels alive without drifting off the frame (over-scaled to stay covered).
        self._bg_anim = QVariantAnimation(self)
        self._bg_anim.setDuration(66000)
        self._bg_anim.setKeyValueAt(0.0, 0.0)
        self._bg_anim.setKeyValueAt(0.5, 1.0)
        self._bg_anim.setKeyValueAt(1.0, 0.0)
        self._bg_anim.setEasingCurve(QEasingCurve.InOutSine)
        self._bg_anim.setLoopCount(-1)
        self._bg_anim.valueChanged.connect(self._set_phase)

        root = QVBoxLayout(self)
        root.setContentsMargins(56, 40, 56, 40)
        root.addStretch(2)

        title = QLabel("LazyAstroPhotoSuite")
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

        # --- pipeline grid so columns align:
        #   col:   0=Stack  1=→  2=Stretch  3=→  4=Develop  5=→  6=Fly
        #   Sun & Moon sits in row 3, column 0 → directly under "Build the master".
        blue = _TOOLS_BY_KEY["stack"]["accent"]
        violet = _TOOLS_BY_KEY["stretch"]["accent"]
        amber = _TOOLS_BY_KEY["moonsun"]["accent"]
        head = Qt.AlignLeft | Qt.AlignBottom
        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(12)
        grid.addWidget(_section_label("Build the master", blue), 0, 0, head)
        grid.addWidget(_section_label("Process the image", violet), 0, 2, 1, 5, head)
        grid.addWidget(ToolCard(_TOOLS_BY_KEY["stack"], on_open), 1, 0)
        grid.addLayout(self._arrow(), 1, 1)
        grid.addWidget(ToolCard(_TOOLS_BY_KEY["stretch"], on_open), 1, 2)
        grid.addLayout(self._arrow(), 1, 3)           # arrow between Stretch and Finish
        grid.addWidget(ToolCard(_TOOLS_BY_KEY["develop"], on_open), 1, 4)
        grid.addLayout(self._arrow(), 1, 5)           # arrow between Finish and Animate
        grid.addWidget(ToolCard(_TOOLS_BY_KEY["fly"], on_open), 1, 6)
        grid.setRowMinimumHeight(2, 26)               # gap before Sun & Moon
        grid.addWidget(_section_label("Sun & Moon", amber), 3, 0, head)
        grid.addWidget(ToolCard(_TOOLS_BY_KEY["moonsun"], on_open), 4, 0)

        center = QHBoxLayout()
        center.addStretch(1)
        center.addLayout(grid)
        center.addStretch(1)
        root.addLayout(center)
        root.addStretch(3)

        # credit footer, bottom-right
        foot = QHBoxLayout()
        foot.addStretch(1)
        footer = QLabel(f"LazyAstroPhotoSuite  v{_VERSION}   ·   kstevica.com/laps   "
                        f"·   kstevica@gmail.com")
        footer.setStyleSheet(
            "color: rgba(196,206,228,0.60); background: transparent; font-size: 12px;")
        foot.addWidget(footer)
        root.addLayout(foot)

    def _arrow(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.addStretch(1)
        arrow = QLabel("→")
        af = QFont(); af.setPointSize(34)
        arrow.setFont(af)
        arrow.setStyleSheet("color: rgba(190,200,225,0.55); background: transparent;")
        col.addWidget(arrow, 0, Qt.AlignCenter)
        col.addStretch(1)
        return col

    def _set_phase(self, v):
        self._phase = float(v)
        self.update()

    def _ensure_bg(self):
        """Cache the background cover-scaled to the frame (the paint transform adds the
        rotate/zoom over-scale on top)."""
        key = (self.width(), self.height())
        if self._bg_key == key or self._bg.isNull() or self.width() < 2:
            return
        self._bg_scaled = self._bg.scaled(self.size(), Qt.KeepAspectRatioByExpanding,
                                          Qt.SmoothTransformation)
        self._bg_key = key

    def resizeEvent(self, e):
        self._bg_key = None                      # rebuild the scaled cache at the new size
        super().resizeEvent(e)

    def showEvent(self, e):
        if self._bg_anim.state() != QVariantAnimation.Running:
            self._bg_anim.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._bg_anim.stop()                     # don't burn CPU while a tool is open
        super().hideEvent(e)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.setRenderHint(QPainter.Antialiasing)
        self._ensure_bg()
        if self._bg_scaled is not None:
            # rotate a few degrees clockwise + zoom in, about the frame centre; BASE_SCALE
            # keeps the rotated image covering the frame at every phase.
            p.save()
            p.translate(self.width() / 2, self.height() / 2)
            p.rotate(_BG_MAX_ANGLE * self._phase)
            s = _BG_BASE_SCALE * (1.0 + _BG_ZOOM * self._phase)
            p.scale(s, s)
            p.drawPixmap(-self._bg_scaled.width() // 2, -self._bg_scaled.height() // 2,
                         self._bg_scaled)
            p.restore()
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
        if key == "fly":
            from .flight_window import LazyFlightPanel
            return LazyFlightPanel()
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
