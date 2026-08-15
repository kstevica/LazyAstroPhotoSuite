"""GUI entry point: ``lazystretch-gui`` (or ``python -m lazystretch.gui.app``)."""
from __future__ import annotations

import sys

# The whole suite (launcher + LazyStretch / LazyStack / LazyDevelop / LazyFlight /
# LazyMoonSun) presents under one name. Change this one string to rebrand (e.g. "LAPS").
APP_NAME = "LazyAstroPhotoSuite"


def _rename_macos_app(name: str) -> bool:
    """Make the macOS app menu (the bold item next to the Apple logo) show ``name``.

    An unbundled interpreter app otherwise shows "Python" there — macOS takes the app-menu
    title from ``CFBundleName``, which ``QApplication.setApplicationName`` does NOT change.
    We rewrite the in-memory bundle-info value via the Objective-C runtime (ctypes, no
    pyobjc dependency). Best-effort and crash-safe: it no-ops on non-macOS, when pyobjc-free
    messaging fails, or when the info dict is immutable. Must run before the menu is built.
    Returns True only if the name was actually set.
    """
    if sys.platform != "darwin":
        return False
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        def msg(obj, sel, *args, argtypes=(), restype=ctypes.c_void_p):
            objc.objc_msgSend.restype = restype
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
            return objc.objc_msgSend(obj, objc.sel_registerName(sel.encode()), *args)

        NSBundle = objc.objc_getClass(b"NSBundle")
        NSString = objc.objc_getClass(b"NSString")
        NSMutableDictionary = objc.objc_getClass(b"NSMutableDictionary")
        if not (NSBundle and NSString and NSMutableDictionary):
            return False

        def nsstr(s):
            return msg(NSString, "stringWithUTF8String:", s.encode("utf-8"),
                       argtypes=[ctypes.c_char_p])

        info = msg(msg(NSBundle, "mainBundle"), "infoDictionary")
        if not info:
            return False
        # Only mutate if it's genuinely an NSMutableDictionary — otherwise setObject:forKey:
        # would raise an ObjC exception (a hard crash, not a Python one).
        is_mut = msg(info, "isKindOfClass:", NSMutableDictionary,
                     argtypes=[ctypes.c_void_p], restype=ctypes.c_bool)
        if not is_mut:
            return False
        msg(info, "setObject:forKey:", nsstr(name), nsstr("CFBundleName"),
            argtypes=[ctypes.c_void_p, ctypes.c_void_p], restype=None)
        return True
    except Exception:
        return False


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from .shell import AppShell

    _rename_macos_app(APP_NAME)               # before QApplication builds the menu bar
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    win = AppShell()          # launcher → LazyStretch / LazyStack / LazyMoonSun
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
