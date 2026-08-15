"""Frozen-app entry point for the PyInstaller build.

A top-level script (not a package submodule) with absolute imports, so it runs cleanly as
``__main__`` in the bundle. ``freeze_support()`` makes spawned multiprocessing workers
(LazyFlight's parallel render) run as workers instead of relaunching the whole app.
"""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from lazystretch.gui.app import main
    raise SystemExit(main())
