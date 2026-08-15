"""Test configuration.

Make external-tool discovery **hermetic**: unit tests must behave the same whether or not
the real StarNet2 / GraXpert / DeepSNR / RC-Astro binaries are installed on the machine
(this dev box really does have the RC-Astro CLI on PATH). This fixture disables
auto-discovery (PATH lookup, the GraXpert .app / RC-Astro install-dir fallbacks, and the
LAZYSTRETCH_* env vars), so a tool is only "available" when an explicit, existing path is
passed to its constructor — exactly what the fake-tool tests do. Tests that want the real
tools pass an explicit path, which still resolves first.
"""
import shutil

import pytest

_TOOL_EXE_NAMES = {
    "starnet2", "starnet++", "StarNet", "starnet",
    "graxpert", "GraXpert", "GraXpert-win64.exe",
    "deepsnr", "DeepSNR",
    "rc-astro", "RC-Astro", "rc-astro.exe",
}


@pytest.fixture(autouse=True)
def hermetic_external_tools(monkeypatch):
    from lazystretch.external import rcastro
    from lazystretch.external.graxpert import GraXpert

    # No app-bundle / install-dir fallback during tests.
    monkeypatch.setattr(GraXpert, "_APP_PATHS", [])
    monkeypatch.setattr(rcastro, "_APP_PATHS", [])
    # No env-var discovery.
    for var in ("LAZYSTRETCH_STARNET", "LAZYSTRETCH_GRAXPERT", "LAZYSTRETCH_DEEPSNR",
                "LAZYSTRETCH_RCASTRO"):
        monkeypatch.delenv(var, raising=False)
    # No PATH discovery of the tool executables (explicit constructor paths still resolve).
    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which",
        lambda name, *a, **k: None if name in _TOOL_EXE_NAMES else real_which(name, *a, **k),
    )
