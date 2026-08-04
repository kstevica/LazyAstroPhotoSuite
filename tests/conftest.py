"""Test configuration.

Make external-tool discovery **hermetic**: unit tests must behave the same whether or not
the real StarNet2 / GraXpert / DeepSNR binaries are installed on the machine. This fixture
disables auto-discovery (PATH lookup, the GraXpert .app fallback, and the LAZYSTRETCH_*
env vars), so a tool is only "available" when an explicit, existing path is passed to its
constructor — exactly what the fake-tool tests do. Tests that want real tools would opt out.
"""
import shutil

import pytest

_TOOL_EXE_NAMES = {
    "starnet2", "starnet++", "StarNet", "starnet",
    "graxpert", "GraXpert", "GraXpert-win64.exe",
    "deepsnr", "DeepSNR",
}


@pytest.fixture(autouse=True)
def hermetic_external_tools(monkeypatch):
    from lazystretch.external.graxpert import GraXpert

    # No app-bundle fallback during tests.
    monkeypatch.setattr(GraXpert, "_APP_PATHS", [])
    # No env-var discovery.
    for var in ("LAZYSTRETCH_STARNET", "LAZYSTRETCH_GRAXPERT", "LAZYSTRETCH_DEEPSNR"):
        monkeypatch.delenv(var, raising=False)
    # No PATH discovery of the tool executables (explicit constructor paths still resolve).
    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which",
        lambda name, *a, **k: None if name in _TOOL_EXE_NAMES else real_which(name, *a, **k),
    )
