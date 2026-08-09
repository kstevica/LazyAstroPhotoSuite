"""LazyMoonSun — lucky-imaging burst stacking & finishing for the Sun and Moon.

A faithful port of the standalone PixInsight ``LazyMoonSun.js`` 1.0.0 engine. The
registration and stacking are self-contained FFT phase-correlation math (no PixInsight
processes, no star detection): ``register`` holds the primitives, ``stack`` the global and
multi-point engines, ``finish`` the deterministic Sun/Moon finish, and ``model`` the dial
set + presets.
"""
from __future__ import annotations

__all__ = ["register", "stack", "finish", "model"]
