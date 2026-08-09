"""LazyMoonSun dial set + presets — ported from ``SUN`` defaults and the preset literals.

Field-locked 2026-08 defaults (LazyMoonSun.js:204-244); the Sun preset IS the factory
default, Moon (:254-255) and Neutral base (:1829-1831) overlay a few keys.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class MoonSunParams:
    # global stack + finish
    keep: float = 0.50          # KEEP — best fraction kept after grading
    sharp: float = 0.30         # SHARP — wavelet sharpen amount
    disc_target: float = 0.72   # DISC_TARGET — disc median after the MTF lift
    do_wb: bool = True          # DO_WB — neutralize the filter tint
    auto_finish: bool = True    # AUTO_FINISH — finish right after stacking
    surface: float = 6.0        # SURFACE — residual-contrast (limb-flatten) boost
    flatten: float = 0.35       # FLATTEN — fraction of limb darkening removed
    tone: str = "golden"        # TONE — "neutral" | "golden"
    contrast: float = 0.0       # CONTRAST — finishing S-curve strength
    lhe: float = 0.0            # LHE — local contrast amount (disc-masked)
    sat: float = 0.0            # SAT — saturation boost after tone
    ht_sh: float = 0.0          # HT_SH — final black point
    ht_mid: float = 0.559       # HT_MID — final midtones
    ht_hi: float = 0.858        # HT_HI — final highlights point
    # multi-point engine
    ap_size: float = 96.0       # APSIZE — alignment-point size (full-res px)
    ap_keep: float = 0.50       # APKEEP — per-AP best fraction
    ap_search: float = 8.0      # APSEARCH — max accepted local residual (px)
    ap_max: float = 600.0       # APMAX — safety cap on alignment points

    @classmethod
    def preset(cls, name: str) -> "MoonSunParams":
        """Return a params set for 'sun' | 'moon' | 'neutral' (unknown → sun/factory)."""
        base = cls()                                    # Sun = factory defaults
        n = (name or "").strip().lower()
        if n == "moon":
            return replace(base, sharp=0.45, surface=0.0, flatten=0.0, lhe=0.40,
                           sat=0.60, ht_sh=0.0, ht_mid=0.50, ht_hi=1.00, tone="neutral")
        if n in ("neutral", "neutral base", "neutral-base"):
            return replace(base, sharp=0.20, surface=0.0, flatten=0.0, contrast=0.0,
                           lhe=0.0, sat=0.0, ht_sh=0.0, ht_mid=0.50, ht_hi=1.00,
                           tone="neutral", disc_target=0.60)
        return base
