"""Pipeline state — the ``LazyStretchParameters`` object (LazyStretch.js:105-150).

Holds the object class, palette, five adjustment dials, and every pipeline toggle,
with the exact PI defaults. ``for_object`` mirrors PI's ``applyClassProfileToOptions``:
selecting a class overwrites the six profile-driven toggles (SCNR / local-contrast /
BXT / NR / HDR / star-reduce) from that class's profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, Optional

import numpy as np

from ..data.loader import LazyStretchData, get_data


@dataclass
class Parameters:
    # --- identity ---
    object_class: str = "generic"
    palette: str = ""
    data_type: str = "auto"
    object_id: str = ""
    object_name: str = ""

    # --- five adjustment dials (nudge the class profile; 0 = class default) ---
    satAdj: float = 0.0
    brightAdj: float = 0.0
    bgAdj: float = 0.0
    blackAdj: float = 0.0
    contrastAdj: float = 0.0

    # --- extra dials ---
    cropPercent: float = 3.0
    gradientCleanup: float = 0.0
    dehaze: float = 0.0
    smallStars: float = 0.0

    # --- pipeline toggles (PI defaults; profile-driven ones set by for_object) ---
    autoAssess: bool = True
    inputStretched: bool = False
    doBgExtract: bool = True
    useGradientCorrection: bool = False
    doColorCal: bool = True
    doSCNR: bool = True
    doLocalContrast: bool = True
    doBXT: bool = True
    doNR: bool = True
    doHDR: bool = False
    doStarReduce: bool = False
    haloTamer: bool = True          # reflection filter-ring suppression (profile-driven)
    useMask: bool = False
    starProtect: bool = False
    preferSPCC: bool = True
    useNRMask: bool = False
    protectCores: bool = False
    adaptiveFloor: bool = True
    enhanceEmission: bool = False
    reduceCast: bool = False
    workOnClone: bool = True
    useClassicalDeconv: bool = False   # opt-in Richardson-Lucy when BlurX is absent (weak)

    # --- v1.4.x params: data-layer synced (recipes/memory round-trip) but their
    #     pipeline steps are not yet ported — see PLAN §12 catch-up backlog. ---
    darkLaneGC: bool = False       # dark-lane-anchored deg-2 gradient model
    removeStars: bool = False      # full star removal (starless result + stars layer)
    chromaNR: float = 0.0          # chroma-only NR on CIE a*/b* (luminance untouched)
    deepen: float = 0.0            # 2nd-pass stretch for already-stretched input
    starsAdj: float = 0.0          # star-reduction nudge (effLevel = class starLevel + this)
    highlights: float = 0.20       # highlight roll-off dial 0..1 (0=dialed fully down/dim,
                                   # 1=bright/untouched). Low pulls bright cores / blown
                                   # stars down + reveals core structure. Default 0.20 =
                                   # dialed down. Not a PI param (port-only finishing dial).
    transparency: float = 0.0      # core-transparency dial 0..1 (0=off, GATED OFF by default):
                                   # large-scale local contrast masked to bright nebulosity.
                                   # Over-cranking crushes dust lanes to black ink-blots (the
                                   # "#2" over-process), so it is opt-in — code kept, off by
                                   # default. Port-only finishing dial; work with Highlights first.
    starShrink: float = 0.0        # software star reduction 0..1 (0=off): thin the star carpet
                                   # morphologically (no StarNet). Reveals dust/nebulosity on
                                   # dense widefields. Port-only finishing dial.
    dimCore: float = 0.0           # dim-core dial 0..1 (0=off): mask the large bright veil and
                                   # multiplicatively lower its luminosity (no ceiling, unlike
                                   # the highlight roll-off). Opt-in. Port-only finishing dial.

    # --- transient narrowband channel arrays (not persisted) ---
    ha: Optional[np.ndarray] = field(default=None, repr=False)
    oiii: Optional[np.ndarray] = field(default=None, repr=False)
    sii: Optional[np.ndarray] = field(default=None, repr=False)

    def slider_dict(self) -> Dict[str, float]:
        return {
            "satAdj": self.satAdj, "brightAdj": self.brightAdj, "bgAdj": self.bgAdj,
            "blackAdj": self.blackAdj, "contrastAdj": self.contrastAdj,
        }

    @classmethod
    def for_object(cls, object_class: str = "generic",
                   data: "LazyStretchData | None" = None, **overrides) -> "Parameters":
        """Build parameters for a class, applying the class profile's six toggles first.

        Mirrors ``applyClassProfileToOptions``: class selection sets doSCNR /
        doLocalContrast / doBXT / doNR / doHDR / doStarReduce from the profile.
        Explicit ``overrides`` win over the profile-driven values.
        """
        if data is None:
            data = get_data()
        prof = data.profile_for(object_class)
        p = cls(object_class=object_class)
        p.doSCNR = prof.scnr
        p.doLocalContrast = prof.localContrast
        p.doBXT = prof.bxt
        p.doNR = prof.nr
        p.doHDR = prof.hdr
        p.doStarReduce = prof.starReduce
        p.haloTamer = prof.haloTamer
        p.chromaNR = prof.chromaNR      # OSC broadband seeds a modest chroma-NR default
        p.doBgExtract = prof.bgExtract  # MW uses darkLaneGC instead of ill-posed ABE
        p.darkLaneGC = prof.darkLaneGC
        p.reduceCast = prof.reduceCast  # masked post-SCNR background re-neutralization
        valid = {f.name for f in fields(cls)}
        for k, v in overrides.items():
            if k not in valid:
                raise TypeError(f"unknown parameter: {k!r}")
            setattr(p, k, v)
        return p
