"""Object-class + data-type classification (LazyStretch.js:734-797, 898-942).

``object_class`` maps a catalog row to a processing class (drives the profile).
``detect_data_type`` labels the input from the FITS FILTER keyword + channel geometry.
Palette helpers mirror ``paletteKey`` / ``palettesFor``.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional

from ..data.loader import LazyStretchData, get_data
from .catalog import CatalogRow

_M_RE = re.compile(r"^M\s*0*([0-9]{1,3})$", re.IGNORECASE)
_ICNGC_RE = re.compile(r"^(IC|NGC)\s*0*([0-9]{1,4})$", re.IGNORECASE)

# Data-type detection regexes (js:746-751).
_NB_RE = re.compile(r"(^|[^a-z])(ha|h[- ]?alpha|sii|s2|oiii|o3|nii|n2)([^a-z]|$)")
_BB_RE = re.compile(r"(lum|^l$|red|^r$|green|^g$|blue|^b$)")
_OSC_RE = re.compile(r"(rgb|color|osc|bayer)")


def object_class(obj: Optional[CatalogRow], data: "LazyStretchData | None" = None) -> str:
    """Determine an object's processing class from its catalog row (js:898-942)."""
    if obj is None:
        return "generic"
    if data is None:
        data = get_data()
    cats = data.catalogs

    # 1) Messier number lookup (supernova bucket treated as emission).
    mm = _M_RE.match(obj.id or "")
    if mm:
        num = int(mm.group(1))
        for cls, nums in cats["messierClass"].items():
            if num in nums:
                return "emission" if cls == "supernova" else cls

    # 1b) NGC/IC class lookups — must run BEFORE the name test (js:913-925).
    ic = _ICNGC_RE.match(obj.id or "")
    if ic:
        cat = ic.group(1).upper()
        n = int(ic.group(2))
        if n in cats["reflectionCat"].get(cat, []):
            return "reflection"
        if n in cats["emissionCat"].get(cat, []):
            return "emission"

    # 2) Common-name keywords (order matters: reflection before the /nebula/ catch-all).
    cn = (obj.commonName or "").lower()
    if "galaxy" in cn:
        return "galaxy"
    if "globular" in cn:
        return "globular"
    if "planetary" in cn:
        return "planetary"
    if re.search(r"reflection|witch head|\biris\b", cn):
        return "reflection"
    if "cluster" in cn:
        return "open"
    if "nebula" in cn:
        return "emission"

    # 3) Weak geometric heuristic: very elongated + small -> likely a galaxy.
    if (math.isfinite(obj.axisRatio) and obj.axisRatio >= 2.2
            and math.isfinite(obj.diameter) and obj.diameter < 15):
        return "galaxy"

    return "generic"


def detect_data_type(filter_kw: Optional[str], is_color: bool,
                     bayer_kw: Optional[str] = None) -> str:
    """Return 'narrowband' | 'broadband' | 'osc' | 'auto' (js:734-759)."""
    if filter_kw:
        f = str(filter_kw).lower()
        if _NB_RE.search(f):
            return "narrowband"
        if _BB_RE.search(f):
            return "broadband"
        if _OSC_RE.search(f):
            return "osc"
    if is_color:
        return "osc"
    return "broadband"


def palettes_for(data_type: str) -> List[str]:
    """Palettes offered per data type (js:773-786)."""
    return {
        "narrowband": ["SHO (Hubble)", "HOO", "HOS"],
        "osc": ["Natural RGB", "SHO (Hubble)", "HOO"],
        "broadband": ["RGB", "LRGB"],
    }.get(data_type, ["RGB", "LRGB", "SHO (Hubble)", "HOO", "HOS"])


def palette_key(palette: Optional[str]) -> Optional[str]:
    """Map a palette display string to a mono-combine key, or None (js:790-797)."""
    if not palette:
        return None
    if re.match(r"^SHO", palette, re.IGNORECASE):
        return "SHO"
    if re.match(r"^HOS", palette, re.IGNORECASE):
        return "HOS"
    if re.match(r"^(HOO|Bicolor)", palette, re.IGNORECASE):
        return "HOO"
    return None
