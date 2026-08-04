"""Deep-sky object catalog — solved centre -> ranked named candidates.

Ports ``DSOCatalog`` and ``LS.angularSepDeg`` (LazyStretch.js:420-428, 583-716). The
Messier / NGC-IC CSVs are bundled under ``data/catalogs/`` (converted once from PI's
AdP folder) so the app is self-contained. Unlike PI's naive ``split(",")``, this uses
a real CSV reader (quoted commas in common names parse correctly) while preserving the
exact column mapping.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

_CATALOG_DIR = Path(__file__).resolve().parent.parent / "data" / "catalogs"
DEG2RAD = math.pi / 180.0


def angular_sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation, degrees in / degrees out (haversine, js:420-428)."""
    d = DEG2RAD
    d_ra = (ra2 - ra1) * d
    d_dec = (dec2 - dec1) * d
    a = (math.sin(d_dec / 2.0) ** 2
         + math.cos(dec1 * d) * math.cos(dec2 * d) * math.sin(d_ra / 2.0) ** 2)
    return (2.0 * math.asin(min(1.0, math.sqrt(a)))) / d


@dataclass(frozen=True)
class CatalogRow:
    id: str
    ra: float          # RA degrees
    dec: float         # Dec degrees
    mag: float
    diameter: float    # arcmin
    axisRatio: float
    commonName: str
    source: str        # "M" or "NGC/IC"


@dataclass(frozen=True)
class Candidate:
    obj: CatalogRow
    sep: float         # degrees from field centre
    score: float


def _parse_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _load_csv(path: Path, source: str) -> List[CatalogRow]:
    rows: List[CatalogRow] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)  # id,alpha,delta,magnitude,diameter,axisRatio,posAngle,Common name,...
        for f in reader:
            if len(f) < 8:
                continue
            ra = _parse_float(f[1])
            dec = _parse_float(f[2])
            if not (math.isfinite(ra) and math.isfinite(dec)):
                continue
            rows.append(CatalogRow(
                id=f[0].strip(),
                ra=ra, dec=dec,
                mag=_parse_float(f[3]),
                diameter=_parse_float(f[4]),
                axisRatio=_parse_float(f[5]),
                commonName=(f[7] or "").strip(),
                source=source,
            ))
    return rows


class DSOCatalog:
    """Bundled Messier + NGC-IC catalog with field search and name lookup."""

    def __init__(self, catalog_dir: "str | Path | None" = None):
        self._dir = Path(catalog_dir) if catalog_dir else _CATALOG_DIR
        self._rows: Optional[List[CatalogRow]] = None

    @property
    def rows(self) -> List[CatalogRow]:
        if self._rows is None:
            self._rows = (_load_csv(self._dir / "Messier.csv", "M")
                          + _load_csv(self._dir / "NGC-IC.csv", "NGC/IC"))
        return self._rows

    def _prominence(self, r: CatalogRow, sep_deg: float, fov_radius_deg: float) -> float:
        """Larger, brighter, more central ranks higher (js:675-685)."""
        diam = r.diameter if math.isfinite(r.diameter) else 0.5           # arcmin
        field_arcmin = max(1e-3, fov_radius_deg * 60.0)
        size_score = min(1.0, diam / field_arcmin)
        mag = r.mag if math.isfinite(r.mag) else 15.0
        mag_score = max(0.0, (14.0 - mag) / 14.0)
        center_score = max(0.0, 1.0 - sep_deg / (fov_radius_deg + 1e-6))
        famous = 0.12 if r.source == "M" else 0.0
        return 0.50 * size_score + 0.28 * mag_score + 0.22 * center_score + famous

    def find_in_field(self, center_ra: float, center_dec: float,
                      fov_radius_deg: float) -> List[Candidate]:
        """Ranked candidates whose extent falls in the field, twins collapsed (js:688-716)."""
        out: List[Candidate] = []
        for r in self.rows:
            sep = angular_sep_deg(center_ra, center_dec, r.ra, r.dec)
            obj_radius_deg = (r.diameter if math.isfinite(r.diameter) else 0.0) / 2.0 / 60.0
            if sep <= fov_radius_deg + obj_radius_deg:
                out.append(Candidate(r, sep, self._prominence(r, sep, fov_radius_deg)))
        out.sort(key=lambda c: c.score, reverse=True)
        seen = set()
        deduped: List[Candidate] = []
        for c in out:
            key = (round(c.obj.ra * 100), round(c.obj.dec * 100))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        return deduped

    def find_by_name_or_id(self, query: Optional[str]) -> Optional[CatalogRow]:
        """Lookup by catalog id or common name (js:638-672)."""
        if not query:
            return None
        q = query.strip()
        if not q:
            return None
        parts = q.split("—")  # em dash
        id_part = parts[0].strip()
        name_part = (parts[1] if len(parts) > 1 else q).strip()
        q_id = id_part.upper().replace(" ", "")
        q_name = name_part.lower()

        by_name_exact = None
        by_name_part = None
        for r in self.rows:
            if r.id.upper().replace(" ", "") == q_id:
                return r  # exact id wins outright
            if r.commonName:
                cn = r.commonName.lower()
                if by_name_exact is None and cn == q_name:
                    by_name_exact = r
                elif (by_name_part is None and len(q_name) >= 3 and q_name in cn):
                    by_name_part = r
        return by_name_exact or by_name_part


@lru_cache(maxsize=1)
def get_catalog() -> DSOCatalog:
    """The bundled catalog, loaded once."""
    return DSOCatalog()


def object_label(r: Optional[CatalogRow]) -> str:
    """Display label for a candidate (js:722-728)."""
    if r is None:
        return "(none)"
    name = f" — {r.commonName}" if r.commonName else ""
    return r.id + name
