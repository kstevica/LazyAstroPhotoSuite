"""Target identification: plate solving, DSO catalog, class + data-type."""
from .catalog import Candidate, CatalogRow, DSOCatalog, angular_sep_deg, get_catalog, object_label
from .classify import detect_data_type, object_class, palette_key, palettes_for
from .solver import SolveResult, solve, solve_from_header, solve_with_astap

__all__ = [
    "Candidate", "CatalogRow", "DSOCatalog", "angular_sep_deg", "get_catalog", "object_label",
    "detect_data_type", "object_class", "palette_key", "palettes_for",
    "SolveResult", "solve", "solve_from_header", "solve_with_astap",
]
