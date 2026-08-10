"""Edge-contract measurement + LZS* keywords (STK.measureEdges/stampContract).

After integration the frame has junk edges (registration leaves black borders where subs
didn't overlap). ``measure_edges`` finds how many rows/cols on each side are junk (dark) and
returns the crop the Stretch tab reads from the ``LZSCROP*`` keywords, so the finish crops by
the measured bounds instead of a guessed percentage.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

EDGE_FLOOR = 0.02          # STK.EDGE_FLOOR: always trim at least 2% as a safety margin


def _lum(img: np.ndarray) -> np.ndarray:
    return img[..., :3].mean(axis=2) if img.ndim == 3 else img


def measure_edges(master: np.ndarray) -> Dict[str, int]:
    """Return {'L','R','T','B'} junk-edge pixel counts (dark-scan + 2% floor)."""
    lum = _lum(np.asarray(master, dtype=np.float64))
    H, W = lum.shape
    interior_med = float(np.median(lum[H // 4:3 * H // 4, W // 4:3 * W // 4]))
    dark = max(1e-4, 0.25 * interior_med)          # a row/col is "junk" below this

    def _scan(profile: np.ndarray, limit: int) -> int:
        n = 0
        for v in profile:
            if v < dark and n < limit:
                n += 1
            else:
                break
        return n

    col_mean = lum.mean(axis=0)
    row_mean = lum.mean(axis=1)
    L = _scan(col_mean, W // 4)
    R = _scan(col_mean[::-1], W // 4)
    T = _scan(row_mean, H // 4)
    B = _scan(row_mean[::-1], H // 4)
    # 2% safety floor on every side
    L = max(L, int(EDGE_FLOOR * W))
    R = max(R, int(EDGE_FLOOR * W))
    T = max(T, int(EDGE_FLOOR * H))
    B = max(B, int(EDGE_FLOOR * H))
    return {"L": L, "R": R, "T": T, "B": B}


def coverage_edges(coverage: np.ndarray, n_sub: int, *,
                   cover_frac: float = 0.98, line_frac: float = 0.90) -> Dict[str, int]:
    """Common-overlap junk-edge counts from the integration coverage map.

    ``coverage`` is the per-pixel count of frames that actually covered each pixel. A row/col
    is "junk" if fewer than ``line_frac`` of its pixels reach ``cover_frac``·N coverage — i.e.
    it is mostly outside the region every (or nearly every) frame overlaps. Leading/trailing
    junk rows/cols on each side become the crop. An undithered set (uniform full coverage)
    yields zero crop; a dithered/rotated set trims exactly the ragged partial-coverage border
    the reviewers saw. Never trims more than 25% per side (degenerate-coverage guard).
    """
    cov = np.asarray(coverage)
    H, W = cov.shape
    thr = max(1, int(round(cover_frac * max(1, n_sub))))
    covered = cov >= thr
    row_ok = covered.mean(axis=1) >= line_frac
    col_ok = covered.mean(axis=0) >= line_frac

    def _lead(flags: np.ndarray) -> int:
        n = 0
        for ok in flags:
            if ok:
                break
            n += 1
        return n

    T, B = _lead(row_ok), _lead(row_ok[::-1])
    L, R = _lead(col_ok), _lead(col_ok[::-1])
    return {"L": min(L, W // 4), "R": min(R, W // 4),
            "T": min(T, H // 4), "B": min(B, H // 4)}


def crop_to_contract(master: np.ndarray, edges: Dict[str, int]) -> np.ndarray:
    """Crop a master by measured edge bounds."""
    a = np.asarray(master)
    H, W = a.shape[0], a.shape[1]
    return a[edges["T"]:H - edges["B"], edges["L"]:W - edges["R"]]


def contract_header(n_sub: int, edges: Dict[str, int], exposure: float = 0.0) -> Dict[str, object]:
    """The LZS* FITS keywords the LazyStretch tab reads (the 'contract')."""
    return {
        "LZSVER": "0.3.0",
        "LZSNSUB": int(n_sub),
        "LZSEXP": float(exposure),
        "LZSCROPL": edges["L"], "LZSCROPR": edges["R"],
        "LZSCROPT": edges["T"], "LZSCROPB": edges["B"],
    }
