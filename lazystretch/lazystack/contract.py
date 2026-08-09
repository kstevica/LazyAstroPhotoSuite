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
