"""External / proprietary tools behind a uniform, feature-detected boundary (PLAN §7).

The "AI wall": GraXpert (denoise + background), StarNet (star removal/reduction/mask),
and SPCC-lite. None are bundled; each is discovered at runtime (explicit path -> env var
-> PATH) and **degrades gracefully when absent** — mirroring PI's ``haveBlurX``/
``haveStarX``/… guards. Deconvolution has no open CLI equivalent, so it uses a local
classical method (``processes.deconv``), off by default.

``Tools`` bundles the resolved wrappers and is passed to the pipeline; ``Tools.status()``
reports what's installed (surfaced by the CLI/GUI).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .deepsnr import DeepSNR
from .graxpert import GraXpert
from .spcc import SPCC
from .starx import StarX, star_recombine

__all__ = ["GraXpert", "StarX", "DeepSNR", "SPCC", "star_recombine", "Tools"]


@dataclass
class Tools:
    """Resolved external-tool wrappers passed to the pipeline."""

    graxpert: GraXpert
    starx: StarX
    deepsnr: DeepSNR
    spcc: SPCC

    @classmethod
    def resolve(cls, *, graxpert_path: Optional[str] = None,
                starnet_path: Optional[str] = None, deepsnr_path: Optional[str] = None,
                gpu: bool = True) -> "Tools":
        return cls(GraXpert(graxpert_path, gpu=gpu), StarX(starnet_path),
                   DeepSNR(deepsnr_path), SPCC())

    def status(self) -> Dict[str, bool]:
        return {
            "GraXpert (background)": self.graxpert.is_available(),
            "StarNet (star reduction)": self.starx.is_available(),
            "DeepSNR (noise reduction)": self.deepsnr.is_available(),
            "SPCC-lite (color)": self.spcc.is_available(),
        }
