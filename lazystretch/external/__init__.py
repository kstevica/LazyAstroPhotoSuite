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
from .rcastro import BlurX, NoiseX, RCStarX
from .spcc import SPCC
from .starx import StarX, star_recombine

__all__ = ["GraXpert", "StarX", "DeepSNR", "SPCC", "BlurX", "NoiseX", "RCStarX",
           "star_recombine", "Tools"]


@dataclass
class Tools:
    """Resolved external-tool wrappers passed to the pipeline.

    When RC-Astro's stand-alone CLI is installed, its BlurX / StarX / NoiseX are the
    *real* products and take priority over the open-source substitutes (classical RL,
    StarNet, DeepSNR/GraXpert/MMT). ``blurx``/``sxt``/``noisex`` are the RC-Astro
    wrappers; ``starx`` is the StarNet fallback. ``star_tool()`` returns whichever star
    backend to use.
    """

    graxpert: GraXpert
    starx: StarX
    deepsnr: DeepSNR
    spcc: SPCC
    blurx: BlurX
    sxt: RCStarX
    noisex: NoiseX

    @classmethod
    def resolve(cls, *, graxpert_path: Optional[str] = None,
                starnet_path: Optional[str] = None, deepsnr_path: Optional[str] = None,
                rcastro_path: Optional[str] = None, gpu: bool = True) -> "Tools":
        return cls(GraXpert(graxpert_path, gpu=gpu), StarX(starnet_path),
                   DeepSNR(deepsnr_path), SPCC(), BlurX(rcastro_path),
                   RCStarX(rcastro_path), NoiseX(rcastro_path))

    def star_tool(self):
        """Preferred available star backend: RC-Astro StarX > StarNet > None."""
        if self.sxt.is_available():
            return self.sxt
        if self.starx.is_available():
            return self.starx
        return None

    def status(self) -> Dict[str, bool]:
        return {
            "BlurXTerminator (RC-Astro deconv)": self.blurx.is_available(),
            "StarXTerminator (RC-Astro stars)": self.sxt.is_available(),
            "NoiseXTerminator (RC-Astro NR)": self.noisex.is_available(),
            "GraXpert (background)": self.graxpert.is_available(),
            "StarNet (star reduction)": self.starx.is_available(),
            "DeepSNR (noise reduction)": self.deepsnr.is_available(),
            "SPCC-lite (color)": self.spcc.is_available(),
        }
