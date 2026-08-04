"""SpectrophotometricColorCalibration — SPCC-lite scaffold (PLAN §7, §11).

Full SPCC-lite (solve -> Gaia DR3 XP synthetic photometry -> per-channel calibration)
is the hardest, multi-week piece and needs GaiaXPy + a Gaia mirror or network. It is
scaffolded and feature-detected here; until the synthetic-photometry chain is wired,
``calibrate`` returns ``None`` so the pipeline uses the honest default,
BackgroundNeutralization + ColorCalibration (``processes.colorcal``).

``is_available`` reports whether the Python prerequisites (astroquery + gaiaxpy) are
importable; a real solve is still required at call time.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class SPCC:
    name = "SpectrophotometricColorCalibration (SPCC-lite)"

    def is_available(self) -> bool:
        try:
            import astroquery  # noqa: F401
            import gaiaxpy  # noqa: F401
            return True
        except ImportError:
            return False

    def calibrate(self, img: np.ndarray, solve_result) -> Optional[np.ndarray]:
        """Attempt Gaia-anchored calibration; ``None`` -> caller falls back to BN+CC.

        Returns ``None`` (with the full synthetic-photometry chain not yet wired) so the
        pipeline degrades to BN+CC, exactly as PI's SPCC does when the Gaia DB/solve is
        missing. The prerequisites and solve are validated so this is ready to grow into
        the real implementation.
        """
        if not self.is_available() or solve_result is None or not getattr(solve_result, "solved", False):
            return None
        # TODO(P5+): Gaia DR3 cone search -> XP synthetic RGB via GaiaXPy -> star detection
        # + cross-match -> robust per-channel scale. Until then, defer to BN+CC.
        return None
