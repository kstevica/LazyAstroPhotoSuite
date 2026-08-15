"""High-level entry point: a finished still in, a fly-through mp4 out."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from . import camera as _camera
from .depth import _as_rgb
from .encode import write_video
from .render import Cam, Flythrough3D

PATHS: Dict[str, Callable[..., List[Cam]]] = {
    "flythrough": _camera.flythrough,
    "flyby": _camera.flyby,
    "orbit": _camera.orbit,
}


def _resize(rgb: np.ndarray, target_w: int) -> np.ndarray:
    """Downscale to ``target_w`` (keeps aspect) with an antialias prefilter."""
    from scipy.ndimage import gaussian_filter, zoom

    h, w = rgb.shape[:2]
    if target_w >= w:
        return rgb
    f = target_w / float(w)
    sig = (1.0 / f - 1.0) * 0.5
    src = rgb
    if sig > 0.4:
        src = np.stack([gaussian_filter(rgb[..., c], sig)
                        for c in range(rgb.shape[2])], axis=-1)
    return np.clip(zoom(src, (f, f, 1.0), order=1), 0.0, 1.0).astype(np.float32)


def build_cameras(path: str, n_frames: int, **kw) -> List[Cam]:
    if path not in PATHS:
        raise ValueError(f"unknown camera path {path!r}; choose from {list(PATHS)}")
    return PATHS[path](n_frames, **kw)


def render_flythrough(img: np.ndarray, out_path: str, *, seconds: float = 8.0,
                      fps: int = 24, path: str = "flythrough",
                      render_width: int = 1280,
                      masks: Optional[Dict[str, np.ndarray]] = None,
                      engine_kw: Optional[dict] = None,
                      path_kw: Optional[dict] = None,
                      on_frame: Optional[Callable[[int, int, np.ndarray], None]] = None
                      ) -> str:
    """Render ``img`` (float [0,1] mono/RGB) to an mp4 at ``out_path``.

    ``on_frame(i, n, frame)`` is called per frame for progress / preview. Returns
    the written path.
    """
    rgb = _resize(_as_rgb(img), int(render_width))
    engine = Flythrough3D(rgb, masks=masks, **(engine_kw or {}))
    n = max(int(round(seconds * fps)), 2)
    cams = build_cameras(path, n, **(path_kw or {}))

    def frames():
        for i, cam in enumerate(cams):
            fr = engine.render_frame(cam)
            if on_frame is not None:
                on_frame(i, n, fr)
            yield fr

    return write_video(frames(), out_path, fps=fps)
