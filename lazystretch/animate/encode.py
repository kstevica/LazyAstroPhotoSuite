"""Encode a stream of float frames to an H.264 mp4.

Frames are streamed straight to ``ffmpeg`` over a pipe (no giant frame buffer in
RAM, no temp PNG sequence). ffmpeg is located on ``PATH`` first, then via the
optional ``imageio-ffmpeg`` wheel, so a plain ``pip install`` still works on a
machine without a system ffmpeg once that extra is added.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Iterable, Optional

import numpy as np


def _ffmpeg_exe() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ffmpeg_available() -> bool:
    return _ffmpeg_exe() is not None


def _to_u8(frame: np.ndarray) -> np.ndarray:
    a = np.asarray(frame)
    if a.dtype == np.uint8:
        u = a
    else:
        u = np.rint(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    if u.ndim == 2:
        u = np.repeat(u[..., None], 3, axis=2)
    return u


def write_video(frames: Iterable[np.ndarray], path: str, *, fps: int = 24,
                crf: int = 17) -> str:
    """Encode ``frames`` (float [0,1] or uint8, ``(H, W, 3)``) to ``path``.

    Returns the output path. Raises ``RuntimeError`` if no ffmpeg is found.
    Dimensions are cropped to even values (libx264 / yuv420p requirement).
    """
    it = iter(frames)
    try:
        first = _to_u8(next(it))
    except StopIteration:
        raise ValueError("no frames to encode")

    h, w = first.shape[:2]
    w -= w % 2
    h -= h % 2
    exe = _ffmpeg_exe()
    if exe is None:
        raise RuntimeError(
            "ffmpeg not found — install a system ffmpeg or `pip install "
            "imageio-ffmpeg` (LazyStretch's [video] extra).")

    cmd = [exe, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def _put(frame: np.ndarray) -> None:
        u = _to_u8(frame)[:h, :w]
        proc.stdin.write(np.ascontiguousarray(u).tobytes())

    try:
        _put(first)
        for fr in it:
            _put(fr)
    finally:
        if proc.stdin:
            proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", "ignore") if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {err.strip()[:500]}")
    return str(path)
