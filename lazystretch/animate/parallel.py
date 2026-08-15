"""Render frames across processes — the numpy/scipy work is GIL-bound, so a
process pool (not threads) is what actually speeds it up.

The engine (depth relief, starless sheet, star cloud) is built once and shipped
to each worker a single time via the pool initializer; each task then only sends
a tiny :class:`~.render.Cam` and returns a uint8 frame. Frames are yielded in
camera order so they stream straight into the encoder.
"""
from __future__ import annotations

import os
from typing import Callable, Iterable, List, Optional

import numpy as np

_WORKER_ENGINE = None                                    # per-process engine copy


def _init_worker(engine) -> None:
    global _WORKER_ENGINE
    _WORKER_ENGINE = engine


def _render_one(cam):
    fr = _WORKER_ENGINE.render_frame(cam)                # type: ignore[union-attr]
    return np.rint(np.clip(fr, 0.0, 1.0) * 255.0).astype(np.uint8)


def auto_workers(cap: int = 8) -> int:
    """A sensible default worker count — leave a core for the UI/encoder."""
    return max(1, min((os.cpu_count() or 2) - 1, cap))


def _sequential(engine, cams, on_frame):
    n = len(cams)
    for i, cam in enumerate(cams):
        fr = np.rint(np.clip(engine.render_frame(cam), 0, 1) * 255).astype(np.uint8)
        if on_frame is not None:
            on_frame(i, n, fr)
        yield fr


def parallel_frames(engine, cams: List, workers: int,
                    on_frame: Optional[Callable[[int, int, np.ndarray], None]] = None
                    ) -> Iterable[np.ndarray]:
    """Yield rendered uint8 frames for ``cams`` in order.

    Uses ``workers`` processes (frames are independent). Falls back to a plain
    loop when ``workers <= 1`` or if the process pool cannot start — the latter
    happens when a top-level script drives this without an ``if __name__ ==
    "__main__"`` guard, so the render still completes rather than crashing.
    """
    import concurrent.futures as cf

    n = len(cams)
    workers = max(1, min(int(workers), n))
    if workers == 1:
        yield from _sequential(engine, cams, on_frame)
        return

    chunk = max(1, n // (workers * 8))
    ex = cf.ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                initargs=(engine,))
    try:
        results = ex.map(_render_one, cams, chunksize=chunk)
        try:
            first = next(results)                        # surfaces pool-start errors here
        except StopIteration:
            return
        except Exception:                                # pool broke before any frame
            ex.shutdown(cancel_futures=True)
            yield from _sequential(engine, cams, on_frame)
            return
        if on_frame is not None:
            on_frame(0, n, first)
        yield first
        for i, frame in enumerate(results, start=1):
            if on_frame is not None:
                on_frame(i, n, frame)
            yield frame
    finally:
        ex.shutdown()
