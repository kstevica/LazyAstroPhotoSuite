"""DevelopDocument — the edit document behind the Develop window.

A document holds one base image plus an ordered list of applied tools (the *op
history*), a PixInsight-style **linear** history (apply → undo → redo → revert-to-step)
and a small store of named masks. Each applied op can be gated by a mask and an
opacity; the blend is exactly the Lighthouse form ``orig + w·mask·(proc − orig)``.

Intermediate results are cached (``_cache[i]`` = the image after the first ``i`` ops)
so undo is instant and re-running from any step is cheap. Everything is float32 in
``[0, 1]``, shape ``(H, W)`` mono or ``(H, W, 3)`` RGB (channel order R, G, B).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def _as_f32(a: np.ndarray) -> np.ndarray:
    """Coerce to a contiguous float32 array in [0, 1], stripping any alpha channel."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    return np.clip(np.ascontiguousarray(a), 0.0, 1.0)


def _clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)


def _resize2d(m: np.ndarray, shape_hw) -> np.ndarray:
    """Nearest/bilinear resize a 2-D mask to ``shape_hw`` (H, W). No SciPy dependency."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if m.shape[0] == h and m.shape[1] == w:
        return m
    ys = np.linspace(0, m.shape[0] - 1, h)
    xs = np.linspace(0, m.shape[1] - 1, w)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, m.shape[0] - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, m.shape[1] - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    top = m[y0][:, x0] * (1 - wx) + m[y0][:, x1] * wx
    bot = m[y1][:, x0] * (1 - wx) + m[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


@dataclass
class OpInstance:
    """One applied tool in the history: a registry op name, its params, and a gate."""
    name: str
    params: dict = field(default_factory=dict)
    mask: Optional[str] = None          # named mask in the document (None = whole frame)
    mask_invert: bool = False
    opacity: float = 1.0                # 0..1 global blend strength
    enabled: bool = True                # kept for future non-linear toggling

    def title(self) -> str:
        from .ops import get
        try:
            base = get(self.name).label
        except KeyError:
            base = self.name
        bits = []
        if self.opacity < 0.999:
            bits.append(f"{int(round(self.opacity * 100))}%")
        if self.mask:
            bits.append(("¬" if self.mask_invert else "") + self.mask)
        return base + (f"  ({', '.join(bits)})" if bits else "")


class DevelopDocument:
    """Base image + ordered op history + named masks, with a linear undo/redo model."""

    def __init__(self, base: np.ndarray, *, path: Optional[str] = None,
                 header: Optional[dict] = None):
        self.base = _as_f32(base)
        self.path = path
        self.header: dict = dict(header or {})
        self.ops: List[OpInstance] = []
        self._cache: List[np.ndarray] = [self.base]     # cache[i] = after ops[:i]
        self._redo: List[OpInstance] = []
        self.masks: Dict[str, np.ndarray] = {}

    # -------------------------------------------------------------- properties
    @property
    def is_color(self) -> bool:
        return self.base.ndim == 3

    @property
    def shape(self):
        return self.result().shape

    def result(self) -> np.ndarray:
        """The current fully-processed image (float32 [0,1])."""
        return self._cache[-1]

    def can_undo(self) -> bool:
        return len(self.ops) > 0

    def can_redo(self) -> bool:
        return len(self._redo) > 0

    # ------------------------------------------------------------------- masks
    def add_mask(self, name: str, mask2d: np.ndarray) -> None:
        m = np.clip(np.asarray(mask2d, dtype=np.float32), 0.0, 1.0)
        if m.ndim == 3:
            m = m.mean(axis=2)
        self.masks[name] = m

    def remove_mask(self, name: str) -> None:
        self.masks.pop(name, None)

    def mask_names(self) -> List[str]:
        return list(self.masks.keys())

    def _mask_for(self, name: str, shape_hw) -> Optional[np.ndarray]:
        m = self.masks.get(name)
        if m is None:
            return None
        return _resize2d(m, shape_hw)

    # ---------------------------------------------------------------- op engine
    def _run_one(self, img: np.ndarray, op: OpInstance) -> np.ndarray:
        """Apply one op to ``img`` with its mask/opacity gate; returns a new array."""
        from .ops import get
        spec = get(op.name)
        proc = _clip01(spec.fn(img, op.params))
        # Ops may change resolution (crop): a gated blend only makes sense when the
        # processed result matches the input footprint.
        same_shape = proc.shape[:2] == img.shape[:2]
        if (op.opacity >= 0.999 and op.mask is None) or not same_shape:
            return proc
        return self._blend(img, proc, op)

    def _blend(self, orig: np.ndarray, proc: np.ndarray, op: OpInstance) -> np.ndarray:
        w = float(np.clip(op.opacity, 0.0, 1.0))
        weight: object = w
        if op.mask:
            m = self._mask_for(op.mask, orig.shape[:2])
            if m is not None:
                if op.mask_invert:
                    m = 1.0 - m
                weight = w * m
        if orig.ndim == 3 and isinstance(weight, np.ndarray) and weight.ndim == 2:
            weight = weight[..., None]
        return _clip01(orig + weight * (proc - orig))

    def apply_op(self, name: str, params: Optional[dict] = None, *,
                 mask: Optional[str] = None, mask_invert: bool = False,
                 opacity: float = 1.0) -> np.ndarray:
        """Run a tool on the current result and push it onto the history."""
        op = OpInstance(name, dict(params or {}), mask, mask_invert, float(opacity))
        newimg = self._run_one(self.result(), op)
        self.ops.append(op)
        self._cache.append(newimg)
        self._redo.clear()
        return newimg

    def preview_op(self, name: str, params: Optional[dict] = None, *,
                   mask: Optional[str] = None, mask_invert: bool = False,
                   opacity: float = 1.0) -> np.ndarray:
        """Compute the result of applying a tool WITHOUT committing it (live preview)."""
        op = OpInstance(name, dict(params or {}), mask, mask_invert, float(opacity))
        return self._run_one(self.result(), op)

    def undo(self) -> bool:
        if not self.ops:
            return False
        self._redo.append(self.ops.pop())
        self._cache.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        op = self._redo.pop()
        self._cache.append(self._run_one(self.result(), op))
        self.ops.append(op)
        return True

    def revert_to(self, n_ops: int) -> None:
        """Undo down to exactly ``n_ops`` applied ops (keeps them redoable)."""
        n_ops = max(0, min(int(n_ops), len(self.ops)))
        while len(self.ops) > n_ops:
            self.undo()

    def reset(self) -> None:
        """Drop all ops and redo history; keep the base image and masks."""
        self.ops.clear()
        self._redo.clear()
        self._cache = [self.base]

    # ------------------------------------------------------------------- recipe
    def to_recipe(self) -> List[dict]:
        """Serialise the op history (masks are not serialised — they are regenerated)."""
        return [
            {"name": op.name, "params": op.params, "mask": op.mask,
             "mask_invert": op.mask_invert, "opacity": op.opacity}
            for op in self.ops
        ]

    def apply_recipe(self, recipe: List[dict]) -> None:
        """Replay a serialised op history on top of the current state."""
        for step in recipe:
            self.apply_op(
                step["name"], step.get("params", {}),
                mask=step.get("mask"), mask_invert=bool(step.get("mask_invert", False)),
                opacity=float(step.get("opacity", 1.0)),
            )
