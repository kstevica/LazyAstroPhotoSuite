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
        # Non-destructive linear history. ``ops`` holds EVERY step (the full recipe);
        # ``cursor`` is the "you are here" position (how many ops are currently applied).
        # Navigating back moves the cursor without discarding the steps after it.
        self.ops: List[OpInstance] = []
        self.cursor: int = 0
        self._cache: List[np.ndarray] = [self.base]     # cache[i] = result after ops[:i]
        self.masks: Dict[str, np.ndarray] = {}

    # -------------------------------------------------------------- properties
    @property
    def is_color(self) -> bool:
        return self.base.ndim == 3

    @property
    def shape(self):
        return self.result().shape

    def result(self) -> np.ndarray:
        """The image at the current history position (float32 [0,1])."""
        return self._cache[self.cursor]

    def can_undo(self) -> bool:
        return self.cursor > 0

    def can_redo(self) -> bool:
        return self.cursor < len(self.ops)

    def op_params(self, index: int) -> dict:
        """The stored params + gate for step ``index`` (for populating its tool panel)."""
        op = self.ops[index]
        return {"params": dict(op.params), "mask": op.mask,
                "mask_invert": op.mask_invert, "opacity": op.opacity}

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

    def _recompute_from(self, start: int) -> None:
        """Rebuild the cache for ops[start:] (``_cache[start]`` must already be valid)."""
        start = max(0, min(start, len(self.ops)))
        self._cache = self._cache[:start + 1]
        img = self._cache[start]
        for i in range(start, len(self.ops)):
            img = self._run_one(img, self.ops[i])
            self._cache.append(img)

    def apply_op(self, name: str, params: Optional[dict] = None, *,
                 mask: Optional[str] = None, mask_invert: bool = False,
                 opacity: float = 1.0) -> np.ndarray:
        """Insert a tool at the current position and advance the cursor onto it.

        Inserting at the cursor (not always the end) means a new adjustment made while
        you are back in the history is added *there* and the later steps are kept and
        recomputed on top — nothing is discarded.
        """
        op = OpInstance(name, dict(params or {}), mask, mask_invert, float(opacity))
        idx = self.cursor
        self.ops.insert(idx, op)
        self._recompute_from(idx)
        self.cursor = idx + 1
        return self.result()

    def preview_op(self, name: str, params: Optional[dict] = None, *,
                   mask: Optional[str] = None, mask_invert: bool = False,
                   opacity: float = 1.0) -> np.ndarray:
        """Result of applying a tool on top of the current view, WITHOUT committing."""
        op = OpInstance(name, dict(params or {}), mask, mask_invert, float(opacity))
        return self._run_one(self.result(), op)

    def preview_replace(self, index: int, name: str, params: Optional[dict] = None, *,
                        mask: Optional[str] = None, mask_invert: bool = False,
                        opacity: float = 1.0, base: Optional[np.ndarray] = None) -> np.ndarray:
        """Preview an EDIT of step ``index`` — the op re-applied to its own input.

        ``base`` overrides the input (used for a downscaled-proxy preview); otherwise the
        cached state before the step is used.
        """
        op = OpInstance(name, dict(params or {}), mask, mask_invert, float(opacity))
        src = self._cache[index] if base is None else base
        return self._run_one(src, op)

    def input_before(self, index: int) -> np.ndarray:
        """The cached image feeding step ``index`` (i.e. the result after ops[:index])."""
        return self._cache[max(0, min(index, len(self._cache) - 1))]

    def replace_op(self, index: int, params: Optional[dict] = None, *,
                   mask: Optional[str] = None, mask_invert: bool = False,
                   opacity: float = 1.0) -> np.ndarray:
        """Edit step ``index`` in place and recompute everything after it (kept)."""
        if not (0 <= index < len(self.ops)):
            return self.result()
        old = self.ops[index]
        self.ops[index] = OpInstance(old.name, dict(params or {}), mask, mask_invert, float(opacity))
        self._recompute_from(index)
        self.cursor = min(self.cursor, len(self.ops))
        return self.result()

    def delete_op(self, index: int) -> np.ndarray:
        """Remove step ``index``; keep and recompute the rest."""
        if not (0 <= index < len(self.ops)):
            return self.result()
        del self.ops[index]
        if self.cursor > index:
            self.cursor -= 1
        self._recompute_from(index)
        self.cursor = min(self.cursor, len(self.ops))
        return self.result()

    def undo(self) -> bool:
        if self.cursor <= 0:
            return False
        self.cursor -= 1
        return True

    def redo(self) -> bool:
        if self.cursor >= len(self.ops):
            return False
        self.cursor += 1
        return True

    def goto(self, position: int) -> np.ndarray:
        """Move the cursor to ``position`` (0 = base) without discarding any step."""
        self.cursor = max(0, min(int(position), len(self.ops)))
        return self.result()

    # Back-compat alias: revert_to now navigates (non-destructive) rather than truncates.
    revert_to = goto

    def reset(self) -> None:
        """Drop all steps; keep the base image and masks."""
        self.ops.clear()
        self.cursor = 0
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
