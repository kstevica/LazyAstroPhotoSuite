"""Per-master run history, persisted to disk.

Each processed master keeps a ``history/`` subfolder next to it holding one JSON index
(``<stem>.history.json``) plus one 16-bit TIFF per remembered run. The index records the
master's full path and, per item, its rendered-image filename, the settings that produced
it (a ``.lsrecipe``-style dict), a 1-5 star rating, a select/discard status, and a one-line
comment. Images live on disk (not RAM); they are loaded only when previewed.

The images are stored at 16 bits (not 8) so a run can be *continued* from — used as the
working base for further polishing — without the banding an 8-bit round-trip would leave.
TIFF (not PNG) because the PNG writer here can't encode 16-bit RGB. ``load_image_data``
reconstructs float64 ``[0, 1]`` from whatever bit depth is on disk, so it doubles as both the
preview loader and the raw-precision base for "continue from this".

Capacity is bounded (``MAX_ITEMS``); when full, the oldest UNrated + non-selected item is
evicted first so rated/kept work survives. The store is UI-agnostic — the GUI drives it.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .image_io import load_image, save_image

MAX_ITEMS = 20
STATUS_NONE = "none"
STATUS_SELECTED = "selected"
STATUS_DISCARDED = "discarded"


def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class HistoryStore:
    """Disk-backed history for a single master image."""

    def __init__(self, master_path: str):
        self.master_path = str(Path(master_path).resolve())
        self.stem = Path(self.master_path).stem
        self.dir = Path(self.master_path).parent / "history"
        self.index_path = self.dir / f"{self.stem}.history.json"
        self.items: list = []          # oldest first
        self._next_id = 1
        self.load()

    # --- persistence ---------------------------------------------------------

    def load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            self.items = [it for it in data.get("items", []) if isinstance(it, dict)]
            self._next_id = int(data.get("next_id", len(self.items) + 1))

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {"master": self.master_path, "next_id": self._next_id, "items": self.items}
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- mutations -----------------------------------------------------------

    def add(self, image: np.ndarray, params: dict, label: str, mode: str, steps: int) -> dict:
        """Write the rendered image to disk and append a new history item; returns it."""
        self.dir.mkdir(parents=True, exist_ok=True)
        iid = f"{self._next_id:04d}"
        self._next_id += 1
        fname = f"{self.stem}_{iid}.tif"
        # 16-bit TIFF so the run can be continued from without banding (see module docstring).
        save_image(str(self.dir / fname), np.asarray(image), bit_depth=16)
        item = {
            "id": iid, "file": fname, "label": label, "mode": mode, "steps": int(steps),
            "rating": 0, "status": STATUS_NONE, "comment": "",
            "params": params, "time": _now_str(),
        }
        self.items.append(item)
        self._evict()
        self.save()
        return item

    def _evict(self) -> None:
        while len(self.items) > MAX_ITEMS:
            victim = next((it for it in self.items
                           if int(it.get("rating", 0)) == 0
                           and it.get("status") != STATUS_SELECTED), self.items[0])
            self._delete_file(victim)
            self.items.remove(victim)

    def set_rating(self, item: dict, rating: int) -> None:
        item["rating"] = int(max(0, min(5, rating)))
        self.save()

    def set_comment(self, item: dict, comment: str) -> None:
        item["comment"] = str(comment)
        self.save()

    def set_status(self, item: dict, status: str) -> None:
        if status in (STATUS_NONE, STATUS_SELECTED, STATUS_DISCARDED):
            item["status"] = status
            self.save()

    def remove(self, item: dict) -> None:
        self._delete_file(item)
        if item in self.items:
            self.items.remove(item)
        self.save()

    def clear(self) -> None:
        for it in list(self.items):
            self._delete_file(it)
        self.items = []
        self.save()

    # --- helpers -------------------------------------------------------------

    def image_path(self, item: dict) -> str:
        return str(self.dir / item["file"])

    def load_image_data(self, item: dict) -> Optional[np.ndarray]:
        """Load the item's rendered image as float64 ``[0, 1]``.

        Full precision from the stored 16-bit TIFF — this is both the preview image and the
        raw base a "continue from this" run resumes processing on.
        """
        p = self.dir / item["file"]
        if not p.exists():
            return None
        return load_image(str(p)).data

    def _delete_file(self, item: dict) -> None:
        try:
            (self.dir / item.get("file", "")).unlink()
        except OSError:
            pass
