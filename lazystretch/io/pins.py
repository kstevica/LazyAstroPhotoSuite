"""Per-master PROC pin persistence.

Pins are ``"group|name" -> value`` overrides the user sets on the Stretch Process tab.
Like the .js, they are stored per-master (next to the history, keyed by the master stem),
NOT in the shareable ``.lsrecipe`` — they are image-specific tuning, not a portable look.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def _pins_path(master_path: str) -> Path:
    p = Path(master_path).resolve()
    return p.parent / "history" / f"{p.stem}.pins.json"


def load_pins(master_path: str) -> Dict[str, object]:
    """Load the pins for a master (empty dict if none / unreadable)."""
    try:
        path = _pins_path(master_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items()}
    except (OSError, ValueError):
        pass
    return {}


def save_pins(master_path: str, pins: Dict[str, object]) -> None:
    """Write the pins for a master (creates the history/ dir; removes the file if empty)."""
    path = _pins_path(master_path)
    try:
        if not pins:
            if path.exists():
                path.unlink()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pins, indent=2), encoding="utf-8")
    except OSError:
        pass
