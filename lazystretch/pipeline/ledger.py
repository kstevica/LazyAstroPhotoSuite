"""PROC ledger + pins — the Stretch Process tab's data model (LazyStretch.js:3136-3310).

Every value the pipeline computes flows through :meth:`Ledger.record`, which logs it and,
if the user has *pinned* that row, substitutes the pinned value and returns it — so
everything downstream re-derives from the pin. Pure plumbing: no image processing, no PI
processes. Pins are keyed ``"group|name"``, type-checked against the computed value, and
ignored (never crash) on a type mismatch. Pins live per-master (not in the shared recipe).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

MEASURED = "measured"      # a measured fact (not pinnable)
DECIDED = "decided"        # a decided setting (pinnable)
INFO = "info"              # context (not pinnable)


@dataclass
class LedgerEntry:
    group: str
    name: str
    kind: str
    computed: object        # the engine's own value
    value: object           # what the run actually used (== computed unless pinned)
    pinned: bool
    note: str

    @property
    def key(self) -> str:
        return f"{self.group}|{self.name}"


class Ledger:
    """Records pipeline decisions and applies user pin overrides."""

    def __init__(self, pins: Optional[Dict[str, object]] = None):
        self.pins: Dict[str, object] = dict(pins or {})
        self.entries: List[LedgerEntry] = []

    @staticmethod
    def key(group: str, name: str) -> str:
        return f"{group}|{name}"

    def record(self, group: str, name: str, value, *, kind: str = DECIDED, note: str = ""):
        """Log ``value`` under (group, name); return the pinned override if one applies."""
        key = self.key(group, name)
        out, pinned = value, False
        if kind == DECIDED and key in self.pins:
            coerced = self._coerce(self.pins[key], value)
            if coerced is not None:
                out, pinned = coerced, True
        self.entries.append(LedgerEntry(group, name, kind, value, out, pinned, note))
        return out

    @staticmethod
    def _coerce(pin, like):
        """Coerce a stored pin to the computed value's type; None if it can't (→ ignored)."""
        try:
            if isinstance(like, bool):
                if isinstance(pin, str):
                    return pin.strip().lower() in ("1", "true", "yes", "on")
                return bool(pin)
            if isinstance(like, int) and not isinstance(like, bool):
                return int(round(float(pin)))
            if isinstance(like, float):
                return float(pin)
            if isinstance(like, str):
                return str(pin)
        except (TypeError, ValueError):
            return None
        return pin
