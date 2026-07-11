"""Per-sleeve virtual position book (``ledger/positions.json``).

One brokerage account backs all sleeves, so ``broker.get_position`` is
account-level: it can't tell whose shares they are. Trading deltas against
it lets one sleeve flatten another's holding whenever their universes
overlap (e.g. sector_rotation's SECTORS universe contains trend_following's
XLK). Each sleeve therefore reconciles against its *own* book here.

Positions are updated when an order is submitted; if the order later dies
(canceled/expired/rejected), the morning reconcile reverts the update.
"""

from __future__ import annotations

import json
from pathlib import Path

POSITIONS_FILE = Path("ledger") / "positions.json"


def load_positions(path: Path = POSITIONS_FILE) -> dict[str, dict[str, float]]:
    """Return {family: {symbol: signed shares}}; {} when missing/corrupt."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_positions(
    positions: dict[str, dict[str, float]], path: Path = POSITIONS_FILE
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(positions, indent=2), encoding="utf-8")


def get_position(
    family: str, symbol: str, path: Path = POSITIONS_FILE
) -> float:
    return load_positions(path).get(family, {}).get(symbol, 0.0)


def set_position(
    family: str, symbol: str, shares: float, path: Path = POSITIONS_FILE
) -> None:
    """Set one sleeve's position in a symbol (drops the entry when ~0)."""
    positions = load_positions(path)
    book = positions.setdefault(family, {})
    if abs(shares) < 1e-9:
        book.pop(symbol, None)
        if not book:
            positions.pop(family, None)
    else:
        book[symbol] = shares
    save_positions(positions, path)


def adjust_position(
    family: str, symbol: str, delta: float, path: Path = POSITIONS_FILE
) -> None:
    """Add ``delta`` shares to one sleeve's position in a symbol."""
    set_position(family, symbol, get_position(family, symbol, path) + delta, path)
