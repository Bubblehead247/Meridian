"""Tracks when each (family, symbol) position was last resized toward target.

Entries and exits (opening or closing a position) are signal-driven decisions
and always happen the day the signal fires — see ``REBALANCE_BAND_PCT`` in
``live_runner.py``. Resizing an *already-open, still-held* position toward a
drifted target is different: left unthrottled it fires daily, adding
commission/spread drag for no strategic reason. This throttles that resize to
at most once per calendar month per (family, symbol), independent of the
existing 5% rebalance band (which still applies within the month it fires).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

REBALANCE_SCHEDULE_FILE = Path("ledger") / "last_rebalance.json"


def _key(family: str, symbol: str) -> str:
    return f"{family}/{symbol}"


def load_last_rebalance(path: Path = REBALANCE_SCHEDULE_FILE) -> dict:
    """Return {"family/symbol": "YYYY-MM-DD" or {"date", "weight"}}; {} when missing/corrupt."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_last_rebalance(data: dict[str, str], path: Path = REBALANCE_SCHEDULE_FILE) -> None:
    from quantcore.statefile import write_json_atomic

    write_json_atomic(path, data)


def _entry(data: dict, family: str, symbol: str) -> tuple[str | None, float | None]:
    """(date, weight) of the last stamp; a legacy entry is a bare date string."""
    raw = data.get(_key(family, symbol))
    if isinstance(raw, dict):
        return raw.get("date"), raw.get("weight")
    return raw, None


def is_rebalance_due(
    family: str, symbol: str, today: date, path: Path = REBALANCE_SCHEDULE_FILE,
    *, weight: float | None = None,
) -> bool:
    """True if this position hasn't been resized this calendar month, or if its
    sleeve's allocation changed since it was (an allocation change is a decision,
    not drift — without this, switching allocation mid-month would leave the old
    sizes in place until the next month)."""
    last, last_weight = _entry(load_last_rebalance(path), family, symbol)
    if not last:
        return True
    if weight is not None and (last_weight is None or abs(last_weight - weight) > 1e-9):
        return True
    try:
        last_date = date.fromisoformat(last)
    except ValueError:
        return True
    return (last_date.year, last_date.month) != (today.year, today.month)


def record_rebalance(
    family: str, symbol: str, today: date, path: Path = REBALANCE_SCHEDULE_FILE,
    *, weight: float | None = None,
) -> None:
    """Stamp this position as considered-for-resize this month (at this weight).

    Called once the monthly gate opens, regardless of whether the existing
    rebalance band ends up suppressing the actual order — the point is "this
    position gets reconsidered once a month," not "once a month that clears
    the band."
    """
    data = load_last_rebalance(path)
    data[_key(family, symbol)] = (
        today.isoformat() if weight is None else {"date": today.isoformat(), "weight": weight})
    save_last_rebalance(data, path)
