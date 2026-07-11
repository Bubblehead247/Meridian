"""Pending-order tracking and next-morning fill reconciliation.

Orders sent by the 16:30 ET session are queued by Alpaca and fill at the next
market open, so their price is unknown when placed. This module:

1. Persists those pending orders (``ledger/pending_orders.json``).
2. Keeps the permanent trade log (``ledger/trade_log.jsonl``) — one JSON line
   per fill, recorded at the price we *actually* filled, never an estimate.
3. Runs the morning reconcile: query Alpaca for each pending order, log the
   real fill price, and push a "Order Filled" ntfy confirmation.

The scheduler fires ``meridian reconcile`` at 9:45 ET on trading days.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from meridian.execution.notify import notify_fill

if TYPE_CHECKING:
    from meridian.execution.broker import AlpacaBroker, Fill

PENDING_ORDERS_FILE = Path("ledger") / "pending_orders.json"
TRADE_LOG_FILE = Path("ledger") / "trade_log.jsonl"

#: Alpaca order states that will never fill — drop these from the pending list.
_DEAD_STATUSES = ("canceled", "expired", "rejected", "done_for_day")


# ---------------------------------------------------------------------------
# Pending-order persistence
# ---------------------------------------------------------------------------

def load_pending_orders(path: Path = PENDING_ORDERS_FILE) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_pending_orders(orders: list[dict], path: Path = PENDING_ORDERS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(orders, indent=2), encoding="utf-8")


def add_pending_order(
    fill: "Fill", family: str, model: str, path: Path = PENDING_ORDERS_FILE
) -> None:
    """Remember an unfilled order so the morning reconcile can price it."""
    orders = load_pending_orders(path)
    orders.append({
        "order_id": fill.order_id,
        "symbol": fill.symbol,
        "qty": fill.qty,
        "family": family,
        "model": model,
        "submitted": date.today().isoformat(),
    })
    save_pending_orders(orders, path)


# ---------------------------------------------------------------------------
# Trade log — the permanent record, always at actual fill prices
# ---------------------------------------------------------------------------

def record_fill(
    fill: "Fill", family: str, model: str,
    *,
    submitted: str | None = None,
    path: Path = TRADE_LOG_FILE,
) -> dict:
    """Append one confirmed fill to the trade log; return the record written."""
    record = {
        "date": date.today().isoformat(),
        "submitted": submitted or date.today().isoformat(),
        "symbol": fill.symbol,
        "qty": fill.qty,
        "side": "buy" if fill.qty > 0 else "sell",
        "price": fill.price,
        "order_id": fill.order_id,
        "family": family,
        "model": model,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_trade_log(path: Path = TRADE_LOG_FILE) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


# ---------------------------------------------------------------------------
# Morning reconcile
# ---------------------------------------------------------------------------

def reconcile_pending_orders(
    broker: "AlpacaBroker",
    *,
    notify: bool = True,
    pending_path: Path = PENDING_ORDERS_FILE,
    log_path: Path = TRADE_LOG_FILE,
) -> list[dict]:
    """Price yesterday's pending orders against the broker's actual fills.

    For each pending order: if filled, append it to the trade log at the real
    fill price and (optionally) push an ntfy confirmation. Dead orders
    (canceled/expired/rejected) are dropped. Orders still open stay pending.

    Returns the list of trade-log records written this run.
    """
    from meridian.execution.broker import Fill

    pending = load_pending_orders(pending_path)
    if not pending:
        return []

    filled_records: list[dict] = []
    still_pending: list[dict] = []

    for order in pending:
        status, price = broker.order_status(order["order_id"])
        if status == "filled" and price > 0:
            fill = Fill(
                symbol=order["symbol"], qty=float(order["qty"]), price=price,
                order_id=order["order_id"], filled=True,
            )
            record = record_fill(
                fill, order.get("family", "?"), order.get("model", "?"),
                submitted=order.get("submitted"), path=log_path,
            )
            filled_records.append(record)
            if notify:
                notify_fill(record)
        elif status in _DEAD_STATUSES:
            pass  # will never fill — drop it
        else:
            still_pending.append(order)  # new/accepted/unknown — keep waiting

    save_pending_orders(still_pending, pending_path)
    return filled_records
