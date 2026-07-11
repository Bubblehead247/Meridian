"""Tests for meridian.execution.reconcile.

Pending-order persistence, the trade log (records at actual fill price), and
the morning reconcile against a fake broker. All file paths go to tmp_path;
notifications are disabled or patched — no network, no repo pollution.
"""

from __future__ import annotations

import json
from datetime import date

from meridian.execution.broker import Fill
from meridian.execution.reconcile import (
    add_pending_order,
    load_pending_orders,
    load_trade_log,
    reconcile_pending_orders,
    record_fill,
)


class _FakeBroker:
    """Broker stub: order_id → (status, filled_avg_price)."""

    def __init__(self, statuses: dict[str, tuple[str, float]]):
        self.statuses = statuses

    def order_status(self, order_id: str) -> tuple[str, float]:
        return self.statuses.get(order_id, ("unknown", 0.0))


# --- pending-order persistence ---------------------------------------------

def test_add_and_load_pending_orders(tmp_path):
    path = tmp_path / "pending.json"
    fill = Fill(symbol="SPY", qty=10.0, price=0.0, order_id="abc-1", filled=False)
    add_pending_order(fill, "mean_reversion", "zscore_21", path=path)

    orders = load_pending_orders(path)
    assert len(orders) == 1
    assert orders[0]["order_id"] == "abc-1"
    assert orders[0]["symbol"] == "SPY"
    assert orders[0]["qty"] == 10.0
    assert orders[0]["family"] == "mean_reversion"
    assert orders[0]["submitted"] == date.today().isoformat()


def test_load_pending_orders_missing_file(tmp_path):
    assert load_pending_orders(tmp_path / "nope.json") == []


def test_load_pending_orders_corrupt_file(tmp_path):
    path = tmp_path / "pending.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_pending_orders(path) == []


# --- trade log ---------------------------------------------------------------

def test_record_fill_appends_jsonl(tmp_path):
    path = tmp_path / "trades.jsonl"
    fill = Fill(symbol="QQQ", qty=-5.0, price=372.50, order_id="xyz", filled=True)
    record_fill(fill, "momentum", "dual_momentum", path=path)
    fill2 = Fill(symbol="SPY", qty=3.0, price=451.10, order_id="xyz2", filled=True)
    record_fill(fill2, "mean_reversion", "zscore_21", path=path)

    records = load_trade_log(path)
    assert len(records) == 2
    assert records[0]["symbol"] == "QQQ"
    assert records[0]["side"] == "sell"
    assert records[0]["price"] == 372.50
    assert records[1]["side"] == "buy"
    assert records[1]["price"] == 451.10


def test_load_trade_log_missing_file(tmp_path):
    assert load_trade_log(tmp_path / "nope.jsonl") == []


# --- reconcile ----------------------------------------------------------------

def _pending(tmp_path, orders):
    path = tmp_path / "pending.json"
    path.write_text(json.dumps(orders), encoding="utf-8")
    return path


def test_reconcile_records_filled_orders(tmp_path):
    pending_path = _pending(tmp_path, [
        {"order_id": "a", "symbol": "SPY", "qty": 10.0,
         "family": "mean_reversion", "model": "zscore_21", "submitted": "2026-07-09"},
    ])
    log_path = tmp_path / "trades.jsonl"
    broker = _FakeBroker({"a": ("filled", 450.30)})

    records = reconcile_pending_orders(
        broker, notify=False, pending_path=pending_path, log_path=log_path)

    assert len(records) == 1
    assert records[0]["price"] == 450.30
    assert records[0]["submitted"] == "2026-07-09"
    assert load_trade_log(log_path)[0]["price"] == 450.30
    assert load_pending_orders(pending_path) == []  # no longer pending


def test_reconcile_drops_dead_orders_and_reverts_position_book(tmp_path):
    pending_path = _pending(tmp_path, [
        {"order_id": "dead", "symbol": "SPY", "qty": 5.0,
         "family": "momentum", "model": "dm", "submitted": "2026-07-09"},
    ])
    # The submit-time update credited momentum with the 5 shares.
    positions_path = tmp_path / "positions.json"
    positions_path.write_text(json.dumps({"momentum": {"SPY": 5.0}}))
    broker = _FakeBroker({"dead": ("canceled", 0.0)})

    records = reconcile_pending_orders(
        broker, notify=False, pending_path=pending_path,
        log_path=tmp_path / "trades.jsonl", positions_path=positions_path)

    assert records == []
    assert load_pending_orders(pending_path) == []
    # Position book reverted, so the sleeve retries next session.
    assert json.loads(positions_path.read_text()) == {}


def test_reconcile_keeps_unfilled_orders_pending(tmp_path):
    pending_path = _pending(tmp_path, [
        {"order_id": "open", "symbol": "SPY", "qty": 5.0,
         "family": "momentum", "model": "dm", "submitted": "2026-07-09"},
    ])
    broker = _FakeBroker({"open": ("accepted", 0.0)})

    records = reconcile_pending_orders(
        broker, notify=False,
        pending_path=pending_path, log_path=tmp_path / "trades.jsonl")

    assert records == []
    still = load_pending_orders(pending_path)
    assert len(still) == 1 and still[0]["order_id"] == "open"


def test_reconcile_mixed_batch(tmp_path):
    pending_path = _pending(tmp_path, [
        {"order_id": "a", "symbol": "SPY", "qty": 10.0,
         "family": "mr", "model": "z", "submitted": "2026-07-09"},
        {"order_id": "b", "symbol": "QQQ", "qty": -4.0,
         "family": "mo", "model": "dm", "submitted": "2026-07-09"},
        {"order_id": "c", "symbol": "IWM", "qty": 2.0,
         "family": "tf", "model": "ma", "submitted": "2026-07-09"},
    ])
    log_path = tmp_path / "trades.jsonl"
    broker = _FakeBroker({
        "a": ("filled", 450.30),
        "b": ("expired", 0.0),
        "c": ("new", 0.0),
    })

    records = reconcile_pending_orders(
        broker, notify=False, pending_path=pending_path, log_path=log_path,
        positions_path=tmp_path / "positions.json")

    assert [r["symbol"] for r in records] == ["SPY"]
    assert [o["order_id"] for o in load_pending_orders(pending_path)] == ["c"]


def test_reconcile_empty_pending_is_noop(tmp_path):
    broker = _FakeBroker({})
    records = reconcile_pending_orders(
        broker, notify=False,
        pending_path=tmp_path / "pending.json", log_path=tmp_path / "trades.jsonl")
    assert records == []
