"""Tests for realized slippage tracking (analytics/slippage.py)."""

from __future__ import annotations

import pytest

from meridian.analytics.slippage import (
    fill_slippage_bps,
    slippage_by_fill,
    slippage_summary,
)


def test_buy_paying_more_than_decision_price_is_positive_cost():
    assert fill_slippage_bps("buy", decision_price=100.0, fill_price=101.0) == pytest.approx(100.0)


def test_buy_paying_less_than_decision_price_is_negative_cost():
    assert fill_slippage_bps("buy", decision_price=100.0, fill_price=99.0) == pytest.approx(-100.0)


def test_sell_receiving_less_than_decision_price_is_positive_cost():
    assert fill_slippage_bps("sell", decision_price=100.0, fill_price=99.0) == pytest.approx(100.0)


def test_sell_receiving_more_than_decision_price_is_negative_cost():
    assert fill_slippage_bps("sell", decision_price=100.0, fill_price=101.0) == pytest.approx(-100.0)


def _lookup(prices: dict) -> callable:
    return lambda symbol, submitted: prices.get((symbol, submitted))


def test_slippage_by_fill_skips_records_with_no_resolvable_price():
    records = [
        {"date": "2026-01-02", "submitted": "2026-01-01", "symbol": "AAPL",
         "side": "buy", "price": 101.0, "qty": 1.0, "family": "momentum"},
        {"date": "2026-01-02", "submitted": "2026-01-01", "symbol": "UNKNOWN",
         "side": "buy", "price": 5.0, "qty": 1.0, "family": "momentum"},
    ]
    lookup = _lookup({("AAPL", "2026-01-01"): 100.0})
    rows = slippage_by_fill(records, lookup)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["cost_bps"] == pytest.approx(100.0)
    assert rows[0]["notional"] == pytest.approx(101.0)


def test_slippage_by_fill_skips_incomplete_records():
    records = [{"symbol": "AAPL", "side": "buy", "price": 101.0}]  # no submitted/qty
    assert slippage_by_fill(records, _lookup({})) == []


def test_slippage_summary_empty():
    summary = slippage_summary([])
    assert summary == {"n": 0, "weighted_bps": 0.0, "total_cost_dollars": 0.0, "by_family": {}}


def test_slippage_summary_weights_by_notional():
    # Small cheap fill costs a lot in bps; large fill costs a little — the
    # notional-weighted average should be dominated by the large fill.
    rows = slippage_by_fill(
        [
            {"date": "d", "submitted": "s", "symbol": "SMALL", "side": "buy",
             "price": 10.0, "qty": 1.0, "family": "mean_reversion"},
            {"date": "d", "submitted": "s", "symbol": "BIG", "side": "buy",
             "price": 100.0, "qty": 100.0, "family": "momentum"},
        ],
        _lookup({("SMALL", "s"): 9.0, ("BIG", "s"): 99.9}),  # SMALL: +1111bps, BIG: +10bps
    )
    summary = slippage_summary(rows)
    assert summary["n"] == 2
    # dominated by BIG's ~10bps, nowhere near SMALL's ~1111bps
    assert summary["weighted_bps"] < 50.0
    assert summary["total_cost_dollars"] > 0  # both fills cost money here


def test_slippage_summary_by_family_breakdown():
    rows = slippage_by_fill(
        [
            {"date": "d", "submitted": "s", "symbol": "A", "side": "buy",
             "price": 101.0, "qty": 1.0, "family": "momentum"},
            {"date": "d", "submitted": "s", "symbol": "B", "side": "sell",
             "price": 99.0, "qty": 1.0, "family": "sector_rotation"},
        ],
        _lookup({("A", "s"): 100.0, ("B", "s"): 100.0}),
    )
    summary = slippage_summary(rows)
    assert set(summary["by_family"]) == {"momentum", "sector_rotation"}
    assert summary["by_family"]["momentum"]["weighted_bps"] == pytest.approx(100.0)
    assert summary["by_family"]["sector_rotation"]["weighted_bps"] == pytest.approx(100.0)
