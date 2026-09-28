"""Trend 6 core through the live session, on the live allocation table (12.5% per sleeve)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from meridian.execution import live_runner
from meridian.execution.broker import SimulatedBroker


class _Long:
    cross_sectional = False
    signal_price_field = "adj_close"

    def signals(self, prices, bars=None):
        return pd.Series(1, index=prices.index)


class _Flat(_Long):
    def signals(self, prices, bars=None):
        return pd.Series(0, index=prices.index)


def _session(monkeypatch, tmp_path, model_cls, book=None):
    fields = []

    def fake_fetch(symbols, start="2023-01-01", field="close"):
        fields.append((tuple(symbols), field))
        return {s: pd.Series(100.0 if field == "close" else 90.0, index=pd.RangeIndex(300)) for s in symbols}

    monkeypatch.setattr(live_runner, "load_live_picks",
                        lambda: {"core_trend_spy": {"model": "sma6_monthly", "symbol": "SPY"}})
    monkeypatch.setattr(live_runner, "_fetch_prices", fake_fetch)
    monkeypatch.setattr(live_runner, "create_model", lambda f, m: model_cls())
    positions = tmp_path / "positions.json"
    positions.write_text(json.dumps(book or {}))
    broker = SimulatedBroker(cash=100_000.0, cost_bps=0.0)
    for sym, qty in (book or {}).get("core_trend_spy", {}).items():
        broker.positions[sym] = qty
    live_runner.run_paper_session(broker, account_equity=100_000.0, positions_path=positions,
                                  rebalance_schedule_path=tmp_path / "sched.json")
    return json.loads(positions.read_text())["core_trend_spy"], fields


def test_a_long_trend_sleeve_holds_its_etf_sized_on_the_real_close(monkeypatch, tmp_path):
    book, fields = _session(monkeypatch, tmp_path, _Long)
    assert book["SPY"] == pytest.approx(125.0)          # 12.5% of $100k at the $100 close
    assert book.get("SGOV", 0.0) == 0.0
    assert (("SPY",), "adj_close") in fields            # signal read adjusted prices


def test_a_flat_trend_sleeve_parks_its_eighth_in_sgov(monkeypatch, tmp_path):
    book, _ = _session(monkeypatch, tmp_path, _Flat)
    assert book.get("SPY", 0.0) == 0.0
    assert book["SGOV"] == pytest.approx(125.0)


def test_turning_flat_sells_the_etf_and_buys_sgov(monkeypatch, tmp_path):
    book, _ = _session(monkeypatch, tmp_path, _Flat, book={"core_trend_spy": {"SPY": 125.0}})
    assert book.get("SPY", 0.0) == 0.0
    assert book["SGOV"] == pytest.approx(125.0)
