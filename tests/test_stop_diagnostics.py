"""Tests for signals/stop_diagnostics.py — diagnostic-only, never changes P&L."""

from __future__ import annotations

import pandas as pd
import pytest

from meridian.signals import backtest
from meridian.signals.stop_diagnostics import flag_intrabar_stop_breaches


def _trades(direction, entry_time, exit_time, exit_price) -> pd.DataFrame:
    return pd.DataFrame([{
        "direction": direction, "entry_time": entry_time, "exit_time": exit_time,
        "bars_held": exit_time - entry_time, "entry_price": 100.0,
        "exit_price": exit_price, "return": 0.0,
    }])


def test_long_trade_breached_early_is_flagged():
    # Long trade entered at t=0, exit recorded at t=4 (exit_price=90).
    # Bar 2's low already touches 90 -> the close-driven exit was 2 bars late.
    idx = pd.RangeIndex(6)
    bars = pd.DataFrame({
        "high": [101, 101, 101, 101, 101, 101],
        "low":  [ 99,  95,  90,  91,  90,  85],
    }, index=idx)
    trades = _trades(direction=1, entry_time=0, exit_time=4, exit_price=90.0)
    out = flag_intrabar_stop_breaches(trades, bars)
    assert out["intrabar_stop_breach"].iloc[0] == True  # noqa: E712
    assert out["bars_late"].iloc[0] == 2  # bars 2,3 are strictly between entry(0) and exit(4)


def test_short_trade_breached_early_is_flagged():
    idx = pd.RangeIndex(6)
    bars = pd.DataFrame({
        "high": [101, 105, 110, 108, 109, 111],
        "low":  [ 95,  95,  95,  95,  95,  95],
    }, index=idx)
    trades = _trades(direction=-1, entry_time=0, exit_time=4, exit_price=110.0)
    out = flag_intrabar_stop_breaches(trades, bars)
    assert out["intrabar_stop_breach"].iloc[0] == True  # noqa: E712
    assert out["bars_late"].iloc[0] == 2


def test_no_breach_when_exit_price_only_reached_at_exit_bar():
    idx = pd.RangeIndex(4)
    bars = pd.DataFrame({"high": [101, 101, 101, 101], "low": [99, 99, 99, 90]}, index=idx)
    trades = _trades(direction=1, entry_time=0, exit_time=3, exit_price=90.0)
    out = flag_intrabar_stop_breaches(trades, bars)
    assert out["intrabar_stop_breach"].iloc[0] == False  # noqa: E712
    assert out["bars_late"].iloc[0] == 0


def test_empty_trades_returns_empty_with_new_columns():
    trades = pd.DataFrame(columns=["direction", "entry_time", "exit_time", "exit_price"])
    bars = pd.DataFrame({"high": [1.0], "low": [1.0]})
    out = flag_intrabar_stop_breaches(trades, bars)
    assert list(out.columns) == ["direction", "entry_time", "exit_time", "exit_price",
                                  "intrabar_stop_breach", "bars_late"]
    assert len(out) == 0


def test_bars_missing_high_low_no_ops_safely():
    trades = _trades(direction=1, entry_time=0, exit_time=2, exit_price=90.0)
    bars = pd.DataFrame({"close": [100.0, 95.0, 90.0]}, index=pd.RangeIndex(3))
    out = flag_intrabar_stop_breaches(trades, bars)
    assert out["intrabar_stop_breach"].iloc[0] == False  # noqa: E712


# --- integration: opt-in via backtest(), never changes returns -------------

def test_backtest_stop_diagnostics_is_additive_only():
    prices = pd.Series([100.0, 100.0, 100.0, 90.0, 90.0])
    bars = pd.DataFrame({
        "open": [100.0, 100.0, 100.0, 90.0, 90.0],
        "high": [101.0, 101.0, 101.0, 91.0, 91.0],
        "low":  [ 99.0,  95.0,  90.0, 89.0, 89.0],  # bar 1 already touches 90
    }, index=prices.index)
    positions = pd.Series([1, 1, 1, 0, 0])

    without = backtest(prices, positions, cost_bps=0.0, bars=bars, stop_diagnostics=False)
    with_diag = backtest(prices, positions, cost_bps=0.0, bars=bars, stop_diagnostics=True)

    assert "intrabar_stop_breach" not in without.trades.columns
    assert "intrabar_stop_breach" in with_diag.trades.columns
    # Diagnostics must never change the executed returns/equity.
    pd.testing.assert_series_equal(without.returns, with_diag.returns)
    pd.testing.assert_series_equal(without.equity, with_diag.equity)


def test_backtest_stop_diagnostics_off_by_default():
    prices = pd.Series([100.0, 100.0, 90.0])
    bars = pd.DataFrame(
        {"open": [100.0, 100.0, 90.0], "high": [101, 101, 91], "low": [99, 90, 89]},
        index=prices.index,
    )
    res = backtest(prices, pd.Series([1, 1, 0]), cost_bps=0.0, bars=bars)
    assert "intrabar_stop_breach" not in res.trades.columns
