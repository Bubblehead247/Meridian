"""Tests for cost-stress and capacity/participation analysis (P2-A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.analytics.capacity import (
    average_daily_volume,
    capacity_stress_sweep,
    cost_stress_conclusion_stable,
    cost_stress_sweep,
    cost_stress_sweep_cross_sectional,
    implied_shares_traded,
    max_capacity,
    participation_rate,
)


def _prices(n=300, seed=0):
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=idx)


def _random_positions(prices, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice([-1, 0, 1], size=len(prices)), index=prices.index)


def _bars(prices, volume=1_000_000.0):
    return pd.DataFrame(
        {"open": prices.shift(1).fillna(prices.iloc[0]), "close": prices,
         "volume": volume},
        index=prices.index,
    )


# --- cost-stress sweep -------------------------------------------------------

def test_cost_stress_sweep_default_grid_matches_brief():
    prices = _prices()
    positions = _random_positions(prices)
    df = cost_stress_sweep(prices, positions)
    assert list(df["cost_bps"]) == [0.0, 5.0, 10.0, 25.0, 50.0, 100.0]


def test_cost_stress_sweep_is_non_increasing_in_total_return():
    prices = _prices()
    positions = _random_positions(prices)
    df = cost_stress_sweep(prices, positions)
    rets = df["total_return"].to_numpy()
    assert all(a >= b - 1e-12 for a, b in zip(rets, rets[1:]))


def test_cost_stress_sweep_zero_cost_matches_direct_backtest():
    from meridian.signals.backtest import backtest

    prices = _prices()
    positions = _random_positions(prices)
    df = cost_stress_sweep(prices, positions, bps_grid=(0.0,))
    direct = backtest(prices, positions, cost_bps=0.0).summary()
    assert df.iloc[0]["sharpe"] == pytest.approx(direct["sharpe"], nan_ok=True)
    assert df.iloc[0]["total_return"] == pytest.approx(direct["total_return"])


def test_cost_stress_sweep_cross_sectional_default_grid():
    idx = pd.date_range("2018-01-01", periods=200, freq="B")
    rng = np.random.default_rng(2)
    symbols = ["A", "B", "C"]
    prices = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx)))) for s in symbols},
        index=idx,
    )
    signals = pd.DataFrame(
        rng.choice([-1, 0, 1], size=(len(idx), len(symbols))), index=idx, columns=symbols
    )
    df = cost_stress_sweep_cross_sectional(signals, prices)
    assert list(df["cost_bps"]) == [0.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    rets = df["total_return"].to_numpy()
    assert all(a >= b - 1e-12 for a, b in zip(rets, rets[1:]))


def test_cost_stress_conclusion_stable_true_when_all_same_sign():
    df = pd.DataFrame({"sharpe": [1.0, 0.8, 0.5, 0.2]})
    assert cost_stress_conclusion_stable(df) is True


def test_cost_stress_conclusion_stable_false_when_sign_flips():
    df = pd.DataFrame({"sharpe": [0.5, 0.1, -0.2, -0.5]})
    assert cost_stress_conclusion_stable(df) is False


def test_cost_stress_conclusion_stable_false_when_all_nan():
    df = pd.DataFrame({"sharpe": [float("nan"), float("nan")]})
    assert cost_stress_conclusion_stable(df) is False


# --- capacity / participation ------------------------------------------------

def test_average_daily_volume_is_rolling_mean():
    idx = pd.date_range("2018-01-01", periods=5, freq="B")
    bars = pd.DataFrame({"volume": [10, 20, 30, 40, 50]}, index=idx)
    adv = average_daily_volume(bars, window=2)
    assert adv.iloc[1] == pytest.approx(15.0)
    assert adv.iloc[4] == pytest.approx(45.0)


def test_implied_shares_traded_basic():
    idx = pd.date_range("2018-01-01", periods=3, freq="B")
    prices = pd.Series([100.0, 100.0, 100.0], index=idx)
    positions = pd.Series([0, 1, 1], index=idx)  # enters at bar 1, holds at bar 2
    shares = implied_shares_traded(1_000_000.0, prices, positions)
    assert shares.iloc[0] == pytest.approx(0.0)      # flat -> flat, no trade
    assert shares.iloc[1] == pytest.approx(10_000.0)  # enters full $1M position at $100/share
    assert shares.iloc[2] == pytest.approx(0.0)       # holds -> no trade


def test_participation_rate_basic():
    idx = pd.date_range("2018-01-01", periods=3, freq="B")
    prices = pd.Series([100.0, 100.0, 100.0], index=idx)
    positions = pd.Series([0, 1, 1], index=idx)
    bars = pd.DataFrame({"volume": [100_000, 100_000, 100_000]}, index=idx)
    rate = participation_rate(1_000_000.0, prices, positions, bars, adv_window=1)
    # $1M / $100 = 10,000 shares traded on bar 1; ADV(window=1) on bar 1 = 100,000
    assert rate.iloc[1] == pytest.approx(0.10)


def test_max_capacity_is_bound_by_the_worst_trading_day():
    idx = pd.date_range("2018-01-01", periods=4, freq="B")
    prices = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
    # Two trading days: bar 1 enters (turnover=1), bar 3 flips to short (turnover=2, the
    # bigger trade) — capacity must be constrained by whichever day is tightest, not the
    # average.
    positions = pd.Series([0, 1, 1, -1], index=idx)
    bars = pd.DataFrame({"volume": [0, 100_000, 0, 50_000]}, index=idx)
    cap = max_capacity(prices, positions, bars, participation_limit=0.10, adv_window=1)
    # Bar 1: 10% of 100,000 shares = 10,000 shares tradeable -> capital = 10,000*$100 = $1,000,000
    # Bar 3: turnover=2 (short flip), 10% of 50,000 = 5,000 shares of *aggregate* turnover
    #   tradeable -> capital*2/100 <= 5,000 -> capital <= $250,000 (the binding constraint)
    assert cap == pytest.approx(250_000.0)


def test_max_capacity_infinite_when_never_trades():
    idx = pd.date_range("2018-01-01", periods=4, freq="B")
    prices = pd.Series(100.0, index=idx)
    positions = pd.Series(0, index=idx)
    bars = pd.DataFrame({"volume": [100_000] * 4}, index=idx)
    assert max_capacity(prices, positions, bars) == float("inf")


def test_max_capacity_zero_when_trading_day_has_no_volume():
    idx = pd.date_range("2018-01-01", periods=2, freq="B")
    prices = pd.Series([100.0, 100.0], index=idx)
    positions = pd.Series([0, 1], index=idx)
    bars = pd.DataFrame({"volume": [0, 0]}, index=idx)
    assert max_capacity(prices, positions, bars, adv_window=1) == 0.0


def test_capacity_stress_sweep_participation_rises_with_capital():
    idx = pd.date_range("2018-01-01", periods=3, freq="B")
    prices = pd.Series([100.0, 100.0, 100.0], index=idx)
    positions = pd.Series([0, 1, 1], index=idx)
    bars = pd.DataFrame({"volume": [100_000, 100_000, 100_000]}, index=idx)
    df = capacity_stress_sweep(
        prices, positions, bars, capital_levels=(100_000.0, 1_000_000.0, 20_000_000.0),
        adv_window=1,
    )
    assert df["worst_participation"].is_monotonic_increasing
    assert df.iloc[-1]["worst_participation"] > 1.0  # $20M order vs 100k-share ADV day: impossible
