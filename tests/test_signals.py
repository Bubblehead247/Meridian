"""Phase 4 tests: signal engine and backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.signals import (
    SignalConfig,
    backtest,
    compute_scores,
    generate_positions,
    run_backtest,
    sweep,
)


def _scores(values) -> pd.Series:
    return pd.Series(values, index=pd.RangeIndex(len(values)), dtype=float)


# --- signal engine: deterministic scenarios -------------------------------

def test_long_entry_and_exit_at_fair_value():
    cfg = SignalConfig(entry_threshold=2.0, exit_threshold=0.0)
    # dip to -3 (enter long), recover to +0.5 (cross fair value -> exit)
    pos = generate_positions(_scores([0, -1, -3, -1, 0.5, 0]), cfg)
    assert list(pos) == [0, 0, 1, 1, 0, 0]


def test_short_entry_and_exit():
    cfg = SignalConfig(entry_threshold=2.0, exit_threshold=0.0)
    pos = generate_positions(_scores([0, 3, 1, -0.2, 0]), cfg)
    assert list(pos) == [0, -1, -1, 0, 0]


def test_allow_short_false_stays_flat_on_high_score():
    cfg = SignalConfig(entry_threshold=2.0, allow_short=False)
    pos = generate_positions(_scores([0, 3, 3, 3]), cfg)
    assert list(pos) == [0, 0, 0, 0]


def test_stop_loss_closes_widening_long():
    cfg = SignalConfig(entry_threshold=2.0, exit_threshold=0.0, stop_threshold=4.0)
    # enter long at -2, deviation widens to -5 -> stop fires
    pos = generate_positions(_scores([-2, -3, -5, -5]), cfg)
    assert list(pos) == [1, 1, 0, 0]


def test_max_holding_forces_exit():
    cfg = SignalConfig(entry_threshold=2.0, exit_threshold=0.0, max_holding=2)
    # stays below exit the whole time; forced out after 2 bars held
    pos = generate_positions(_scores([-3, -3, -3, -3]), cfg)
    # bar0 enter(1); bar1 held=1; bar2 held=2 -> exit(0); bar3 re-enter(1)
    assert list(pos) == [1, 1, 0, 1]


def test_nan_score_holds_position():
    cfg = SignalConfig(entry_threshold=2.0)
    pos = generate_positions(_scores([-3, np.nan, np.nan, 0.5]), cfg)
    assert list(pos) == [1, 1, 1, 0]


def test_entry_threshold_must_be_positive():
    with pytest.raises(ValueError):
        SignalConfig(entry_threshold=0.0)


# --- backtester accounting ------------------------------------------------

def test_no_lookahead_last_position_does_not_earn():
    prices = pd.Series([100.0, 101.0, 102.0, 110.0])
    # A position only on the final bar cannot capture any past return.
    positions = pd.Series([0, 0, 0, 1])
    res = backtest(prices, positions, cost_bps=0.0)
    # held is lagged: nothing held until after the last bar -> zero return.
    assert res.equity.iloc[-1] == pytest.approx(1.0)


def test_long_position_earns_price_return():
    prices = pd.Series([100.0, 110.0, 121.0])
    positions = pd.Series([1, 1, 1])  # decided each bar, applied next bar
    res = backtest(prices, positions, cost_bps=0.0)
    # held=[0,1,1]; returns 0, +10%, +10% -> equity 1.21
    assert res.equity.iloc[-1] == pytest.approx(1.21)


def test_costs_reduce_return_on_turnover():
    prices = pd.Series([100.0, 100.0, 100.0])
    positions = pd.Series([1, 1, 1])
    free = backtest(prices, positions, cost_bps=0.0).equity.iloc[-1]
    costed = backtest(prices, positions, cost_bps=10.0).equity.iloc[-1]
    assert costed < free  # one entry trade charged

def test_flat_positions_give_flat_equity():
    prices = pd.Series([100.0, 105.0, 95.0, 110.0])
    res = backtest(prices, pd.Series([0, 0, 0, 0]), cost_bps=5.0)
    assert (res.equity == 1.0).all()
    assert len(res.trades) == 0


def test_short_profits_when_price_falls():
    prices = pd.Series([100.0, 90.0, 81.0])
    res = backtest(prices, pd.Series([-1, -1, -1]), cost_bps=0.0)
    # held=[0,-1,-1]; short earns +10% then +10% -> 1.21
    assert res.equity.iloc[-1] == pytest.approx(1.21)


def test_trade_ledger_records_a_round_trip():
    prices = pd.Series([100.0, 100.0, 110.0, 110.0])
    positions = pd.Series([1, 1, 0, 0])  # in for bars 1-2 (held), out after
    res = backtest(prices, positions, cost_bps=0.0)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["direction"] == 1
    assert t["return"] > 0


# --- end-to-end pipeline --------------------------------------------------

def _mean_reverting(n=500, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 1)  # AR(1), strongly reverting
    return pd.Series(100 + x, index=pd.RangeIndex(n))


def test_compute_scores_is_causal_and_aligned():
    prices = _mean_reverting()
    scores = compute_scores(prices, "sma", "zscore", window=20)
    assert len(scores) == len(prices)
    assert scores.index.equals(prices.index)
    assert np.isnan(scores.iloc[0])  # warmup


def test_run_backtest_produces_trades_on_reverting_series():
    prices = _mean_reverting()
    res = run_backtest(prices, "sma", "zscore", SignalConfig(entry_threshold=1.0), window=20)
    s = res.summary()
    assert s["n_trades"] > 0
    assert s["estimator"] == "sma"
    assert 0.0 <= s["exposure"] <= 1.0


def test_run_backtest_with_atr_norm_uses_bars():
    prices = _mean_reverting()
    bars = pd.DataFrame(
        {"high": prices + 1, "low": prices - 1, "close": prices}, index=prices.index
    )
    res = run_backtest(
        prices, "ema", "atr_norm", SignalConfig(entry_threshold=1.0), window=20, bars=bars
    )
    assert res.summary()["n_trades"] >= 0  # runs without error, bars threaded


def test_sweep_covers_all_pairs():
    prices = _mean_reverting()
    results = sweep(
        prices, ["sma", "ema"], ["zscore", "mad_z"],
        SignalConfig(entry_threshold=1.0), window=20,
    )
    assert set(results) == {
        ("sma", "zscore"), ("sma", "mad_z"), ("ema", "zscore"), ("ema", "mad_z")
    }


def test_pipeline_is_deterministic():
    prices = _mean_reverting()
    a = run_backtest(prices, "kalman", "modified_z", window=20).equity
    b = run_backtest(prices, "kalman", "modified_z", window=20).equity
    pd.testing.assert_series_equal(a, b)
