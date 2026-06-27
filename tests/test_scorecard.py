"""Tests for the per-strategy scorecard (wraps analytics.metrics + extra fields)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.scoring import scorecard, scorecard_from_backtest
from meridian.scoring.scorecard import (
    _avg_holding_period,
    _slippage_sensitivity,
    _turnover,
    _ulcer_index,
)


def _dates(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


# --- individual fields ----------------------------------------------------

def test_ulcer_zero_when_no_drawdown_positive_otherwise():
    up = pd.Series(np.full(50, 0.001))            # only gains -> never underwater
    assert _ulcer_index(up) < 1e-9
    dd = pd.Series([0.05, -0.10, -0.05, 0.02])    # a real drop
    assert _ulcer_index(dd) > 0


def test_avg_holding_period_from_trades():
    trades = pd.DataFrame({"bars_held": [2, 4, 6], "return": [0.1, -0.05, 0.2]})
    assert _avg_holding_period(trades) == 4.0
    assert np.isnan(_avg_holding_period(pd.DataFrame()))


def test_turnover_exact_from_positions_and_fallback_from_trades():
    idx = _dates(5)
    positions = pd.Series([0, 1, 1, 0, -1], index=idx)   # Σ|Δ| = 0+1+0+1+1 = 3
    years = 5 / 252
    assert abs(_turnover(positions, None, n=5, periods_per_year=252) - 3 / years) < 1e-9
    # fallback: 2 units per round-trip trade
    trades = pd.DataFrame({"bars_held": [1, 1], "return": [0.0, 0.0]})
    assert abs(_turnover(None, trades, n=5, periods_per_year=252) - 4 / years) < 1e-9


def test_slippage_sensitivity_monotonic_and_empty_without_positions():
    idx = _dates(100)
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 100), index=idx)
    positions = pd.Series(rng.integers(-1, 2, 100), index=idx).astype(float)
    sens = _slippage_sensitivity(returns, positions, bps_grid=(1, 5, 10, 20), periods_per_year=252)
    assert list(sens) == [1, 5, 10, 20]
    vals = list(sens.values())
    assert all(a >= b for a, b in zip(vals, vals[1:], strict=False))  # more slippage never helps
    assert _slippage_sensitivity(returns, None, bps_grid=(1, 5), periods_per_year=252) == {}


# --- full scorecard -------------------------------------------------------

def test_scorecard_has_base_plus_extra_fields():
    idx = _dates(80)
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0005, 0.01, 80), index=idx)
    positions = pd.Series(rng.integers(-1, 2, 80), index=idx).astype(float)
    trades = pd.DataFrame({"bars_held": [3, 5], "return": [0.02, -0.01]})
    card = scorecard(returns, trades=trades, positions=positions)
    # base block (from analytics.metrics)
    for k in ("cagr", "sharpe", "sortino", "calmar", "max_drawdown", "profit_factor"):
        assert k in card
    # added block
    for k in ("ulcer_index", "turnover", "avg_holding_period",
              "slippage_sensitivity", "regime_conditional", "sleeve_correlation"):
        assert k in card
    assert card["avg_holding_period"] == 4.0
    assert isinstance(card["slippage_sensitivity"], dict)


def test_regime_conditional_splits_by_dimension_and_skips_unknown():
    idx = _dates(10)
    returns = pd.Series(np.linspace(0.01, 0.02, 10), index=idx)
    frame = pd.DataFrame(
        {
            "trend": ["unknown", "unknown"] + ["bull"] * 4 + ["bear"] * 4,
            "volatility": ["low"] * 10,
            "breadth": ["neutral"] * 10,
        },
        index=idx,
    )
    rc = scorecard(returns, regime_frame=frame)["regime_conditional"]
    assert set(rc) == {"trend", "volatility", "breadth"}
    assert set(rc["trend"]) == {"bull", "bear"}           # unknown bucket skipped
    assert rc["trend"]["bull"]["n_periods"] == 4
    assert rc["volatility"]["low"]["n_periods"] == 10


def test_sleeve_correlation_detects_relationships():
    idx = _dates(60)
    rng = np.random.default_rng(2)
    base = pd.Series(rng.normal(0, 0.01, 60), index=idx)
    noise = pd.Series(rng.normal(0, 0.01, 60), index=idx)
    card = scorecard(
        base, sleeve_returns={"twin": base.copy(), "inverse": -base, "noise": noise}
    )
    corr = card["sleeve_correlation"]
    assert abs(corr["twin"] - 1.0) < 1e-9
    assert abs(corr["inverse"] + 1.0) < 1e-9
    assert abs(corr["noise"]) < 0.5


def test_scorecard_from_backtest_end_to_end():
    from meridian.signals import run_backtest

    n = 400
    idx = _dates(n)
    # oscillating, reverting series -> generates trades
    prices = pd.Series(100 + 5 * np.sin(np.linspace(0, 40, n)), index=idx)
    res = run_backtest(prices, "sma", "zscore", window=20)
    frame = pd.DataFrame(
        {"trend": "neutral", "volatility": "low", "breadth": "neutral"}, index=idx
    )
    card = scorecard_from_backtest(res, regime_frame=frame)
    assert card["n_trades"] >= 1
    assert not np.isnan(card["ulcer_index"])
    assert card["regime_conditional"]["trend"]["neutral"]["n_periods"] > 0
