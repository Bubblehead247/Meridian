"""Metamorphic/property tests for the backtest engines (P1-D).

These do not check a specific expected number for a specific input — they check an
invariant that must hold across *any* input, which is exactly the kind of test that
would have caught the P0 same-bar-close regression (a fix applied to one engine,
silently absent from a sibling engine one commit later). See
research_integrity_gap_analysis.md §3.11/§14 for why this class of test is a priority.

Covers all six invariants from the audit brief's §14: no-lookahead / future-data
mutation, truncation invariance, price-scaling invariance, zero-signal -> zero P&L,
cost monotonicity, and (split-invariance / duplicate-assets, added last) — across both
the single-asset engine (signals/backtest.py) and the cross-sectional engine
(portfolio/portfolio.py), since the P0 bug was specifically an engine that diverged
from its sibling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.portfolio.portfolio import backtest_portfolio
from meridian.signals.backtest import backtest, run_backtest
from meridian.signals.engine import SignalConfig


def _prices(n=500, seed=0, start="2018-01-01"):
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))), index=idx)


def _bars(prices):
    return pd.DataFrame(
        {"open": prices.shift(1).fillna(prices.iloc[0]), "high": prices * 1.01,
         "low": prices * 0.99, "close": prices},
        index=prices.index,
    )


def _random_positions(prices, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice([-1, 0, 1], size=len(prices)), index=prices.index)


# --- no-lookahead / future-data mutation ------------------------------------
# "Modify data after time t. Nothing before t should change." The single most
# important invariant per the audit brief — it is a direct regression guard for the
# P0-A bug (same-bar-close look-ahead silently present in one engine, absent in its
# sibling).

def test_no_lookahead_full_pipeline_mutating_future_prices():
    prices = _prices(n=600)
    bars = _bars(prices)
    cut = 400

    res1 = run_backtest(prices, "sma", "zscore", window=20, bars=bars)

    mutated = prices.copy()
    rng = np.random.default_rng(99)
    mutated.iloc[cut:] = mutated.iloc[cut:] * (1.0 + rng.normal(0, 0.5, len(mutated) - cut))
    mutated_bars = _bars(mutated)
    res2 = run_backtest(mutated, "sma", "zscore", window=20, bars=mutated_bars)

    # 2-bar buffer absorbs the next-open fill's 1-bar lookahead into t+1.
    safe = cut - 2
    pd.testing.assert_series_equal(res1.positions.iloc[:safe], res2.positions.iloc[:safe])
    pd.testing.assert_series_equal(res1.returns.iloc[:safe], res2.returns.iloc[:safe])


def test_no_lookahead_single_asset_engine_mutating_future_bars():
    prices = _prices(n=300)
    positions = _random_positions(prices)
    bars = _bars(prices)
    cut = 200

    res1 = backtest(prices, positions, bars=bars)

    mutated_bars = bars.copy()
    mutated_bars.iloc[cut:] = mutated_bars.iloc[cut:] * 3.0
    res2 = backtest(prices, positions, bars=mutated_bars)

    safe = cut - 2
    pd.testing.assert_series_equal(res1.returns.iloc[:safe], res2.returns.iloc[:safe])
    pd.testing.assert_series_equal(res1.equity.iloc[:safe], res2.equity.iloc[:safe])


def test_no_lookahead_cross_sectional_engine_mutating_future_prices():
    idx = pd.date_range("2018-01-01", periods=400, freq="B")
    rng = np.random.default_rng(7)
    symbols = ["A", "B", "C"]
    prices = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx)))) for s in symbols},
        index=idx,
    )
    opens = prices.shift(1).bfill()
    signals = pd.DataFrame(
        rng.choice([-1, 0, 1], size=(len(idx), len(symbols))), index=idx, columns=symbols
    )
    cut = 250

    res1 = backtest_portfolio(signals, prices, open_prices=opens)

    mutated_prices = prices.copy()
    mutated_prices.iloc[cut:] = mutated_prices.iloc[cut:] * 5.0
    mutated_opens = mutated_prices.shift(1).bfill()
    res2 = backtest_portfolio(signals, mutated_prices, open_prices=mutated_opens)

    safe = cut - 2
    pd.testing.assert_series_equal(res1.returns.iloc[:safe], res2.returns.iloc[:safe])


def test_no_lookahead_cross_sectional_close_approx_mutating_future_prices():
    # Same invariant, but exercising the close_approx (no open_prices) fallback path —
    # the exact path that leaked in the P0-A bug when bars_by_symbol was never threaded
    # through. A same-bar-close implementation must still not depend on data past t.
    idx = pd.date_range("2018-01-01", periods=300, freq="B")
    rng = np.random.default_rng(11)
    symbols = ["A", "B"]
    prices = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx)))) for s in symbols},
        index=idx,
    )
    signals = pd.DataFrame(
        rng.choice([-1, 0, 1], size=(len(idx), len(symbols))), index=idx, columns=symbols
    )
    cut = 200

    res1 = backtest_portfolio(signals, prices)  # no open_prices -> close_approx

    mutated_prices = prices.copy()
    mutated_prices.iloc[cut:] = mutated_prices.iloc[cut:] * 5.0
    res2 = backtest_portfolio(signals, mutated_prices)

    safe = cut - 1  # close_approx has no extra fill-lag buffer
    pd.testing.assert_series_equal(res1.returns.iloc[:safe], res2.returns.iloc[:safe])


# --- truncation invariance ---------------------------------------------------
# "Removing future bars must not alter historical trades."

def test_truncation_invariance_single_asset():
    prices = _prices(n=500)
    bars = _bars(prices)
    cut = 350

    full = run_backtest(prices, "sma", "zscore", window=20, bars=bars)
    truncated = run_backtest(
        prices.iloc[:cut], "sma", "zscore", window=20, bars=bars.iloc[:cut]
    )

    safe = cut - 2
    pd.testing.assert_series_equal(
        full.positions.iloc[:safe], truncated.positions.iloc[:safe]
    )
    pd.testing.assert_series_equal(full.returns.iloc[:safe], truncated.returns.iloc[:safe])


def test_truncation_invariance_cross_sectional():
    idx = pd.date_range("2018-01-01", periods=400, freq="B")
    rng = np.random.default_rng(5)
    symbols = ["A", "B", "C"]
    prices = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx)))) for s in symbols},
        index=idx,
    )
    opens = prices.shift(1).bfill()
    signals = pd.DataFrame(
        rng.choice([-1, 0, 1], size=(len(idx), len(symbols))), index=idx, columns=symbols
    )
    cut = 300

    full = backtest_portfolio(signals, prices, open_prices=opens)
    truncated = backtest_portfolio(
        signals.iloc[:cut], prices.iloc[:cut], open_prices=opens.iloc[:cut]
    )

    safe = cut - 2
    pd.testing.assert_series_equal(full.returns.iloc[:safe], truncated.returns.iloc[:safe])


# --- zero-signal -> zero P&L except costs ------------------------------------

def test_zero_signal_single_asset_produces_zero_pnl():
    prices = _prices(n=200)
    flat = pd.Series(0, index=prices.index)
    res = backtest(prices, flat, cost_bps=5.0)
    assert (res.gross_returns == 0.0).all()
    assert (res.returns == 0.0).all()  # no position changes -> no turnover -> no cost either
    assert res.equity.iloc[-1] == pytest.approx(1.0)


def test_zero_signal_cross_sectional_produces_zero_pnl():
    idx = pd.date_range("2018-01-01", periods=200, freq="B")
    prices = pd.DataFrame({"A": 100.0, "B": 50.0}, index=idx)
    signals = pd.DataFrame(0, index=idx, columns=["A", "B"])
    res = backtest_portfolio(signals, prices, cost_bps=5.0)
    assert (res.returns == 0.0).all()
    assert res.equity.iloc[-1] == pytest.approx(1.0)


# --- cost monotonicity --------------------------------------------------------
# "Increasing transaction costs must not increase net performance."

def test_cost_monotonicity_single_asset():
    prices = _prices(n=300)
    positions = _random_positions(prices)
    costs = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    totals = [
        backtest(prices, positions, cost_bps=c).equity.iloc[-1] for c in costs
    ]
    assert all(a >= b - 1e-12 for a, b in zip(totals, totals[1:]))


def test_cost_monotonicity_cross_sectional():
    idx = pd.date_range("2018-01-01", periods=300, freq="B")
    rng = np.random.default_rng(3)
    symbols = ["A", "B", "C"]
    prices = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx)))) for s in symbols},
        index=idx,
    )
    signals = pd.DataFrame(
        rng.choice([-1, 0, 1], size=(len(idx), len(symbols))), index=idx, columns=symbols
    )
    costs = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    totals = [
        backtest_portfolio(signals, prices, cost_bps=c).equity.iloc[-1] for c in costs
    ]
    assert all(a >= b - 1e-12 for a, b in zip(totals, totals[1:]))


# --- price-scaling invariance -------------------------------------------------
# "Multiplying all prices by a constant should not change percentage-return-based
# strategy results." sma/zscore is a relative-deviation pipeline (moving-average
# residual normalized by its own rolling std), so it is scale-invariant by
# construction — this test protects that property from a future estimator/deviation
# change that accidentally introduces an absolute (non-relative) threshold.

def test_price_scaling_invariance_full_pipeline():
    prices = _prices(n=400)
    bars = _bars(prices)
    res1 = run_backtest(prices, "sma", "zscore", window=20, bars=bars)

    scaled = prices * 137.0
    scaled_bars = _bars(scaled)
    res2 = run_backtest(scaled, "sma", "zscore", window=20, bars=scaled_bars)

    pd.testing.assert_series_equal(res1.positions, res2.positions)
    np.testing.assert_allclose(
        res1.returns.to_numpy(), res2.returns.to_numpy(), atol=1e-9
    )


# --- split invariance ---------------------------------------------------------
# "A correctly modeled stock split must not create an artificial return." A properly
# split-adjusted price series (what `adj_close` is supposed to provide) is continuous
# across the split date — historical prices are retroactively rescaled so there's no
# jump. This is the positive half of the invariant: feeding the backtester correctly
# adjusted data produces no artificial return at the split date. The negative control
# (feeding it raw, unadjusted data — e.g. accidentally using `close` instead of
# `adj_close`) confirms the failure mode this invariant protects against is real and
# would actually be caught: a genuine data-adjustment bug shows up as a huge spurious
# single-bar return, not silently absorbed.

def test_split_invariance_adjusted_prices_show_no_artificial_return():
    n = 300
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(21)
    split_at = 150
    split_factor = 2.0  # a 2:1 split

    # Properly adjusted: continuous random walk, no jump anywhere (what adj_close
    # is supposed to be — history is retroactively rescaled by the split factor).
    adjusted = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))), index=idx)

    # Negative control: an *unadjusted* feed. Pre-split raw closes were nominally
    # split_factor times higher (fewer, pricier shares) than the split-adjusted
    # series shows — exactly what you'd get from serving raw `close` instead of
    # `adj_close` across a real split.
    unadjusted = adjusted.copy()
    unadjusted.iloc[:split_at] = unadjusted.iloc[:split_at] * split_factor

    positions = pd.Series(1, index=idx)  # constant long exposure -> return = price return

    adj_res = backtest(adjusted, positions, cost_bps=0.0)
    unadj_res = backtest(unadjusted, positions, cost_bps=0.0)

    # The split boundary's return: adjusted stays a normal daily move; unadjusted
    # shows the artificial ~-50% jump from the un-rescaled pre-split history meeting
    # the (correctly-scaled) post-split price.
    assert abs(adj_res.returns.iloc[split_at]) < 0.10
    assert unadj_res.returns.iloc[split_at] < -0.30
    assert abs(unadj_res.returns.iloc[split_at]) > 5 * abs(adj_res.returns.iloc[split_at])

    # Away from the split boundary, both series agree exactly (same underlying path).
    pd.testing.assert_series_equal(
        adj_res.returns.iloc[split_at + 1:], unadj_res.returns.iloc[split_at + 1:]
    )


# --- duplicate assets ----------------------------------------------------------
# "Duplicating identical assets should produce predictable portfolio behavior." For
# equal_weight sizing, "predictable" is not "unchanged" — a duplicate is one more
# active vote, which dilutes every name's weight by a known, closed-form factor
# (n_active / (n_active + 1)). The test asserts the exact closed-form result, not just
# "nothing crashes" — silent double-counting or a NaN/inf weight would fail this.

def test_duplicate_asset_dilutes_weights_by_the_exact_closed_form_factor():
    idx = pd.date_range("2018-01-01", periods=200, freq="B")
    rng = np.random.default_rng(22)
    prices = pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx)))),
        "B": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx)))),
    }, index=idx)
    signals = pd.DataFrame({"A": 1, "B": 1}, index=idx)  # both always active -> n=2

    dup_prices = prices.assign(A_dup=prices["A"])  # identical price series
    dup_signals = signals.assign(A_dup=signals["A"])  # identical signal

    orig = backtest_portfolio(signals, prices)
    dup = backtest_portfolio(dup_signals, dup_prices)

    # Original: 2 active names -> each weight = 1/2. With the duplicate: 3 active
    # names -> each weight = 1/3. B's weight scales by exactly 2/3 = n/(n+1).
    scale = 2.0 / 3.0
    np.testing.assert_allclose(
        dup.weights["B"].to_numpy(), (orig.weights["B"] * scale).to_numpy(), atol=1e-12
    )
    # A's combined exposure (A + A_dup) is 2/3 of total, vs. A alone at 1/2 originally
    # — the duplicate doesn't cancel out or vanish, it's counted as its own vote.
    combined_a = dup.weights["A"] + dup.weights["A_dup"]
    np.testing.assert_allclose(
        combined_a.to_numpy(), (orig.weights["A"] * (4.0 / 3.0)).to_numpy(), atol=1e-12
    )
    assert np.isfinite(dup.weights.to_numpy()).all()
    assert np.isfinite(dup.returns.to_numpy()).all()

    # A and its duplicate must have byte-identical realized returns (same price path).
    pd.testing.assert_series_equal(
        dup.returns_by_symbol["A"], dup.returns_by_symbol["A_dup"], check_names=False
    )
