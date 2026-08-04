"""Tests for the universe-wide portfolio study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.portfolio import (
    backtest_portfolio,
    common_index,
    equal_weight,
    inverse_vol,
    per_symbol_signals,
    run_universe_backtest,
    union_index,
    validate_universe,
)
from meridian.signals import SignalConfig
from meridian.validation import WalkForwardSpec
from meridian.validation.stats import sharpe


def _reverting(n=600, seed=0, level=100.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + rng.normal(0, 1)
    return pd.Series(level + x, index=pd.RangeIndex(n))


def _universe(k=6, n=600) -> dict[str, pd.Series]:
    return {f"S{i}": _reverting(n, seed=i) for i in range(k)}


# --- sizing ---------------------------------------------------------------

def test_equal_weight_splits_active_names():
    sig = pd.DataFrame({"A": [1, 0], "B": [1, 0], "C": [-1, 0]})
    w = equal_weight(sig)
    assert list(w.iloc[0]) == pytest.approx([1 / 3, 1 / 3, -1 / 3])
    assert list(w.iloc[1]) == [0.0, 0.0, 0.0]  # flat row


def test_equal_weight_gross_is_one_when_active():
    sig = pd.DataFrame({"A": [1, 1], "B": [-1, 0]})
    w = equal_weight(sig)
    assert w.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)


def test_inverse_vol_normalizes_and_favors_quiet_names():
    rng = np.random.default_rng(0)
    n = 100
    returns = pd.DataFrame({
        "quiet": rng.normal(0, 0.005, n),
        "wild": rng.normal(0, 0.05, n),
    })
    sig = pd.DataFrame({"quiet": np.ones(n), "wild": np.ones(n)})
    w = inverse_vol(sig, returns, lookback=20)
    last = w.iloc[-1]
    assert last.abs().sum() == pytest.approx(1.0)
    assert last["quiet"] > last["wild"]  # quieter name gets more capital


# --- portfolio backtest ---------------------------------------------------

def test_portfolio_no_lookahead():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 55.0, 60.5]})
    signals = pd.DataFrame({"A": [0, 0, 1], "B": [0, 0, 1]})  # only last bar active
    res = backtest_portfolio(signals, prices, cost_bps=0.0)
    assert res.equity.iloc[-1] == pytest.approx(1.0)  # lagged weight earns nothing


def test_portfolio_earns_blended_return():
    prices = pd.DataFrame({"A": [100.0, 110.0], "B": [100.0, 120.0]})
    signals = pd.DataFrame({"A": [1, 1], "B": [1, 1]})
    res = backtest_portfolio(signals, prices, cost_bps=0.0)
    # held=[0.5,0.5] from bar0; bar1 return = 0.5*10% + 0.5*20% = 15%
    assert res.returns.iloc[1] == pytest.approx(0.15)


def test_portfolio_costs_reduce_return():
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0], "B": [100.0, 100.0, 100.0]})
    signals = pd.DataFrame({"A": [1, 1, 1], "B": [1, 1, 1]})
    free = backtest_portfolio(signals, prices, cost_bps=0.0).equity.iloc[-1]
    costed = backtest_portfolio(signals, prices, cost_bps=10.0).equity.iloc[-1]
    assert costed < free


# --- fill realism (mirrors signals/backtest.py's single-asset fix) ---------

def test_portfolio_no_open_prices_falls_back_to_close_approx():
    prices = pd.DataFrame({"A": [100.0, 101.0], "B": [100.0, 99.0]})
    signals = pd.DataFrame({"A": [1, 1], "B": [1, 1]})
    res = backtest_portfolio(signals, prices, cost_bps=0.0)
    assert res.meta["fill_realism"] == "close_approx"


def test_portfolio_open_prices_uses_next_open_fill():
    # Flat closes but a gap between close and next open on bar 1.
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0], "B": [100.0, 100.0, 100.0]})
    open_prices = pd.DataFrame({"A": [100.0, 105.0, 110.0], "B": [100.0, 105.0, 110.0]})
    signals = pd.DataFrame({"A": [1, 1, 1], "B": [1, 1, 1]})

    close_approx = backtest_portfolio(signals, prices, cost_bps=0.0)
    next_open = backtest_portfolio(signals, prices, cost_bps=0.0, open_prices=open_prices)

    assert close_approx.meta["fill_realism"] == "close_approx"
    assert next_open.meta["fill_realism"] == "next_open"
    assert close_approx.equity.iloc[-1] == pytest.approx(1.0)   # flat close series
    assert next_open.equity.iloc[-1] != pytest.approx(1.0)      # gap is captured


def test_portfolio_open_prices_all_nan_falls_back():
    prices = pd.DataFrame({"A": [100.0, 101.0], "B": [100.0, 99.0]})
    open_prices = pd.DataFrame({"A": [np.nan, np.nan], "B": [np.nan, np.nan]})
    signals = pd.DataFrame({"A": [1, 1], "B": [1, 1]})
    res = backtest_portfolio(signals, prices, cost_bps=0.0, open_prices=open_prices)
    assert res.meta["fill_realism"] == "close_approx"


# --- universe runner ------------------------------------------------------

def test_common_index_is_intersection():
    a = pd.Series(range(5), index=pd.RangeIndex(0, 5))
    b = pd.Series(range(5), index=pd.RangeIndex(2, 7))
    idx = common_index({"a": a, "b": b})
    assert list(idx) == [2, 3, 4]


def test_per_symbol_signals_shape():
    uni = _universe(k=3, n=300)
    sigs = per_symbol_signals(uni, "sma", "zscore", SignalConfig(entry_threshold=1.0), window=20)
    assert set(sigs.columns) == set(uni)
    assert set(np.unique(sigs.fillna(0).to_numpy())) <= {-1.0, 0.0, 1.0}


def test_union_index_handles_staggered_listings():
    early = pd.Series(range(10), index=pd.RangeIndex(0, 10))
    late = pd.Series(range(5), index=pd.RangeIndex(7, 12))  # "IPO" at bar 7
    uni = {"old": early, "new": late}
    assert list(union_index(uni)) == list(range(12))        # full span
    assert list(common_index(uni)) == [7, 8, 9]             # intersection collapses


def test_universe_backtest_runs_with_staggered_listings():
    # A recent 'IPO' (shorter history) must not collapse the study window.
    full = _reverting(400, seed=1)
    ipo = _reverting(400, seed=2).iloc[300:]   # only the last 100 bars exist
    res = run_universe_backtest(
        {"OLD": full, "NEW": ipo}, "sma", "zscore", SignalConfig(entry_threshold=1.0), window=20
    )
    assert len(res.returns) == 400              # union calendar, not 100
    assert res.gross_exposure.iloc[:50].sum() >= 0  # early bars: only OLD can be active


def test_run_universe_backtest_uses_next_open_fill_when_bars_given():
    uni = _universe(k=3, n=300)
    bars = {
        sym: pd.DataFrame({"open": px.values, "high": px + 1, "low": px - 1}, index=px.index)
        for sym, px in uni.items()
    }
    without = run_universe_backtest(uni, "sma", "zscore", SignalConfig(entry_threshold=1.0), window=20)
    with_bars = run_universe_backtest(
        uni, "sma", "zscore", SignalConfig(entry_threshold=1.0), window=20, bars_by_symbol=bars,
    )
    assert without.meta["fill_realism"] == "close_approx"
    assert with_bars.meta["fill_realism"] == "next_open"


# --- intraday: flatten-at-close (no overnight) -----------------------------

def test_flatten_overnight_forces_flat_at_session_close():
    idx = pd.DatetimeIndex(
        ["2024-01-02 09:30", "2024-01-02 10:30", "2024-01-02 11:30",   # session 1
         "2024-01-03 09:30", "2024-01-03 10:30", "2024-01-03 11:30"]   # session 2
    )
    cols = ["A", "B"]
    signals = pd.DataFrame(1, index=idx, columns=cols)   # always long both
    prices = pd.DataFrame({"A": [10, 11, 12, 13, 14, 15], "B": [20, 21, 22, 23, 24, 25]},
                          index=idx, dtype=float)
    res = backtest_portfolio(signals, prices, cost_bps=0.0, flatten_overnight=True)
    # held = weights lagged; flattened on each session's last bar (rows 2 and 5),
    # so the held book is 0 on the first bar of each session (rows 0 and 3).
    assert res.weights.loc[idx[3]].abs().sum() == pytest.approx(0.0)  # session-2 open: flat
    # ...and the overnight return (row 3) is therefore not booked
    assert res.returns.loc[idx[3]] == pytest.approx(0.0)


def test_flatten_overnight_default_false_unchanged():
    idx = pd.date_range("2024-01-02 09:30", periods=6, freq="1h")
    signals = pd.DataFrame(1, index=idx, columns=["A", "B"])
    prices = pd.DataFrame({"A": np.linspace(10, 15, 6), "B": np.linspace(20, 25, 6)}, index=idx)
    a = backtest_portfolio(signals, prices, cost_bps=0.0)
    b = backtest_portfolio(signals, prices, cost_bps=0.0, flatten_overnight=False)
    pd.testing.assert_series_equal(a.returns, b.returns)


def test_validate_universe_flatten_overnight_runs():
    rng = np.random.default_rng(0)
    # 4 sessions x ~60 intraday bars, 4 names
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02") + pd.Timedelta(days=d, hours=h)
         for d in range(40) for h in range(7)]
    )
    uni = {f"S{i}": pd.Series(100 + np.cumsum(rng.normal(0, 0.2, len(idx))), index=idx)
           for i in range(4)}
    spec = WalkForwardSpec(mode="anchored", min_train=120, test_span=60, step=60)
    df = validate_universe(
        uni, ["sma"], "zscore", SignalConfig(entry_threshold=1.0), spec=spec, window=10,
        n_boot=80, n_mc=60, block=10, seed=0, periods_per_year=1764, flatten_overnight=True,
    )
    assert df["estimator"].tolist() == ["sma"]
    assert df["significant"].dtype == bool


# --- signal series separate from P&L prices (relative-value strategies) ----

def test_signal_prices_default_matches_absolute():
    """signal_prices_by_symbol=None must reproduce the absolute strategy exactly."""
    uni = _universe(k=5)
    sig = SignalConfig(entry_threshold=1.0)
    base = run_universe_backtest(uni, "sma", "zscore", sig, window=20)
    same = run_universe_backtest(uni, "sma", "zscore", sig, window=20,
                                 signal_prices_by_symbol=None)
    pd.testing.assert_series_equal(base.returns, same.returns)


def test_signals_from_relative_pnl_from_actual():
    """Signals come from the relative series, P&L from the tradable prices."""
    from meridian.features import cross_sectional_demean

    uni = _universe(k=6)
    rel = cross_sectional_demean(uni)
    res = run_universe_backtest(
        uni, "sma", "zscore", SignalConfig(entry_threshold=1.0), window=20,
        signal_prices_by_symbol=rel,
    )
    # Uses the relative series for signals -> different from the absolute run,
    # but P&L is real (finite equity, returns indexed by the price calendar).
    base = run_universe_backtest(uni, "sma", "zscore", SignalConfig(entry_threshold=1.0), window=20)
    assert np.isfinite(res.equity.iloc[-1])
    assert not res.returns.equals(base.returns)
    assert res.returns.index.equals(union_index(uni))


def test_validate_universe_accepts_signal_prices():
    from meridian.features import cross_sectional_demean

    uni = _universe(k=6, n=700)
    rel = cross_sectional_demean(uni)
    spec = WalkForwardSpec(mode="anchored", min_train=250, test_span=100, step=100)
    df = validate_universe(
        uni, ["sma", "ou"], "zscore", SignalConfig(entry_threshold=1.0),
        spec=spec, window=20, n_boot=120, n_mc=100, block=10, seed=0,
        signal_prices_by_symbol=rel,
    )
    assert set(df["estimator"]) == {"sma", "ou"}
    assert df["significant"].dtype == bool


def test_handles_missing_prices_from_membership_gating():
    """Membership-gated universes have NaN prices (pre-listing / post-removal).
    The pipeline must skip those bars without crashing the estimators."""
    full = _reverting(400, seed=1)
    gated = _reverting(400, seed=2).copy()
    gated.iloc[:150] = np.nan   # "not yet a member" early
    gated.iloc[350:] = np.nan   # "removed from index" late
    uni = {"FULL": full, "GATED": gated}

    # ou uses a least-squares fit that previously crashed on NaN windows.
    res = run_universe_backtest(uni, "ou", "zscore", SignalConfig(entry_threshold=1.0), window=20)
    assert np.isfinite(res.equity.iloc[-1])
    sigs = per_symbol_signals(uni, "ou", "zscore", SignalConfig(entry_threshold=1.0), window=20)
    # GATED is inactive (NaN signal) exactly where it had no price
    assert sigs["GATED"].iloc[:150].isna().all()
    assert sigs["GATED"].iloc[350:].isna().all()


def test_run_universe_backtest_produces_returns():
    uni = _universe(k=5)
    res = run_universe_backtest(uni, "ou", "zscore", SignalConfig(entry_threshold=1.0), window=20)
    assert len(res.returns) == len(common_index(uni))
    assert res.meta["n_symbols"] == 5
    assert (res.gross_exposure <= 1.0 + 1e-9).all()


# --- the payoff: more symbols -> tighter Sharpe estimate ------------------

def test_portfolio_lifts_sharpe_above_any_single_name():
    """Diversifying an edge across many uncorrelated names raises the Sharpe
    toward ~SR*sqrt(K) — the statistical-power gain motivating the universe
    study. (The Sharpe *CI width* does not shrink: it tracks the number of
    time periods, not cross-sectional breadth — so power comes from a higher
    point estimate, not a narrower interval.)"""
    uni = _universe(k=12, n=800)
    sig = SignalConfig(entry_threshold=1.0)

    port_sharpe = sharpe(run_universe_backtest(uni, "ou", "zscore", sig, window=20).returns)
    single_sharpes = [
        sharpe(run_universe_backtest({k: uni[k]}, "ou", "zscore", sig, window=20).returns)
        for k in uni
    ]
    # The portfolio beats even the best single name (diversification benefit).
    assert port_sharpe > max(single_sharpes)


# --- universe validation --------------------------------------------------

def test_validate_universe_verdict_table():
    uni = _universe(k=6, n=700)
    spec = WalkForwardSpec(mode="anchored", min_train=250, test_span=100, step=100)
    df = validate_universe(
        uni, ["sma", "ou", "ens_invvar"], "zscore", SignalConfig(entry_threshold=1.0),
        spec=spec, window=20, n_boot=150, n_mc=120, block=10, seed=0,
    )
    for col in ["estimator", "oos_sharpe", "boot_ci_low", "mc_pvalue", "q_value",
                "significant", "n_symbols", "dsr_pvalue", "m_eff", "q_value_eff",
                "significant_eff"]:
        assert col in df.columns
    assert set(df["estimator"]) == {"sma", "ou", "ens_invvar"}
    assert df["significant"].dtype == bool
    assert df["significant_eff"].dtype == bool
    assert (df["n_symbols"] == 6).all()
    assert df["oos_sharpe"].is_monotonic_decreasing
    assert (df["m_eff"] <= len(df)).all() and (df["m_eff"] >= 1.0).all()
    assert df["dsr_pvalue"].between(0.0, 1.0).all()


def test_validate_universe_m_eff_lower_for_correlated_estimators():
    """Near-identical estimators on the same data should show heavy collinearity."""
    uni = _universe(k=6, n=700)
    spec = WalkForwardSpec(mode="anchored", min_train=250, test_span=100, step=100)
    # sma/wma/trima are all short-window moving-average variants -> highly correlated.
    df = validate_universe(
        uni, ["sma", "wma", "trima"], "zscore", SignalConfig(entry_threshold=1.0),
        spec=spec, window=20, n_boot=100, n_mc=100, block=10, seed=0,
    )
    assert df["m_eff"].iloc[0] < 3.0
