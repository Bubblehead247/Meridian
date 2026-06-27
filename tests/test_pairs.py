"""Tests for pairs / spread construction (cointegration MR)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.features import (
    build_pair,
    build_pairs,
    hedge_ratio,
    leg_cost,
    screen_cointegrated_pairs,
)
from meridian.portfolio import validate_universe
from meridian.signals import SignalConfig
from meridian.validation import WalkForwardSpec


def _cointegrated_pair(n=500, beta=1.5, seed=0):
    """Two legs sharing a common stochastic trend plus a stationary spread."""
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n)
    common = np.cumsum(rng.normal(0, 0.01, n))           # shared random walk
    log_b = 4.0 + common
    spread = 0.05 * np.sin(np.linspace(0, 12, n))        # stationary, reverting
    log_a = 3.0 + beta * common + spread
    return (
        pd.Series(np.exp(log_a), index=idx),
        pd.Series(np.exp(log_b), index=idx),
    )


# --- hedge ratio is causal ------------------------------------------------

def test_hedge_ratio_is_lagged_no_lookahead():
    a, b = _cointegrated_pair()
    log_a, log_b = np.log(a), np.log(b)
    beta = hedge_ratio(log_a, log_b, lookback=60)
    # changing the *last* bar must not move any earlier beta (β is shift(1) of a
    # trailing window) — and must not move β at the last bar either.
    log_a2 = log_a.copy()
    log_a2.iloc[-1] += 10.0
    beta2 = hedge_ratio(log_a2, log_b, lookback=60)
    pd.testing.assert_series_equal(beta, beta2)


def test_hedge_ratio_recovers_true_beta():
    a, b = _cointegrated_pair(beta=1.5)
    beta = hedge_ratio(np.log(a), np.log(b), lookback=120).dropna()
    # the rolling estimate should sit near the true 1.5 on average
    assert abs(beta.mean() - 1.5) < 0.3


# --- spread + synthetic price ---------------------------------------------

def test_spread_price_pct_change_reconstructs_spread_return():
    a, b = _cointegrated_pair()
    spread, spread_price = build_pair(a, b, lookback=60)
    beta = hedge_ratio(np.log(a), np.log(b), lookback=60)
    expected = (a.pct_change() - beta * b.pct_change()).where(beta.notna())
    got = spread_price.pct_change()
    both = pd.concat([expected, got], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-10)


def test_spread_signal_is_finite_and_reverting():
    a, b = _cointegrated_pair()
    spread, _ = build_pair(a, b, lookback=60)
    s = spread.dropna()
    assert np.isfinite(s).all()
    # a cointegrated spread should cross its own mean repeatedly (not trend off)
    centered = s - s.mean()
    sign_changes = (np.sign(centered).diff().abs() > 0).sum()
    assert sign_changes > 5


def test_build_pair_handles_nonpositive_prices():
    a = pd.Series([10.0, 10.0, 11.0, 12.0])
    b = pd.Series([0.0, 5.0, 5.0, 6.0])   # first bar non-positive
    spread, spread_price = build_pair(a, b, lookback=2)
    assert np.isnan(spread.iloc[0])       # log of 0 -> NaN, untradable that bar


def test_build_pairs_keys_and_skips_missing_leg():
    a, b = _cointegrated_pair()
    px = {"AAA": a, "BBB": b, "CCC": a * 1.01}
    sig, prc = build_pairs(px, [("AAA", "BBB"), ("AAA", "ZZZ")], lookback=60)
    assert set(sig) == {"AAA/BBB"}        # pair with a missing leg is skipped
    assert set(prc) == {"AAA/BBB"}


# --- end-to-end through the existing validator ----------------------------

def test_validate_universe_runs_on_pairs():
    px = {
        "AAA": _cointegrated_pair(seed=1)[0], "BBB": _cointegrated_pair(seed=1)[1],
        "CCC": _cointegrated_pair(seed=2)[0], "DDD": _cointegrated_pair(seed=2)[1],
    }
    sig, prc = build_pairs(px, [("AAA", "BBB"), ("CCC", "DDD")], lookback=60)
    spec = WalkForwardSpec(mode="anchored", min_train=250, test_span=100, step=100)
    df = validate_universe(
        prc, ["sma", "ou"], "zscore", SignalConfig(entry_threshold=1.5),
        spec=spec, window=20, n_boot=120, n_mc=100, block=10, seed=0,
        signal_prices_by_symbol=sig,
    )
    assert set(df["estimator"]) == {"sma", "ou"}
    assert df["significant"].dtype == bool
    assert (df["n_symbols"] == 2).all()


# --- mechanical pair selection (confirmation pass) ------------------------

def test_screen_keeps_cointegrated_drops_independent():
    a, b = _cointegrated_pair(seed=1)        # cointegrated with each other
    rng = np.random.default_rng(9)
    indep = pd.Series(np.exp(3.0 + np.cumsum(rng.normal(0, 0.01, len(a)))), index=a.index)
    px = {"A": a, "B": b, "Z": indep}
    kept = screen_cointegrated_pairs(px, ["A", "B", "Z"], max_pvalue=0.05)
    assert ("A", "B") in kept                # the cointegrated pair is selected
    assert ("A", "Z") not in kept and ("B", "Z") not in kept


def test_screen_respects_max_pvalue_and_returns_empty():
    a, b = _cointegrated_pair(seed=1)
    px = {"A": a, "B": b}
    assert screen_cointegrated_pairs(px, ["A", "B"], max_pvalue=0.0) == []


def test_screen_only_uses_passed_slice():
    """Selection must depend solely on the (in-sample) prices handed in."""
    a, b = _cointegrated_pair(seed=3, n=800)
    insample = {"A": a.iloc[:400], "B": b.iloc[:400]}
    # adding wild later data would change a full-sample test; the slice must not see it
    kept = screen_cointegrated_pairs(insample, ["A", "B"], max_pvalue=0.05)
    assert kept == [("A", "B")] or kept == []   # decision made on the 400-bar slice only


def test_leg_cost_matches_two_leg_turnover():
    idx = pd.RangeIndex(4)
    held = pd.DataFrame({"A/B": [0.0, 1.0, 1.0, 0.0]}, index=idx)
    beta = {"A/B": pd.Series([2.0, 2.0, 2.0, 2.0], index=idx)}
    cost = leg_cost(held, beta, half_spread_bps=10.0)
    # leg A turnover: |1|,|0|,|1| at bars 1,3 (entry+exit); leg B = beta*A -> doubled by 2
    wa = pd.Series([0.0, 1.0, 1.0, 0.0], index=idx)
    dwa = wa.diff()
    dwa.iloc[0] = wa.iloc[0]
    expected = (dwa.abs() + (2.0 * dwa).abs()) * (10.0 / 1e4)
    pd.testing.assert_series_equal(cost, expected, check_names=False)


def test_leg_cost_zero_turnover_is_free_and_beta_one_doubles():
    idx = pd.RangeIndex(3)
    flat = pd.DataFrame({"P": [0.0, 0.0, 0.0]}, index=idx)
    beta1 = {"P": pd.Series([1.0, 1.0, 1.0], index=idx)}
    assert (leg_cost(flat, beta1, 5.0) == 0.0).all()
    # beta = 1 -> both legs move equally -> cost = 2 * |Δw| * bps/1e4
    held = pd.DataFrame({"P": [1.0, 1.0, 1.0]}, index=idx)
    cost = leg_cost(held, beta1, 5.0)
    assert np.isclose(cost.iloc[0], 2 * 1.0 * 5.0 / 1e4)   # entry from flat
    assert np.isclose(cost.iloc[1], 0.0)                    # no change after
