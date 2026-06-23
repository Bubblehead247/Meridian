"""Phase 6 tests: walk-forward, bootstrap, Monte-Carlo, correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.signals import SignalConfig, run_backtest
from meridian.validation import (
    WalkForwardSpec,
    benjamini_hochberg,
    block_bootstrap_sharpe,
    bonferroni,
    make_folds,
    monte_carlo_pvalue,
    validate,
    walk_forward,
)
from meridian.validation.stats import sharpe, strategy_net


def _mean_reverting(n=600, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 1)
    return pd.Series(100 + x, index=pd.RangeIndex(n))


# --- fold scheduling ------------------------------------------------------

def test_anchored_folds_tile_and_are_forward_only():
    folds = make_folds(10, WalkForwardSpec(mode="anchored", min_train=4, test_span=2, step=2))
    assert [(f.train_start, f.train_end, f.test_start, f.test_end) for f in folds] == [
        (0, 4, 4, 6), (0, 6, 6, 8), (0, 8, 8, 10),
    ]
    # test windows are disjoint and forward-only
    ends = [f.test_start for f in folds]
    assert ends == sorted(ends)


def test_rolling_folds_use_fixed_train_window():
    folds = make_folds(10, WalkForwardSpec(mode="rolling", train_span=4, min_train=4, test_span=2, step=2))
    assert folds[0].train_start == 0 and folds[1].train_start == 2
    for f in folds:
        assert f.train_end - f.train_start == 4  # fixed length


def test_spec_validates_mode():
    with pytest.raises(ValueError):
        WalkForwardSpec(mode="sideways")


# --- walk-forward correctness (no look-ahead) -----------------------------

def test_anchored_stitched_equals_full_backtest_slices():
    """Anchored OOS returns must equal a single full backtest sliced to the
    test windows — proving the fold machinery introduces no look-ahead."""
    prices = _mean_reverting()
    spec = WalkForwardSpec(mode="anchored", min_train=200, test_span=100, step=100)
    sig = SignalConfig(entry_threshold=1.0)

    wf = walk_forward(prices, ["sma"], ["zscore"], sig, spec=spec, window=20, cost_bps=1.0)
    full = run_backtest(prices, "sma", "zscore", sig, window=20, cost_bps=1.0)

    stitched = wf.stitched_returns(("sma", "zscore"))
    expected = pd.concat(
        [full.returns.iloc[f.test_start : f.test_end] for f in wf.folds]
    )
    pd.testing.assert_series_equal(stitched, expected)


def test_summary_and_selection_run():
    prices = _mean_reverting()
    spec = WalkForwardSpec(mode="anchored", min_train=200, test_span=100, step=100)
    wf = walk_forward(prices, ["sma", "ema", "ou"], ["zscore"], SignalConfig(entry_threshold=1.0), spec=spec)
    summ = wf.summary()
    assert set(summ["estimator"]) == {"sma", "ema", "ou"}
    assert "oos_sharpe" in summ.columns

    sel = wf.selection()
    assert len(sel.chosen) == len(wf.folds)
    # each chosen key is the per-fold argmax of the train metric
    for fd, chosen in zip(wf.fold_data, sel.chosen):
        best = max(wf.keys, key=lambda k: (-np.inf if np.isnan(fd[k]["train_metric"]) else fd[k]["train_metric"]))
        assert chosen == best


def test_rolling_mode_runs():
    prices = _mean_reverting()
    spec = WalkForwardSpec(mode="rolling", train_span=200, min_train=200, test_span=100, step=100)
    wf = walk_forward(prices, ["sma"], ["zscore"], SignalConfig(entry_threshold=1.0), spec=spec)
    assert len(wf.stitched_returns(("sma", "zscore"))) > 0


# --- bootstrap ------------------------------------------------------------

def test_bootstrap_ci_brackets_point_and_flags_positive():
    rng = np.random.default_rng(1)
    rets = pd.Series(0.001 + rng.normal(0, 0.005, 500))  # clearly positive drift
    out = block_bootstrap_sharpe(rets, n_boot=400, block=10, seed=0)
    assert out["ci_low"] <= out["point"] <= out["ci_high"]
    assert out["p_value"] < 0.1  # rarely <= 0 -> significant


def test_bootstrap_noise_is_not_significant():
    rng = np.random.default_rng(2)
    rets = pd.Series(rng.normal(0, 0.01, 500))  # zero mean
    out = block_bootstrap_sharpe(rets, n_boot=400, block=10, seed=0)
    assert out["p_value"] > 0.1  # zero-mean noise -> not significant


def test_bootstrap_degenerate_returns_nan():
    out = block_bootstrap_sharpe(pd.Series([0.01, 0.02]), n_boot=100, block=20)
    assert np.isnan(out["point"])


# --- monte carlo ----------------------------------------------------------

def test_monte_carlo_detects_perfect_timing():
    rng = np.random.default_rng(3)
    mret = rng.normal(0, 0.01, 400)
    held = np.sign(mret)  # perfectly positioned for each bar -> all gains
    out = monte_carlo_pvalue(mret, held, n=300, seed=0)
    assert out["p_value"] < 0.05  # real timing beats almost all rotations


def test_monte_carlo_random_positions_not_significant():
    rng = np.random.default_rng(4)
    mret = rng.normal(0, 0.01, 400)
    held = rng.choice([-1.0, 0.0, 1.0], size=400)  # unrelated to returns
    out = monte_carlo_pvalue(mret, held, n=300, seed=0)
    assert out["p_value"] > 0.05


def test_monte_carlo_flat_positions_nan():
    mret = np.random.default_rng(5).normal(0, 0.01, 100)
    out = monte_carlo_pvalue(mret, np.zeros(100), n=100)
    assert np.isnan(out["p_value"])


# --- multiple-testing correction ------------------------------------------

def test_benjamini_hochberg_rejects_expected_set():
    p = [0.01, 0.02, 0.03, 0.50, 0.51]  # m=5, alpha=0.05
    res = benjamini_hochberg(p, alpha=0.05)
    assert list(res["reject"]) == [True, True, True, False, False]
    assert (res["qvalues"] >= np.array(p)).all()  # q-values >= raw p


def test_bonferroni_threshold_and_adjusted():
    p = [0.01, 0.02, 0.03, 0.50, 0.51]
    res = bonferroni(p, alpha=0.05)
    assert res["alpha_adj"] == pytest.approx(0.01)
    assert list(res["reject"]) == [True, False, False, False, False]
    assert res["adjusted"][0] == pytest.approx(0.05)


def test_strategy_net_matches_manual():
    held = np.array([0.0, 1.0, 1.0, 0.0])
    mret = np.array([0.0, 0.10, 0.05, 0.20])
    net = strategy_net(held, mret, cost_bps=0.0)
    # gross: 0, 0.10, 0.05, 0 ; no cost
    assert net == pytest.approx([0.0, 0.10, 0.05, 0.0])


# --- end-to-end verdict table ---------------------------------------------

def test_validate_produces_verdict_table():
    prices = _mean_reverting()
    spec = WalkForwardSpec(mode="anchored", min_train=200, test_span=100, step=100)
    df = validate(
        prices, ["sma", "ema", "ou"], "zscore",
        SignalConfig(entry_threshold=1.0), spec=spec, window=20,
        n_boot=200, n_mc=200, block=10, seed=0,
    )
    for col in ["estimator", "oos_sharpe", "boot_ci_low", "boot_ci_high",
                "mc_pvalue", "q_value", "significant"]:
        assert col in df.columns
    assert len(df) == 3
    assert df["significant"].dtype == bool
    # sorted by oos_sharpe descending
    assert df["oos_sharpe"].is_monotonic_decreasing
