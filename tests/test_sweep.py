"""Tests for the parameter sweep (out-of-sample-validated tuning)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.experiments.sweep import (
    format_confirm,
    format_sweep,
    grid_for,
    sweep_and_confirm,
    sweep_single,
)
from meridian.validation import WalkForwardSpec

_SPEC = WalkForwardSpec(mode="anchored", min_train=300, test_span=150, step=150)


def _prices(n=800):
    idx = pd.date_range("2016-01-01", periods=n, freq="B")
    return pd.Series(100 + 6 * np.sin(np.linspace(0, 50, n)) + np.linspace(0, 18, n), index=idx)


def _bars(prices):
    return pd.DataFrame(
        {"open": prices.shift(1).fillna(prices.iloc[0]), "high": prices + 1,
         "low": prices - 1, "close": prices, "volume": np.linspace(1e6, 2e6, len(prices))}
    )


# --- grids ----------------------------------------------------------------

def test_grid_for_known_and_cross_sectional():
    assert grid_for("trend_following", "ma_trend") == {"window": [20, 50, 100, 150, 200]}
    assert grid_for("pullback_continuation", "ma_pullback") == {"fast": [10, 20], "slow": [50, 100]}
    assert grid_for("momentum", "relative_strength") == {}     # cross-sectional: no window knob


# --- sweep ----------------------------------------------------------------

def test_sweep_single_scores_in_and_out_of_sample():
    prices = _prices()
    df = sweep_single(prices, "trend_following", "ma_trend", bars=_bars(prices), spec=_SPEC)
    assert len(df) == 5                                        # one row per window
    for col in ("window", "is_sharpe", "is_passed", "oos_sharpe", "oos_passed", "oos_folds"):
        assert col in df.columns
    oos = df["oos_sharpe"].dropna().to_numpy()
    assert (np.diff(oos) <= 1e-9).all()                       # ranked by OOS Sharpe (desc)
    assert df["oos_folds"].iloc[0] >= 1                       # walk-forward actually ran


def test_sweep_two_parameter_grid_expands():
    prices = _prices()
    df = sweep_single(prices, "pullback_continuation", "ma_pullback", spec=_SPEC)
    assert len(df) == 4                                        # 2 fast x 2 slow
    assert {"fast", "slow"} <= set(df.columns)


def test_sweep_rejects_cross_sectional():
    with pytest.raises(ValueError, match="cross-sectional"):
        sweep_single(_prices(), "momentum", "relative_strength", spec=_SPEC)


def test_format_sweep_has_caveat_and_verdict():
    prices = _prices()
    df = sweep_single(prices, "trend_following", "ma_trend", bars=_bars(prices), spec=_SPEC)
    text = format_sweep(df, "trend_following", "ma_trend")
    assert "settings tried" in text
    assert "Multiple-testing" in text
    assert "Out-of-sample passes:" in text


# --- sweep + auto-confirm -------------------------------------------------

def _spanning_prices(n=3600):
    # spans the fixed in-sample (2010-2019) and OOS holdout (2020-2022) windows
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.Series(100 + 8 * np.sin(np.linspace(0, 70, n)) + np.linspace(0, 30, n), index=idx)


def test_sweep_and_confirm_runs_full_pipeline():
    prices = _spanning_prices()
    r = sweep_and_confirm(prices, "trend_following", "ma_trend", bars=_bars(prices), spec=_SPEC)
    assert {"sweep", "params", "pipeline", "confirmed", "winner_oos_passed"} <= set(r)
    assert set(r["pipeline"]) == {"backtest", "walk_forward", "oos"}     # incl. the fixed holdout
    assert isinstance(r["confirmed"], bool)
    assert "window" in r["params"]


def test_format_confirm_shows_holdout_and_verdict():
    prices = _spanning_prices()
    r = sweep_and_confirm(prices, "trend_following", "ma_trend", bars=_bars(prices), spec=_SPEC)
    text = format_confirm(r, "trend_following", "ma_trend")
    assert "Auto-confirmation through the full pipeline" in text
    assert "VERDICT:" in text
    assert ("CONFIRMED" in text) or ("REJECTED" in text)
