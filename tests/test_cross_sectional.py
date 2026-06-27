"""Tests for cross-sectional (multi-asset ranking) signals and models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.families import create_model
from meridian.features import momentum_scores, rank_signals
from meridian.portfolio import PortfolioResult


def _universe(n=300):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return {
        "A": pd.Series(np.linspace(100, 300, n), index=idx),   # strongest
        "B": pd.Series(np.linspace(100, 150, n), index=idx),   # mild up
        "C": pd.Series(np.full(n, 100.0), index=idx),          # flat
        "D": pd.Series(np.linspace(100, 60, n), index=idx),    # weakest
    }


# --- signal builders ------------------------------------------------------

def test_momentum_scores_is_trailing_return():
    px = _universe()
    sc = momentum_scores(px, lookback=20)
    assert list(sc.columns) == ["A", "B", "C", "D"]
    assert sc["A"].iloc[-1] > sc["D"].iloc[-1]       # A rising, D falling


def test_rank_signals_long_top_short_bottom():
    sc = momentum_scores(_universe(), lookback=20)
    sig = rank_signals(sc, quantile=0.3)
    last = sig.iloc[-1]
    assert last["A"] == 1 and last["D"] == -1        # strongest long, weakest short
    assert set(np.unique(sig.to_numpy())) <= {-1, 0, 1}


def test_rank_signals_long_only_has_no_shorts():
    sc = momentum_scores(_universe(), lookback=20)
    sig = rank_signals(sc, quantile=0.34, long_only=True)
    assert (sig >= 0).all().all()
    assert (sig.iloc[-1] == 1).any()


# --- cross-sectional models -----------------------------------------------

def test_relative_strength_model_signals_and_backtest():
    m = create_model("momentum", "relative_strength")
    assert m.cross_sectional is True
    px = _universe()
    sig = m.signals(px)
    assert sig.iloc[-1]["A"] == 1 and sig.iloc[-1]["D"] == -1
    res = m.backtest(px)
    assert isinstance(res, PortfolioResult)
    assert np.isfinite(res.equity.iloc[-1])
    assert res.meta["family"] == "momentum" and res.meta["model"] == "relative_strength"


def test_dual_momentum_drops_longs_without_positive_absolute_momentum():
    # all names falling: relative rank would still long the "least bad", dual momentum must not
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    px = {
        "A": pd.Series(np.linspace(100, 90, 300), index=idx),   # least bad (top rank)
        "B": pd.Series(np.linspace(100, 70, 300), index=idx),
        "C": pd.Series(np.linspace(100, 50, 300), index=idx),   # worst
    }
    sig = create_model("momentum", "dual_momentum").signals(px).iloc[-1]
    assert sig["A"] != 1          # not bought despite being top-ranked (absolute momentum < 0)
    assert sig["C"] != 1


def test_sector_relative_strength_is_long_only():
    m = create_model("sector_rotation", "relative_strength")
    sig = m.signals(_universe())
    assert (sig >= 0).all().all()
    res = m.backtest(_universe())
    assert isinstance(res, PortfolioResult)
