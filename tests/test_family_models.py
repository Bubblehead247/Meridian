"""Tests for the concrete family models + the StrategyFamily permission hook."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.families import StrategyFamily, all_models, create_model, list_families
from meridian.families.base import BacktestResult
from meridian.regimes import RegimeLabel


def _prices(n=400, trend="flat"):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    if trend == "up":
        base = np.linspace(100, 200, n)
    else:
        base = 100 + 5 * np.sin(np.linspace(0, 40, n))
    return pd.Series(base, index=idx)


def _bars(prices):
    return pd.DataFrame(
        {
            "open": prices.shift(1).fillna(prices.iloc[0]),
            "high": prices + 1, "low": prices - 1, "close": prices,
            "volume": pd.Series(np.linspace(1e6, 2e6, len(prices)), index=prices.index),
        }
    )


# --- every registered model is valid --------------------------------------

def test_all_models_produce_valid_positions_and_backtest():
    prices = _prices()
    bars = _bars(prices)
    assert len(all_models()) >= 19
    for (family, name) in all_models():
        m = create_model(family, name)
        if getattr(m, "cross_sectional", False):
            continue                                  # multi-asset; tested separately
        sig = m.signals(prices, bars=bars)
        assert sig.index.equals(prices.index)
        assert set(np.unique(sig.dropna())) <= {-1, 0, 1}, f"{family}/{name}"
        res = m.backtest(prices, bars=bars)
        assert isinstance(res, BacktestResult)
        assert res.meta["family"] == family and res.meta["model"] == name


# --- archetypes behave distinctly -----------------------------------------

def test_trend_model_goes_long_in_uptrend():
    m = create_model("trend_following", "ma_trend")
    sig = m.signals(_prices(trend="up"))
    assert sig.iloc[-1] == 1                       # price above its MA in a clean uptrend


def test_breakout_holds_until_reversed():
    m = create_model("breakouts", "donchian_breakout")
    sig = m.signals(_prices(trend="up"))
    assert sig.iloc[-1] == 1                       # new highs -> long, held


def test_long_term_etf_is_long_or_flat_only():
    m = create_model("long_term_etf", "above_200ma")
    sig = m.signals(_prices(n=400, trend="up"))
    assert set(np.unique(sig)) <= {0, 1}           # never short


def test_reversion_differs_from_trend_on_same_prices():
    prices = _prices(trend="up")
    rev = create_model("mean_reversion", "zscore_reversion").signals(prices)
    trend = create_model("trend_following", "ma_trend").signals(prices)
    assert not rev.equals(trend)


# --- StrategyFamily permission hook ---------------------------------------

def test_strategy_family_permitted_uses_matrix():
    fam = StrategyFamily("trend_following")
    assert fam.permitted(RegimeLabel("bull", "low", "neutral")) is True
    assert fam.permitted(RegimeLabel("bear", "low", "neutral")) is False


def test_strategy_family_gate_flattens_restricted_bars():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    positions = pd.Series([1, 1, 1], index=idx)
    frame = pd.DataFrame(
        {"trend": ["bull", "bear", "bull"], "volatility": "low", "breadth": "neutral"},
        index=idx,
    )
    gated = StrategyFamily("trend_following").gate(positions, frame)
    assert gated.tolist() == [1, 0, 1]             # flat on the bear bar


def test_nr7_breakout_needs_bars_and_breaks_out():
    m = create_model("breakouts", "nr7_breakout")
    prices = _prices(trend="up")
    assert (m.signals(prices) == 0).all()                       # no bars -> flat
    sig = m.signals(prices, bars=_bars(prices))
    assert set(np.unique(sig)) <= {-1, 0, 1}
    assert (sig != 0).any()                                     # some breakout taken


def test_channel_breakout_long_only_in_uptrend():
    m = create_model("trend_following", "channel_breakout")
    sig = m.signals(_prices(n=400, trend="up"))                 # close-only, rising
    assert sig.iloc[-1] == 1                                    # breakout above MA -> long
    assert (sig >= 0).all()                                     # never shorts a clean uptrend


def test_active_families_each_have_a_model():
    expected = {
        "mean_reversion", "trend_following", "momentum", "breakouts",
        "pullback_continuation", "sector_rotation", "long_term_etf",
    }
    assert expected <= set(list_families())
