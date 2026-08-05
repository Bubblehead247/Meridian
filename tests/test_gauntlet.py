"""Tests for the strategy gauntlet (ranked scoreboard)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.experiments.gauntlet import (
    format_gauntlet,
    gauntlet_single,
    gauntlet_universe,
)

_COLS = [
    "family", "model", "passed", "sharpe", "cagr", "total_return",
    "max_drawdown", "n_trades", "trades_per_year", "fill_realism",
]


def _prices(n=500):
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(100 + 5 * np.sin(np.linspace(0, 40, n)) + np.linspace(0, 15, n), index=idx)


def _bars(prices):
    return pd.DataFrame(
        {"open": prices.shift(1).fillna(prices.iloc[0]), "high": prices + 1,
         "low": prices - 1, "close": prices,
         "volume": np.linspace(1e6, 2e6, len(prices))}
    )


def _universe(n=400):
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    return {f"S{i}": pd.Series(100 * np.exp(np.cumsum(rng.normal(d, 0.01, n))), index=idx)
            for i, d in enumerate([0.0006, 0.0002, -0.0002, -0.0006])}


def test_gauntlet_single_ranks_all_single_asset_models():
    prices = _prices()
    df = gauntlet_single(prices, bars=_bars(prices))
    assert list(df.columns) == _COLS
    assert len(df) >= 14                                  # all single-asset models
    assert (df["model"] != "relative_strength").all()     # no cross-sectional models here
    sharpes = df["sharpe"].dropna().to_numpy()
    assert (np.diff(sharpes) <= 1e-9).all()               # sorted best-Sharpe first


def test_gauntlet_universe_ranks_cross_sectional_models():
    df = gauntlet_universe(_universe())
    assert list(df.columns) == _COLS
    assert len(df) >= 3
    # Verify no single-asset models bled into the universe gauntlet
    from meridian.families import create_model
    for _, row in df.iterrows():
        m = create_model(row["family"], row["model"])
        assert getattr(m, "cross_sectional", False), f"{row['model']} is not a CS model"


def test_format_gauntlet_renders_a_table():
    df = gauntlet_single(_prices(), bars=None)
    text = format_gauntlet(df)
    assert "model" in text and "sharpe" in text
    assert format_gauntlet(df.iloc[0:0]) == "(no models ran)"
