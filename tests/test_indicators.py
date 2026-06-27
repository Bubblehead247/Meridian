"""Tests for the single-series technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.families import create_model
from meridian.features import overnight_gap, rsi, volume_ratio


def _series(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="B"))


# --- RSI ------------------------------------------------------------------

def test_rsi_is_bounded_0_to_100():
    rng = np.random.default_rng(0)
    r = rsi(_series(100 + np.cumsum(rng.normal(0, 1, 200)))).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_rsi_high_on_rally_low_on_decline():
    up = rsi(_series(np.linspace(100, 200, 100))).iloc[-1]
    down = rsi(_series(np.linspace(200, 100, 100))).iloc[-1]
    assert up > 70          # only gains -> overbought
    assert down < 30        # only losses -> oversold


# --- overnight gap --------------------------------------------------------

def test_overnight_gap_value():
    bars = pd.DataFrame(
        {"open": [100, 110, 99], "close": [100, 100, 100]},
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )
    gap = overnight_gap(bars)
    assert np.isnan(gap.iloc[0])           # no prior close
    assert abs(gap.iloc[1] - 0.10) < 1e-9  # opened 10% above prior close
    assert abs(gap.iloc[2] + 0.01) < 1e-9  # opened 1% below prior close


# --- volume ratio ---------------------------------------------------------

def test_volume_ratio_around_one_and_spikes():
    vol = _series(np.r_[np.full(30, 1000.0), [5000.0]])
    vr = volume_ratio(vol, window=20)
    assert abs(vr.iloc[29] - 1.0) < 1e-9   # steady volume -> ~1
    assert vr.iloc[-1] > 3                 # the spike stands out


# --- indicator-driven models behave as intended ---------------------------

def test_rsi_exhaustion_longs_when_oversold():
    m = create_model("mean_reversion", "rsi_exhaustion")
    assert m.signals(_series(np.linspace(200, 100, 100))).iloc[-1] == 1   # oversold -> long
    assert m.signals(_series(np.linspace(100, 200, 100))).iloc[-1] == -1  # overbought -> short


def test_gap_fill_needs_bars():
    m = create_model("mean_reversion", "gap_fill")
    prices = _series(np.linspace(100, 110, 20))
    assert (m.signals(prices) == 0).all()                # no bars -> no positions
    bars = pd.DataFrame(
        {"open": prices * 1.03, "close": prices}, index=prices.index  # persistent 3% gap-up
    )
    assert (m.signals(prices, bars=bars) == -1).any()    # fade the gap-up -> short
