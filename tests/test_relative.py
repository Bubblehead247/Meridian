"""Tests for cross-sectional relativization (relative-value MR)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.features import cross_sectional_demean


def _prices(n=20):
    idx = pd.RangeIndex(n)
    return {
        "A": pd.Series(100.0 + np.arange(n), index=idx),   # highest
        "B": pd.Series(50.0 + np.arange(n), index=idx),
        "C": pd.Series(20.0 + np.arange(n), index=idx),    # lowest
    }


def test_relative_values_are_zero_mean_each_bar():
    rel = cross_sectional_demean(_prices())
    df = pd.DataFrame(rel)
    # each bar's relative values sum/mean to ~0 across the names present
    assert np.allclose(df.mean(axis=1).to_numpy(), 0.0, atol=1e-12)


def test_rich_name_positive_cheap_name_negative():
    rel = cross_sectional_demean(_prices())
    # A is always the most expensive -> positive relative; C the cheapest -> negative
    assert (rel["A"] > 0).all()
    assert (rel["C"] < 0).all()


def test_missing_prices_excluded_from_mean_and_propagate_nan():
    px = _prices(10)
    px["C"] = px["C"].copy()
    px["C"].iloc[:4] = np.nan          # C "not listed" early
    rel = cross_sectional_demean(px)
    # C is NaN where it had no price
    assert rel["C"].iloc[:4].isna().all()
    # On those bars the mean is over A and B only -> A and B still sum to ~0
    early = pd.DataFrame({"A": rel["A"], "B": rel["B"]}).iloc[:4]
    assert np.allclose(early.mean(axis=1).to_numpy(), 0.0, atol=1e-12)


def test_nonpositive_prices_become_nan():
    px = {"A": pd.Series([10.0, 10.0]), "B": pd.Series([0.0, 5.0])}
    rel = cross_sectional_demean(px)
    assert np.isnan(rel["B"].iloc[0])   # price 0 -> NaN, excluded from the mean
