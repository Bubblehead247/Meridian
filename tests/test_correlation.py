"""Tests for inter-sleeve correlation monitoring (PLAN.md §3 portfolio/correlation)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from meridian.portfolio import (
    StrategyLedger,
    average_correlation,
    build_correlation_matrix,
    flag_high_correlation,
    pairwise_correlations,
    update_ledger_correlations,
)


def _sleeves(n=100, seed=0):
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    base = pd.Series(rng.normal(0, 0.01, n), index=idx)
    return {
        "a": base,
        "twin": base.copy(),                       # ρ = +1 with a
        "inverse": -base,                          # ρ = -1 with a
        "noise": pd.Series(rng.normal(0, 0.01, n), index=idx),
    }


def test_build_matrix_captures_relationships():
    m = build_correlation_matrix(_sleeves())
    assert math.isclose(m.at["a", "a"], 1.0)
    assert math.isclose(m.at["a", "twin"], 1.0)
    assert math.isclose(m.at["a", "inverse"], -1.0)
    assert abs(m.at["a", "noise"]) < 0.5


def test_build_matrix_empty_input():
    assert build_correlation_matrix({}).empty


def test_pairwise_excludes_self():
    m = build_correlation_matrix(_sleeves())
    pw = pairwise_correlations("a", m)
    assert "a" not in pw
    assert math.isclose(pw["twin"], 1.0)


def test_average_correlation_off_diagonal():
    # two perfectly correlated sleeves -> off-diagonal mean = 1.0
    idx = pd.date_range("2022-01-01", periods=20, freq="B")
    s = pd.Series(np.linspace(0, 1, 20), index=idx)
    m = build_correlation_matrix({"x": s, "y": s})
    assert math.isclose(average_correlation(m), 1.0)
    assert math.isnan(average_correlation(build_correlation_matrix({"only": s})))


def test_flag_high_correlation_pairs():
    m = build_correlation_matrix(_sleeves())
    flagged = flag_high_correlation(m, threshold=0.9)
    assert ("a", "twin") in flagged
    assert ("a", "inverse") not in flagged          # -1 is not > 0.9


def test_update_ledger_correlations_writes_each_ledger():
    sleeves = _sleeves()
    ledgers = [StrategyLedger(name=n, family=n) for n in sleeves]
    matrix = update_ledger_correlations(ledgers, sleeves)
    by_name = {x.name: x for x in ledgers}
    assert math.isclose(by_name["a"].correlations["twin"], 1.0)
    assert "a" not in by_name["a"].correlations        # self excluded
    assert not matrix.empty
