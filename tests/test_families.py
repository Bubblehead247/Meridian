"""Tests for the strategy-family abstraction + registry (PLAN.md §11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.families import (
    EstimatorModel,
    Model,
    StrategyFamily,
    create_model,
    get_model,
    list_families,
    list_models,
    register_model,
)
from meridian.signals import run_backtest


def _reverting(n=400, seed=0):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(100 + 5 * np.sin(np.linspace(0, 40, n)), index=idx)


# --- registry -------------------------------------------------------------

def test_zscore_reversion_is_registered():
    assert "zscore_reversion" in list_models("mean_reversion")
    assert "mean_reversion" in list_families()
    cls = get_model("mean_reversion", "zscore_reversion")
    assert issubclass(cls, Model)


def test_create_model_sets_identity():
    m = create_model("mean_reversion", "zscore_reversion")
    assert m.family == "mean_reversion" and m.name == "zscore_reversion"


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="Unknown model"):
        get_model("mean_reversion", "nope")


def test_duplicate_registration_raises():
    """Registers into the global registry, so it must clean up after itself.

    It did not, and left a permanent ``testfam/dup`` entry behind: every later
    test in the same run saw 9 families and 53 models instead of 8 and 52. That
    was invisible until something counted them — the W9 doc-drift tests, which
    passed alone and failed in the full suite.
    """
    from meridian.families import registry

    before = dict(registry._REGISTRY)
    try:
        @register_model("testfam", "dup")
        class _A(EstimatorModel):
            pass

        with pytest.raises(ValueError, match="already registered"):
            @register_model("testfam", "dup")
            class _B(EstimatorModel):
                pass
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(before)


# --- model behaviour ------------------------------------------------------

def test_model_signals_are_positions():
    m = create_model("mean_reversion", "zscore_reversion")
    sig = m.signals(_reverting())
    assert set(np.unique(sig.dropna())) <= {-1, 0, 1}


def test_model_backtest_matches_run_backtest_pipeline():
    """A model reuses the shared pipeline, so it must equal run_backtest of the same config."""
    prices = _reverting()
    m = create_model("mean_reversion", "zscore_reversion")
    mine = m.backtest(prices, cost_bps=1.0, allow_shorts=True)
    ref = run_backtest(prices, "sma", "zscore", window=20, cost_bps=1.0)
    pd.testing.assert_series_equal(mine.returns, ref.returns)
    assert mine.meta["family"] == "mean_reversion" and mine.meta["model"] == "zscore_reversion"


def test_model_backtest_holds_short_signals_flat_by_default():
    """The fund trades long only; a backtest must not book shorts it can't take."""
    prices = _reverting()
    m = create_model("mean_reversion", "zscore_reversion")
    assert (m.signals(prices) < 0).any(), "fixture must produce short signals"
    res = m.backtest(prices, cost_bps=1.0)
    assert (res.positions >= 0).all()
    both = m.backtest(prices, cost_bps=1.0, allow_shorts=True)
    assert (both.positions < 0).any()


def test_cross_sectional_backtest_gives_the_longs_the_whole_sleeve():
    """Shorts are dropped before sizing, as live splits a sleeve over its longs."""
    import numpy as np
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    universe = {f"S{i}": pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005 * i, 0.01, 300))),
                                   index=idx) for i in range(6)}
    m = create_model("momentum", "relative_strength")
    assert (m.filtered_signals(universe) < 0).any().any(), "model must emit shorts"
    res = m.backtest(universe, cost_bps=0.0)
    w = res.weights.iloc[-50:]
    assert (w >= 0).all().all()
    assert w.sum(axis=1).round(9).eq(1.0).all()


def test_instance_overrides_window():
    m = create_model("mean_reversion", "zscore_reversion", window=50)
    assert m.window == 50


# --- StrategyFamily -------------------------------------------------------

def test_strategy_family_lists_and_creates():
    fam = StrategyFamily("mean_reversion")
    assert "zscore_reversion" in fam.model_names()
    models = fam.create_all()
    assert set(models) == set(fam.model_names())
    assert all(isinstance(m, Model) for m in models.values())
