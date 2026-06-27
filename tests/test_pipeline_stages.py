"""Tests for the pipeline stage runners (PLAN.md §6 steps 4-6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.families import create_model
from meridian.pipeline import (
    StageResult,
    advance,
    compare_oos_to_is,
    passes_metric_bar,
    run_backtest_stage,
    run_oos_stage,
    run_stage,
    run_walk_forward_stage,
)
from meridian.portfolio import StrategyLedger
from meridian.validation import WalkForwardSpec


def _model():
    return create_model("mean_reversion", "zscore_reversion")


def _prices(n=1000, start="2018-01-01"):
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.Series(100 + 5 * np.sin(np.linspace(0, 60, n)), index=idx)


# --- backtest stage -------------------------------------------------------

def test_backtest_stage_shape_and_pass_flag():
    res = run_backtest_stage(_model(), _prices())
    assert isinstance(res, StageResult)
    assert res.stage == "backtest" and res.model == "zscore_reversion"
    assert "sharpe" in res.scorecard
    assert res.passed == passes_metric_bar(res.scorecard)   # pass flag is the metric gate
    assert res.detail["n_periods"] > 0


# --- walk-forward stage ---------------------------------------------------

def test_walk_forward_stage_stitches_oos_folds():
    res = run_walk_forward_stage(_model(), _prices(n=1200))
    assert res.stage == "walk_forward"
    assert res.detail["n_folds"] >= 1
    assert res.detail["oos_periods"] > 0
    assert isinstance(res.passed, bool)


def test_walk_forward_no_folds_when_too_short():
    res = run_walk_forward_stage(_model(), _prices(n=100))   # < min_train
    assert res.passed is False and res.detail["n_folds"] == 0


def test_walk_forward_custom_spec():
    spec = WalkForwardSpec(mode="anchored", min_train=300, test_span=100, step=100)
    res = run_walk_forward_stage(_model(), _prices(n=800), spec=spec)
    assert res.detail["n_folds"] >= 1


# --- OOS stage ------------------------------------------------------------

def test_oos_stage_runs_on_holdout_window():
    # spans the default in-sample (2010-2019) and OOS (2020-2022) windows
    prices = _prices(n=3200, start="2010-01-01")
    res = run_oos_stage(_model(), prices)
    assert res.stage == "oos"
    assert res.detail["n_oos"] > 0
    assert "degraded" in res.detail and isinstance(res.passed, bool)


def test_compare_oos_to_is_flags_degradation():
    assert compare_oos_to_is({"sharpe": 0.1}, {"sharpe": 1.0})["degraded"] is True
    assert compare_oos_to_is({"sharpe": 0.9}, {"sharpe": 1.0})["degraded"] is False
    # no in-sample edge to degrade from
    assert compare_oos_to_is({"sharpe": -0.5}, {"sharpe": -0.2})["degraded"] is False


# --- metric gate + dispatcher + graduation integration --------------------

def test_passes_metric_bar_gate():
    assert passes_metric_bar({"sharpe": 1.0, "trade_expectancy": 0.01, "max_drawdown": -0.05})
    assert not passes_metric_bar({"sharpe": 0.1})
    assert not passes_metric_bar({"sharpe": float("nan")})


def test_run_stage_dispatch_and_unknown():
    res = run_stage("backtest", _model(), _prices())
    assert res.stage == "backtest"
    with pytest.raises(KeyError, match="unknown stage"):
        run_stage("bogus", _model(), _prices())


def test_stage_result_feeds_graduation_advance():
    """The stage scorecard drives the graduation state machine (closes the loop)."""
    res = run_backtest_stage(_model(), _prices())
    led = StrategyLedger(name="zscore_reversion", family="mean_reversion", stage="research")
    action = advance(led, res.scorecard, as_of="2026-06-26")
    assert action in ("promote", "hold", "retire")
    # a passing scorecard promotes research -> backtest; a failing one holds/retires
    if res.passed:
        assert led.stage == "backtest"
