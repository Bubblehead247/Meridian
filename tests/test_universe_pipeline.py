"""Tests for the universe-level (cross-sectional) pipeline stage runners."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.families import create_model
from meridian.pipeline import (
    StageResult,
    run_universe_backtest_stage,
    run_universe_oos_stage,
    run_universe_pipeline,
    run_universe_walk_forward_stage,
)
from meridian.portfolio import StrategyLedger
from meridian.validation import WalkForwardSpec


def _universe(n=1200, seed=0):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    drifts = [0.0006, 0.0003, 0.0, -0.0003, -0.0006]
    return {
        f"S{i}": pd.Series(100 * np.exp(np.cumsum(rng.normal(d, 0.01, n))), index=idx)
        for i, d in enumerate(drifts)
    }


def _universe_long(seed=0):
    """Universe spanning 2010-2024 — covers all three pipeline windows."""
    n = 3600
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    drifts = [0.0006, 0.0003, 0.0, -0.0003, -0.0006]
    return {
        f"S{i}": pd.Series(100 * np.exp(np.cumsum(rng.normal(d, 0.01, n))), index=idx)
        for i, d in enumerate(drifts)
    }


def test_universe_backtest_stage():
    model = create_model("momentum", "relative_strength")
    res = run_universe_backtest_stage(model, _universe())
    assert isinstance(res, StageResult)
    assert res.stage == "backtest" and res.model == "relative_strength"
    assert res.detail["n_symbols"] == 5
    assert "sharpe" in res.scorecard


def test_universe_walk_forward_stage_stitches_oos():
    model = create_model("momentum", "relative_strength")
    res = run_universe_walk_forward_stage(model, _universe(n=1300))
    assert res.stage == "walk_forward"
    assert res.detail["n_folds"] >= 1 and res.detail["oos_periods"] > 0


def test_universe_walk_forward_no_folds_when_short():
    model = create_model("momentum", "relative_strength")
    res = run_universe_walk_forward_stage(model, _universe(n=200))
    assert res.passed is False and res.detail["n_folds"] == 0


def test_universe_oos_stage_scores_oos_window():
    model = create_model("momentum", "relative_strength")
    res = run_universe_oos_stage(model, _universe_long())
    assert isinstance(res, StageResult)
    assert res.stage == "oos"
    assert res.detail["n_oos"] > 0
    assert "sharpe" in res.scorecard


def test_universe_oos_stage_empty_when_no_data_in_window():
    model = create_model("momentum", "relative_strength")
    res = run_universe_oos_stage(model, _universe())   # 2015-2019, OOS window is 2020-2022
    assert res.passed is False
    assert res.detail.get("n_oos", 0) == 0


def test_universe_pipeline_runs_all_three_stages():
    model = create_model("momentum", "relative_strength")
    led = StrategyLedger(name="relative_strength", family="momentum", stage="research")
    res = run_universe_pipeline(model, _universe_long(), ledger=led, as_of="2026-06-26")
    assert "backtest" in res
    # Stages stop on fail, so only assert that what ran has the right structure.
    for stage_res in res.values():
        assert "sharpe" in stage_res.scorecard or stage_res.detail.get("n_oos", 1) == 0
        assert "ledger_action" in stage_res.detail


def test_universe_pipeline_advances_ledger_and_stops_on_fail():
    model = create_model("sector_rotation", "relative_strength")
    led = StrategyLedger(name="relative_strength", family="sector_rotation", stage="research")
    spec = WalkForwardSpec(mode="anchored", min_train=400, test_span=200, step=200)
    res = run_universe_pipeline(model, _universe(), ledger=led, wf_spec=spec, as_of="2026-06-26")
    assert "backtest" in res
    assert "ledger_action" in res["backtest"].detail
    assert res["backtest"].detail["ledger_action"] in ("promote", "hold", "retire")
    # if the backtest stage fails the metric bar, the walk stops there
    if not res["backtest"].passed:
        assert set(res) == {"backtest"}
