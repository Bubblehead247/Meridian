"""Tests for the pipeline stage runners (PLAN.md §6 steps 4-6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.families import create_model
from meridian.pipeline import (
    OOSGuard,
    GraduationCriteria,
    StageResult,
    advance,
    check_stale,
    compare_oos_to_is,
    criteria_for_family,
    passes_metric_bar,
    run_backtest_stage,
    run_oos_stage,
    run_pipeline,
    run_stage,
    run_walk_forward_stage,
)
from meridian.portfolio import StrategyLedger
from meridian.portfolio.allocation import SLEEVE_ALLOCATIONS
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


def test_oos_stage_without_guard_has_no_run_count():
    prices = _prices(n=3200, start="2010-01-01")
    res = run_oos_stage(_model(), prices)
    assert "oos_run_count" not in res.detail


def test_oos_guard_counts_repeated_runs(tmp_path):
    guard = OOSGuard(root=tmp_path / "oos_runs")
    prices = _prices(n=3200, start="2010-01-01")
    res1 = run_oos_stage(_model(), prices, guard=guard, symbol="SPY")
    res2 = run_oos_stage(_model(), prices, guard=guard, symbol="SPY")
    assert res1.detail["oos_run_count"] == 1
    assert res2.detail["oos_run_count"] == 2


def test_oos_guard_keys_are_isolated_by_symbol(tmp_path):
    guard = OOSGuard(root=tmp_path / "oos_runs")
    prices = _prices(n=3200, start="2010-01-01")
    run_oos_stage(_model(), prices, guard=guard, symbol="SPY")
    res = run_oos_stage(_model(), prices, guard=guard, symbol="QQQ")
    assert res.detail["oos_run_count"] == 1  # different symbol, fresh counter


def test_run_pipeline_forwards_guard_to_oos_stage(tmp_path):
    guard = OOSGuard(root=tmp_path / "oos_runs")
    prices = _prices(n=3200, start="2010-01-01")
    results = run_pipeline(
        _model(), prices, stages=("oos",), stop_on_fail=False,
        oos_guard=guard, symbol="SPY",
    )
    assert results["oos"].detail["oos_run_count"] == 1
    results2 = run_pipeline(
        _model(), prices, stages=("oos",), stop_on_fail=False,
        oos_guard=guard, symbol="SPY",
    )
    assert results2["oos"].detail["oos_run_count"] == 2


def test_run_pipeline_tracks_oos_runs_by_default(monkeypatch, tmp_path):
    # Regression for P1-B: OOSGuard existed but nothing ever instantiated it, so the
    # "final" holdout could be re-run indefinitely with no record. run_pipeline must
    # now track by default, with no explicit oos_guard/symbol required from the caller.
    from meridian.pipeline import oos_guard as oos_guard_mod

    monkeypatch.setattr(oos_guard_mod, "DEFAULT_GUARD_DIR", tmp_path / "oos_runs")
    prices = _prices(n=3200, start="2010-01-01")
    ledger = StrategyLedger(name="zscore_reversion", family="mean_reversion", stage="research")
    results = run_pipeline(_model(), prices, ledger=ledger, stages=("oos",), stop_on_fail=False)
    assert results["oos"].detail["oos_run_count"] == 1
    results2 = run_pipeline(_model(), prices, ledger=ledger, stages=("oos",), stop_on_fail=False)
    assert results2["oos"].detail["oos_run_count"] == 2


def test_run_pipeline_track_oos_runs_false_disables_counting(tmp_path):
    guard_would_write_here = tmp_path / "oos_runs"
    prices = _prices(n=3200, start="2010-01-01")
    results = run_pipeline(
        _model(), prices, stages=("oos",), stop_on_fail=False, track_oos_runs=False,
    )
    assert "oos_run_count" not in results["oos"].detail
    assert not guard_would_write_here.exists()


# --- graduation criteria versioning (P1-C) --------------------------------

def test_advance_binds_criteria_version_on_first_call():
    led = StrategyLedger(name="zscore_reversion", family="mean_reversion", stage="research")
    assert led.criteria_version is None
    criteria = criteria_for_family("mean_reversion")
    advance(led, {"sharpe": -1.0}, criteria=criteria)
    assert led.criteria_version == criteria.version


def test_advance_rejects_a_different_criteria_version_without_override():
    from meridian.pipeline.graduation import CriteriaVersionMismatch

    led = StrategyLedger(name="zscore_reversion", family="mean_reversion", stage="research")
    v1 = GraduationCriteria(min_sharpe=0.35, version="1.0.0")
    v2 = GraduationCriteria(min_sharpe=0.50, version="2.0.0")
    advance(led, {"sharpe": -1.0}, criteria=v1)
    with pytest.raises(CriteriaVersionMismatch):
        advance(led, {"sharpe": -1.0}, criteria=v2)
    assert led.criteria_version == "1.0.0"  # rejected call must not have rebound it


def test_advance_accepts_a_version_override_with_reason_and_logs_it(monkeypatch, tmp_path):
    from meridian.experiments import run_log as run_log_mod

    monkeypatch.setattr(run_log_mod, "DEFAULT_RUN_LOG", tmp_path / "run_log.jsonl")
    led = StrategyLedger(name="zscore_reversion", family="mean_reversion", stage="research")
    v1 = GraduationCriteria(min_sharpe=0.35, version="1.0.0")
    v2 = GraduationCriteria(min_sharpe=0.50, version="2.0.0")
    advance(led, {"sharpe": -1.0}, criteria=v1)
    advance(
        led, {"sharpe": -1.0}, criteria=v2,
        allow_criteria_override=True, override_reason="deliberate threshold bump for a retest",
    )
    assert led.criteria_version == "2.0.0"
    logged = run_log_mod.read_runs(tmp_path / "run_log.jsonl")
    assert len(logged) == 1
    assert logged[0].kind == "graduation_criteria_override"
    assert logged[0].meta["from_version"] == "1.0.0"
    assert logged[0].meta["to_version"] == "2.0.0"
    assert "deliberate" in logged[0].meta["reason"]


def test_advance_override_without_reason_raises():
    led = StrategyLedger(name="zscore_reversion", family="mean_reversion", stage="research")
    v1 = GraduationCriteria(version="1.0.0")
    v2 = GraduationCriteria(version="2.0.0")
    advance(led, {"sharpe": -1.0}, criteria=v1)
    with pytest.raises(ValueError):
        advance(led, {"sharpe": -1.0}, criteria=v2, allow_criteria_override=True)


def test_load_criteria_config_missing_file_falls_back():
    from meridian.pipeline.graduation import load_criteria_config

    cfg = load_criteria_config("does/not/exist.yaml")
    assert cfg["version"] == "1.0.0-fallback"
    assert cfg["min_sharpe"] == 0.35


# --- stage staleness (P1-C) -------------------------------------------------

def test_check_stale_flags_a_pre_live_stage_past_its_dwell_limit():
    led = StrategyLedger(name="x", family="mean_reversion", stage="research")
    led.stage_entered = "2020-01-01"
    assert check_stale(led, as_of="2026-01-01") is True


def test_check_stale_false_within_the_dwell_limit():
    led = StrategyLedger(name="x", family="mean_reversion", stage="research")
    led.stage_entered = "2026-01-01"
    assert check_stale(led, as_of="2026-01-10") is False


def test_check_stale_false_for_live_stages():
    led = StrategyLedger(name="x", family="mean_reversion", stage="elite")
    led.stage_entered = "2010-01-01"
    assert check_stale(led, as_of="2026-01-01") is False


def test_compare_oos_to_is_flags_degradation():
    assert compare_oos_to_is({"sharpe": 0.1}, {"sharpe": 1.0})["degraded"] is True
    assert compare_oos_to_is({"sharpe": 0.9}, {"sharpe": 1.0})["degraded"] is False
    # no in-sample edge to degrade from
    assert compare_oos_to_is({"sharpe": -0.5}, {"sharpe": -0.2})["degraded"] is False


# --- graduation criteria: family/sleeve key resolution --------------------

def test_criteria_for_family_direct_key_match():
    crit = criteria_for_family("mean_reversion")
    assert crit.allocation_weight == SLEEVE_ALLOCATIONS["mean_reversion"]


def test_criteria_for_family_remapped_families_get_their_sleeve_criteria():
    # breakouts now has its own sleeve (no longer remapped to trend_following)
    breakouts = criteria_for_family("breakouts")
    assert breakouts.allocation_weight == SLEEVE_ALLOCATIONS["breakouts"]
    # volatility funds off the experimental_research sleeve
    volatility = criteria_for_family("volatility")
    assert volatility.allocation_weight == SLEEVE_ALLOCATIONS["experimental_research"]


def test_criteria_for_family_unknown_falls_back_to_default():
    crit = criteria_for_family("not_a_real_family")
    assert crit.allocation_weight is None


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
