"""Tests for the graduation state machine (PLAN.md §6)."""

from __future__ import annotations

from meridian.pipeline import (
    GraduationCriteria,
    advance,
    check_promotion,
    check_retirement,
    evaluate,
    next_stage,
)
from meridian.portfolio import StrategyLedger

GOOD = {"sharpe": 1.0, "trade_expectancy": 0.01, "max_drawdown": -0.05}
WEAK = {"sharpe": 0.1, "trade_expectancy": 0.0, "max_drawdown": -0.05}


def _led(stage="research", entered="2026-01-01", **kw):
    return StrategyLedger(name="m", family="mean_reversion", stage=stage,
                          stage_entered=entered, **kw)


# --- ordering -------------------------------------------------------------

def test_next_stage_order_and_terminals():
    assert next_stage("research") == "backtest"
    assert next_stage("paper") == "pilot"
    assert next_stage("core") == "elite"
    assert next_stage("elite") is None        # top
    assert next_stage("retired") is None      # terminal


# --- promotion: metric gate (early stages, no time gate) ------------------

def test_promote_early_stage_on_good_scorecard():
    assert check_promotion(_led("research"), GOOD, as_of="2026-01-01") == "backtest"


def test_no_promote_when_metrics_below_bar():
    assert check_promotion(_led("research"), WEAK, as_of="2026-01-01") is None


def test_promotion_fails_closed_on_missing_or_nan_sharpe():
    assert check_promotion(_led("research"), {"sharpe": float("nan")}, as_of="2026-01-01") is None
    assert check_promotion(_led("research"), {}, as_of="2026-01-01") is None


# --- promotion: time gate (live stages) -----------------------------------

def test_paper_needs_30_days_before_pilot():
    led = _led("paper", entered="2026-01-01")
    assert check_promotion(led, GOOD, as_of="2026-01-10") is None     # 9 days
    assert check_promotion(led, GOOD, as_of="2026-03-01") == "pilot"  # 59 days


def test_proven_needs_180_days_before_core():
    led = _led("proven", entered="2026-01-01")
    assert check_promotion(led, GOOD, as_of="2026-04-01") is None     # ~90 days
    assert check_promotion(led, GOOD, as_of="2026-09-01") == "core"   # ~243 days


# --- retirement -----------------------------------------------------------

def test_retire_on_negative_expectancy():
    assert check_retirement(_led("core"), {"trade_expectancy": -0.01, "max_drawdown": -0.05})


def test_retire_on_drawdown_breach():
    # Pin explicit criteria so the test doesn't break when the global default changes.
    strict = GraduationCriteria(max_drawdown_limit=0.20)
    assert check_retirement(_led("core"), {"trade_expectancy": 0.01, "max_drawdown": -0.30}, criteria=strict)
    assert check_retirement(_led("core", drawdown_cur=-0.25), GOOD, criteria=strict)


def test_retirement_blocks_promotion():
    led = _led("proven", entered="2025-01-01")     # plenty of days
    sick = {"sharpe": 1.0, "trade_expectancy": -0.01, "max_drawdown": -0.05}
    assert check_promotion(led, sick, as_of="2026-06-01") is None


# --- evaluate / advance ---------------------------------------------------

def test_evaluate_three_actions():
    assert evaluate(_led("research"), GOOD, as_of="2026-01-01") == ("promote", "backtest")
    assert evaluate(_led("research"), WEAK, as_of="2026-01-01") == ("hold", None)
    assert evaluate(_led("core"), {"trade_expectancy": -0.01}) == ("retire", "retired")


def test_advance_applies_stage_transition():
    led = _led("research")
    action = advance(led, GOOD, as_of="2026-02-01")
    assert action == "promote"
    assert led.stage == "backtest" and led.stage_entered == "2026-02-01"

    strict = GraduationCriteria(max_drawdown_limit=0.20)
    led2 = _led("core", drawdown_cur=-0.30)
    assert advance(led2, GOOD, as_of="2026-02-01", criteria=strict) == "retire"
    assert led2.stage == "retired"


def test_custom_criteria_raise_the_bar():
    strict = GraduationCriteria(min_sharpe=2.0)
    assert check_promotion(_led("research"), GOOD, as_of="2026-01-01", criteria=strict) is None
