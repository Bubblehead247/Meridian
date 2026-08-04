"""Tests for the risk budget: heat, contributions, suspension (PLAN.md §8)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from meridian.portfolio import (
    RiskLimits,
    StrategyLedger,
    allocate_heat_budget,
    apply_risk_contributions,
    check_suspension,
    compute_portfolio_heat,
    marginal_risk_contributions,
    portfolio_heat_breached,
    risk_contributions,
    strategy_heat,
)
from meridian.portfolio.risk_budget import position_risk, position_weight, sector_exposures

ACCOUNT = 10_000.0


def _pos(entry, stop=None, size=10.0, **kw):
    p = {"symbol": kw.get("symbol", "X"), "entry_price": entry, "size": size}
    if stop is not None:
        p["stop_price"] = stop
    p.update(kw)
    return p


def _led(name="mr", cap=1_500.0, positions=None, dd=0.0):
    return StrategyLedger(
        name=name, family=name, capital_alloc=cap,
        open_positions=positions or [], drawdown_cur=dd,
    )


# --- position-level -------------------------------------------------------

def test_position_risk_and_no_stop_is_zero():
    assert math.isclose(position_risk(_pos(100, 95, size=10), ACCOUNT), 0.005)  # 5*10/10000
    assert position_risk(_pos(100, size=10), ACCOUNT) == 0.0                    # no stop
    assert position_risk(_pos(100, 95), 0.0) == 0.0                            # no equity


def test_position_weight():
    assert math.isclose(position_weight(_pos(100, size=10), ACCOUNT), 0.10)     # 1000/10000


# --- strategy / portfolio heat -------------------------------------------

def test_strategy_and_portfolio_heat_sum():
    a = _led("a", positions=[_pos(100, 95, 10), _pos(50, 48, 20)])  # 0.005 + 0.004 = 0.009
    b = _led("b", positions=[_pos(100, 90, 10)])                    # 0.010
    assert math.isclose(strategy_heat(a, ACCOUNT), 0.009)
    assert math.isclose(compute_portfolio_heat([a, b], ACCOUNT), 0.019)


def test_heat_budget_is_capital_share_of_cap():
    led = _led(cap=1_500.0)        # 15% of 10k * 5% cap = 0.0075
    assert math.isclose(allocate_heat_budget([led], ACCOUNT)["mr"], 0.0075)


def test_risk_contributions_sum_to_one_and_write_back():
    a = _led("a", positions=[_pos(100, 95, 10)])   # 0.005
    b = _led("b", positions=[_pos(100, 90, 10)])   # 0.010
    rc = risk_contributions([a, b], ACCOUNT)
    assert math.isclose(rc["a"] + rc["b"], 1.0)
    assert rc["b"] > rc["a"]
    apply_risk_contributions([a, b], ACCOUNT)
    assert math.isclose(a.risk_contribution, rc["a"])


def test_risk_contributions_zero_when_no_heat():
    a, b = _led("a"), _led("b")
    assert risk_contributions([a, b], ACCOUNT) == {"a": 0.0, "b": 0.0}


# --- suspension -----------------------------------------------------------

def test_suspend_on_drawdown_breach():
    led = _led(dd=-0.15)                          # past the 10% default limit
    suspend, reasons = check_suspension(led, ACCOUNT)
    assert suspend and "drawdown" in reasons


def test_suspend_on_heat_over_budget():
    # small capital (tiny budget) but a large at-risk position
    led = _led(cap=500.0, positions=[_pos(100, 90, 100)])   # heat 0.1 >> budget 0.0025
    suspend, reasons = check_suspension(led, ACCOUNT)
    assert suspend and "heat" in reasons


def test_no_suspension_when_within_limits():
    led = _led(cap=5_000.0, positions=[_pos(100, 99, 10)], dd=-0.02)  # heat 0.001, budget 0.025
    assert check_suspension(led, ACCOUNT) == (False, [])


def test_portfolio_heat_breached_against_cap():
    hot = [_led("a", positions=[_pos(100, 50, 100)])]   # heat 0.5 > 0.05 cap
    assert portfolio_heat_breached(hot, ACCOUNT)
    assert not portfolio_heat_breached([_led(positions=[_pos(100, 99, 10)])], ACCOUNT)


def test_custom_limits_respected():
    led = _led(dd=-0.09)
    assert not check_suspension(led, ACCOUNT, RiskLimits(max_drawdown=0.10))[0]
    assert check_suspension(led, ACCOUNT, RiskLimits(max_drawdown=0.08))[0]


# --- sector ---------------------------------------------------------------

def test_sector_exposures_group_by_sector():
    positions = [
        _pos(100, size=10, symbol="AAPL", sector="tech"),
        _pos(100, size=10, symbol="MSFT", sector="tech"),
        _pos(100, size=10, symbol="XOM", sector="energy"),
    ]
    exp = sector_exposures(positions, ACCOUNT)
    assert math.isclose(exp["tech"], 0.20) and math.isclose(exp["energy"], 0.10)


# --- marginal risk contribution (MCTR) -------------------------------------

def test_mctr_hand_computed_two_uncorrelated_sleeves():
    # A: std=0.02, B: std=0.01, zero correlation, equal 50/50 weights.
    # By hand: cov=[[4e-4,0],[0,1e-4]], cov@w=[2e-4,5e-5], port_var=1.25e-4,
    # sigma_p=0.011180; shares = (w*cov_w/sigma_p)/sigma_p = 0.8 / 0.2.
    n = 200_000
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 0.02, n))
    b = pd.Series(rng.normal(0, 0.01, n))  # independent draw -> ~zero correlation
    out = marginal_risk_contributions({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    assert out["a"] == pytest.approx(0.8, abs=0.02)
    assert out["b"] == pytest.approx(0.2, abs=0.02)
    assert out["a"] + out["b"] == pytest.approx(1.0, abs=1e-6)


def test_mctr_sums_to_one_euler_identity():
    rng = np.random.default_rng(2)
    common = rng.normal(0, 0.01, 5000)
    returns = {
        name: pd.Series(common * beta + rng.normal(0, 0.01, 5000))
        for name, beta in [("mr", 0.3), ("tf", -0.2), ("mom", 0.5), ("sr", 0.1)]
    }
    weights = {"mr": 0.25, "tf": 0.25, "mom": 0.25, "sr": 0.25}
    out = marginal_risk_contributions(returns, weights)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)
    assert set(out) == set(weights)


def test_mctr_excludes_sleeves_without_return_series():
    rng = np.random.default_rng(3)
    returns = {"mr": pd.Series(rng.normal(0, 0.01, 500)), "tf": pd.Series(rng.normal(0, 0.01, 500))}
    weights = {"mr": 0.4, "tf": 0.4, "cash_reserve": 0.2}  # cash has no return series
    out = marginal_risk_contributions(returns, weights)
    assert out["cash_reserve"] == 0.0
    assert set(out) == set(weights)


def test_mctr_fewer_than_two_sleeves_is_all_zero():
    out = marginal_risk_contributions({"mr": pd.Series([0.01, 0.02])}, {"mr": 1.0, "tf": 0.0})
    assert out == {"mr": 0.0, "tf": 0.0}


def test_mctr_empty_is_safe():
    assert marginal_risk_contributions({}, {"mr": 1.0}) == {"mr": 0.0}
