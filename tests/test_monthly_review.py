"""Tests for the month-end review (§9) and its markdown report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.portfolio import (
    StrategyLedger,
    run_monthly_review,
)
from meridian.regimes import RegimeLabel
from meridian.reporting import render_monthly_report, write_monthly_report

GOOD = {"sharpe": 1.0, "trade_expectancy": 0.01, "max_drawdown": -0.05}
ACCOUNT = 100_000.0


def _led(name, family, stage="proven", cap=1000.0, **kw):
    return StrategyLedger(
        name=name, family=family, stage=stage, stage_entered="2026-01-01",
        capital_alloc=cap, **kw,
    )


def _pos(symbol="X", entry=100.0, stop=99.0, size=10.0, **kw):
    return {"symbol": symbol, "entry_price": entry, "stop_price": stop, "size": size, **kw}


# --- per-sleeve actions ---------------------------------------------------

def test_promote_eligible_sleeve_gets_increase():
    led = _led("mr", "mean_reversion", stage="research", realized_pnl=50.0)
    rev = run_monthly_review([led], {"mr": GOOD}, account_equity=ACCOUNT, as_of="2026-02-01")
    assert rev.sleeves[0].action == "increase"
    assert "promote" in rev.sleeves[0].rationale[0]


def test_drawdown_breach_suspends():
    led = _led("td", "trend_following", drawdown_cur=-0.15)
    rev = run_monthly_review([led], {"td": GOOD}, account_equity=ACCOUNT, as_of="2026-06-01")
    assert rev.sleeves[0].action == "suspend"


def test_underperformance_reduces():
    led = _led("mo", "momentum", realized_pnl=0.0)             # 0% return
    rev = run_monthly_review(
        [led], {"mo": GOOD}, account_equity=ACCOUNT, benchmark_return=0.10, as_of="2026-06-01"
    )
    assert rev.sleeves[0].action == "reduce"
    assert "trails benchmark" in " ".join(rev.sleeves[0].rationale)


def test_moderate_drawdown_reduces():
    led = _led("br", "breakouts", drawdown_cur=-0.08)          # below suspend, above reduce
    rev = run_monthly_review([led], {"br": GOOD}, account_equity=ACCOUNT, as_of="2026-06-01")
    assert rev.sleeves[0].action == "reduce"


def test_healthy_sleeve_holds():
    led = _led("se", "sector_rotation", realized_pnl=20.0)     # always-permitted, tiny gain
    rev = run_monthly_review([led], {"se": GOOD}, account_equity=ACCOUNT, as_of="2026-06-01")
    assert rev.sleeves[0].action == "hold"


def test_regime_noncompliance_pulls_back_active_sleeve():
    led = _led("td", "trend_following", open_positions=[_pos()])
    bear = RegimeLabel(trend="bear", volatility="normal", breadth="neutral")
    rev = run_monthly_review(
        [led], {"td": GOOD}, account_equity=ACCOUNT, current_regime=bear, as_of="2026-06-01"
    )
    s = rev.sleeves[0]
    assert s.permitted is False
    assert s.action == "reduce"
    assert any("regime non-compliant" in r for r in s.rationale)


# --- portfolio-level context ----------------------------------------------

def test_portfolio_context_heat_and_correlation():
    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 0.01, 60), index=idx)
    leds = [_led("a", "momentum", open_positions=[_pos(stop=90)]),
            _led("b", "breakouts")]
    rev = run_monthly_review(
        leds, {"a": GOOD, "b": GOOD}, account_equity=ACCOUNT,
        sleeve_returns={"a": base, "b": base.copy()}, as_of="2026-06-01",
    )
    assert rev.portfolio_heat > 0                       # 'a' has an at-risk position
    assert abs(rev.avg_correlation - 1.0) < 1e-9        # identical streams
    assert ("a", "b") in rev.high_corr_pairs


# --- rendering ------------------------------------------------------------

def test_render_and_write_report(tmp_path):
    led = _led("mr", "mean_reversion", stage="research", realized_pnl=50.0)
    regime = RegimeLabel(trend="neutral", volatility="low", breadth="neutral")
    rev = run_monthly_review(
        [led], {"mr": GOOD}, account_equity=ACCOUNT, current_regime=regime, as_of="2026-06-30"
    )
    md = render_monthly_report(rev)
    assert "# Monthly Review — 2026-06-30" in md
    assert "## Sleeve actions" in md and "## Action summary" in md
    assert "trend `neutral`" in md

    path = write_monthly_report(rev, tmp_path)
    assert path.name == "monthly_review_2026-06-30.md"
    assert path.read_text(encoding="utf-8") == md
