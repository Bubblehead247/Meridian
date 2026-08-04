"""Month-end evaluation process producing per-sleeve action recommendations.

This module owns the month-end evaluation checklist and per-sleeve action output;
it does NOT own rendering (that stays in reporting/monthly_report.py).

It reads every ledger, its scorecard, and the current regime, runs the §9 checklist (P&L
vs benchmark, drawdown, risk contribution, regime/permission compliance, position-sizing
adherence) and emits one action per sleeve — ``increase`` / ``hold`` / ``reduce`` /
``suspend`` — plus portfolio-level context (heat, inter-sleeve correlation). The action
re-uses the risk-budget suspension signal and the graduation verdict; it does not
recompute either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # annotations only — runtime import is lazy (avoids portfolio<->pipeline cycle)
    from meridian.pipeline.graduation import GraduationCriteria

from meridian.families.permissions import PERMISSIONS, is_permitted
from meridian.portfolio.correlation import (
    average_correlation,
    build_correlation_matrix,
    flag_high_correlation,
)
from meridian.portfolio.ledger import StrategyLedger
from meridian.portfolio.risk_budget import (
    RiskLimits,
    check_suspension,
    compute_portfolio_heat,
    marginal_risk_contributions,
    portfolio_heat_breached,
    sector_exposures,
)
from meridian.regimes.labeler import RegimeLabel

ACTIONS = ("increase", "hold", "reduce", "suspend")


@dataclass(frozen=True)
class ReviewConfig:
    """Thresholds that turn checklist findings into capital actions."""

    reduce_drawdown: float = 0.06          # reduce past this current DD (below the suspend limit)
    reduce_underperformance: float = 0.05  # reduce if it trails the benchmark by more than this


@dataclass
class SleeveAction:
    """One sleeve's month-end verdict."""

    sleeve: str
    family: str
    stage: str
    action: str
    pnl: float
    return_pct: float
    excess_vs_benchmark: float
    drawdown_cur: float
    risk_contribution: float
    permitted: bool
    suspend_reasons: list[str]
    graduation: tuple[str, str | None]
    rationale: list[str] = field(default_factory=list)


@dataclass
class MonthlyReview:
    """The full month-end review: portfolio context + per-sleeve actions."""

    as_of: str
    account_equity: float
    benchmark_return: float
    portfolio_heat: float
    heat_breached: bool
    avg_correlation: float
    high_corr_pairs: list[tuple[str, str]]
    regime: tuple[str, str, str] | None
    sleeves: list[SleeveAction]
    marginal_risk_contribution: dict[str, float] = field(default_factory=dict)


def _permitted(family: str, regime: RegimeLabel | None) -> bool:
    """Regime permission, treating sleeves outside the matrix (e.g. cash) as ungated."""
    if regime is None or family not in PERMISSIONS:
        return True
    return is_permitted(family, regime)


def _evaluate_checklist(
    ledger: StrategyLedger,
    scorecard: dict,
    *,
    account_equity: float,
    benchmark_return: float,
    current_regime: RegimeLabel | None,
    limits: RiskLimits,
    criteria: GraduationCriteria,
    config: ReviewConfig,
    as_of: str,
) -> SleeveAction:
    """Run the checklist for one sleeve and decide its capital action."""
    from meridian.pipeline.graduation import evaluate  # lazy: avoids portfolio<->pipeline cycle

    pnl = ledger.realized_pnl + ledger.unrealized_pnl
    return_pct = pnl / ledger.capital_alloc if ledger.capital_alloc > 0 else 0.0
    excess = return_pct - benchmark_return

    suspend, reasons = check_suspension(ledger, account_equity, limits)
    grad = evaluate(ledger, scorecard, as_of=as_of, criteria=criteria)
    permitted = _permitted(ledger.family, current_regime)

    rationale: list[str] = []
    if suspend or grad[0] == "retire":
        action = "suspend"
        if grad[0] == "retire":
            rationale.append("graduation: retire (expectancy/drawdown)")
        if reasons:
            rationale.append("risk suspension: " + ", ".join(reasons))
    elif grad[0] == "promote":
        action = "increase"
        rationale.append(f"graduation: promote to {grad[1]}")
    elif ledger.drawdown_cur < -config.reduce_drawdown:
        action = "reduce"
        rationale.append(f"drawdown {ledger.drawdown_cur:.1%} beyond reduce threshold")
    elif excess < -config.reduce_underperformance:
        action = "reduce"
        rationale.append(f"trails benchmark by {excess:.1%}")
    else:
        action = "hold"

    # regime/permission compliance: an active, non-permitted sleeve is pulled back
    if not permitted and ledger.open_positions:
        rationale.append("regime non-compliant: family restricted but holding positions")
        if action in ("hold", "increase"):
            action = "reduce"

    # position-sizing adherence (flagged, does not by itself change the action)
    for p in ledger.open_positions:
        from meridian.portfolio.risk_budget import position_weight

        if position_weight(p, account_equity) > limits.max_position:
            rationale.append(f"position {p.get('symbol', '?')} exceeds max position weight")
    for sector, w in sector_exposures(ledger.open_positions, account_equity).items():
        if w > limits.max_sector:
            rationale.append(f"sector {sector} exposure {w:.0%} exceeds limit")

    return SleeveAction(
        sleeve=ledger.name, family=ledger.family, stage=ledger.stage, action=action,
        pnl=pnl, return_pct=return_pct, excess_vs_benchmark=excess,
        drawdown_cur=ledger.drawdown_cur, risk_contribution=ledger.risk_contribution,
        permitted=permitted, suspend_reasons=reasons, graduation=grad, rationale=rationale,
    )


def run_monthly_review(
    ledgers: list[StrategyLedger],
    scorecards: dict[str, dict],
    *,
    account_equity: float,
    as_of: str | None = None,
    current_regime: RegimeLabel | None = None,
    benchmark_return: float = 0.0,
    sleeve_returns: dict[str, pd.Series] | None = None,
    limits: RiskLimits | None = None,
    criteria: GraduationCriteria | None = None,
    config: ReviewConfig | None = None,
) -> MonthlyReview:
    """Run the month-end review across all sleeves (PLAN.md §9).

    Args:
        ledgers: every strategy ledger.
        scorecards: ``{sleeve: scorecard dict}`` (from scoring/scorecard).
        account_equity: total account equity (for heat / sizing).
        current_regime: today's RegimeLabel (for permission compliance).
        benchmark_return: the benchmark's return over the review period (e.g. SPY).
        sleeve_returns: optional ``{sleeve: returns}`` for the correlation block.
    """

    from meridian.pipeline.graduation import criteria_for_family  # lazy: avoids import cycle

    limits = limits or RiskLimits()
    config = config or ReviewConfig()
    as_of = as_of or date.today().isoformat()

    actions = [
        _evaluate_checklist(
            led, scorecards.get(led.name, {}),
            account_equity=account_equity, benchmark_return=benchmark_return,
            current_regime=current_regime, limits=limits,
            criteria=criteria or criteria_for_family(led.family),
            config=config, as_of=as_of,
        )
        for led in ledgers
    ]

    matrix = build_correlation_matrix(sleeve_returns or {})
    regime_tuple = (
        (current_regime.trend, current_regime.volatility, current_regime.breadth)
        if current_regime is not None else None
    )

    weights = {
        led.name: (led.capital_alloc / account_equity if account_equity > 0 else 0.0)
        for led in ledgers
    }
    mctr = marginal_risk_contributions(sleeve_returns or {}, weights)

    return MonthlyReview(
        as_of=as_of,
        account_equity=account_equity,
        benchmark_return=benchmark_return,
        portfolio_heat=compute_portfolio_heat(ledgers, account_equity),
        heat_breached=portfolio_heat_breached(ledgers, account_equity, limits),
        avg_correlation=average_correlation(matrix) if not matrix.empty else float("nan"),
        high_corr_pairs=flag_high_correlation(matrix) if not matrix.empty else [],
        regime=regime_tuple,
        sleeves=actions,
        marginal_risk_contribution=mctr,
    )
