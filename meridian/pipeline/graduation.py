"""Stage state machine that governs promotion and retirement of strategy models.

This module owns stage state and promotion/retirement rules (steps 6-10); it
does NOT own capital math (reads ledger and scorecard, does not compute them).

A strategy advances one stage at a time. The early validation stages (research → … →
paper) advance on *evidence* — a scorecard that clears the metric bar — while the live
stages (paper → pilot → proven → core → elite) add a *time-in-stage* gate (30/90/180/365
days). Retirement is a separate downgrade triggered by degraded expectancy or a drawdown
breach. Capital levels per stage are exposed as a table for allocation/reporting; setting
dollars is allocation's job, not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from meridian.portfolio.allocation import SLEEVE_ALLOCATIONS
from meridian.portfolio.ledger import STAGES, StrategyLedger

#: Live-capital range (fraction of equity) per stage — policy table (PLAN.md §6).
STAGE_CAPITAL: dict[str, tuple[float, float]] = {
    "research": (0.0, 0.0), "backtest": (0.0, 0.0), "walk_forward": (0.0, 0.0),
    "oos": (0.0, 0.0), "paper": (0.0, 0.0),
    "pilot": (0.01, 0.03), "proven": (0.05, 0.10), "core": (0.10, 0.20),
    "elite": (0.20, 1.0), "retired": (0.0, 0.0),
}

#: Minimum days in the *current* stage before it may promote (live stages only).
MIN_DAYS_IN_STAGE: dict[str, int] = {"paper": 30, "pilot": 90, "proven": 180, "core": 365}


@dataclass(frozen=True)
class GraduationCriteria:
    """The metric bar a scorecard must clear to promote, and the retirement limits.

    Drawdown is evaluated as a portfolio contribution when ``allocation_weight`` is set:
    the strategy's max drawdown × sleeve weight must not exceed ``max_portfolio_dd_contribution``.
    Without a weight the raw ``max_drawdown_limit`` fallback is used instead.
    """

    min_sharpe: float = 0.35
    min_expectancy: float = 0.0
    max_drawdown_limit: float = 0.45                # fallback when allocation_weight is None
    max_portfolio_dd_contribution: float = 0.08     # max allowed portfolio drawdown impact
    allocation_weight: float | None = None          # sleeve's share of total equity
    min_periods: int = 1260                         # minimum bars required (~5 years)


#: Per-family criteria keyed by sleeve name, each carrying its allocation weight.
#: Families not listed fall back to GraduationCriteria() with no allocation_weight.
FAMILY_CRITERIA: dict[str, GraduationCriteria] = {
    family: GraduationCriteria(allocation_weight=weight)
    for family, weight in SLEEVE_ALLOCATIONS.items()
}


def criteria_for_family(family: str | None) -> GraduationCriteria:
    """Return the graduation criteria for ``family``, falling back to the global default."""
    if family and family in FAMILY_CRITERIA:
        return FAMILY_CRITERIA[family]
    return GraduationCriteria()


def _days_in_stage(ledger: StrategyLedger, as_of: date | str | None) -> int:
    as_of = date.today() if as_of is None else (
        date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    )
    return (as_of - date.fromisoformat(ledger.stage_entered)).days


def _get(scorecard: dict, *keys, default=None):
    """First present, non-None value among ``keys`` (scorecard schemas vary)."""
    for k in keys:
        v = scorecard.get(k)
        if v is not None:
            return v
    return default


def _dd_within_limit(dd: float, criteria: GraduationCriteria) -> bool:
    """True if the drawdown magnitude is within the allowed limit.

    When the criteria carries an allocation weight the check is portfolio-impact-based:
    abs(dd) × weight ≤ max_portfolio_dd_contribution.  Otherwise the raw
    max_drawdown_limit fallback applies.
    """
    mag = abs(dd)
    if criteria.allocation_weight is not None:
        return mag * criteria.allocation_weight <= criteria.max_portfolio_dd_contribution
    return mag <= criteria.max_drawdown_limit


def passes_metric_bar(scorecard: dict, criteria: GraduationCriteria | None = None) -> bool:
    """Whether a scorecard alone clears the promotion metric bar (fails closed on missing).

    Sharpe ≥ ``min_sharpe``, expectancy > ``min_expectancy`` (if present), and the
    drawdown within the portfolio contribution limit. Used both by promotion and by the
    pipeline stage runners as the per-stage pass test.
    """
    criteria = criteria or GraduationCriteria()
    n_periods = _get(scorecard, "n_periods")
    if n_periods is not None and n_periods == n_periods and n_periods < criteria.min_periods:
        return False
    sharpe = _get(scorecard, "sharpe")
    if sharpe is None or sharpe != sharpe or sharpe < criteria.min_sharpe:   # NaN-safe
        return False
    exp = _get(scorecard, "trade_expectancy", "expectancy")
    if exp is not None and exp == exp and exp <= criteria.min_expectancy:   # NaN-safe
        return False
    dd = _get(scorecard, "max_drawdown")
    if dd is not None and dd == dd and not _dd_within_limit(dd, criteria):
        return False
    return True


def _metrics_pass(scorecard: dict, ledger: StrategyLedger, criteria: GraduationCriteria) -> bool:
    """Promotion metric gate with ledger fallbacks for a thin scorecard."""
    merged = dict(scorecard)
    merged.setdefault("expectancy", ledger.expectancy)
    merged.setdefault("max_drawdown", ledger.drawdown_max)
    return passes_metric_bar(merged, criteria)


def next_stage(stage: str) -> str | None:
    """The stage immediately above ``stage`` (None at ``elite``/``retired``)."""
    i = STAGES.index(stage)
    if STAGES[i] in ("elite", "retired"):
        return None
    nxt = STAGES[i + 1]
    return None if nxt == "retired" else nxt   # retirement is a separate downgrade


# Stages where the strategy has never been deployed with live capital.
# Retirement signals don't apply here — the strategy was never "running".
_PRE_LIVE_STAGES = frozenset(("research", "backtest", "walk_forward", "oos", "paper"))


def check_retirement(
    ledger: StrategyLedger, scorecard: dict, criteria: GraduationCriteria | None = None
) -> bool:
    """True when a *live* strategy's expectancy has degraded or drawdown breaches the limit.

    Pre-live stages (research through paper) are never retired — they are still under
    validation and have not been deployed with real capital, so degraded backtest metrics
    are expected noise, not a live-capital risk signal.
    """
    if ledger.stage in _PRE_LIVE_STAGES:
        return False
    criteria = criteria or GraduationCriteria()
    expectancy = _get(scorecard, "trade_expectancy", "expectancy", default=ledger.expectancy)
    if expectancy is not None and expectancy == expectancy and expectancy <= criteria.min_expectancy:
        return True
    dd = _get(scorecard, "max_drawdown", default=ledger.drawdown_max)
    if dd is not None and dd == dd and not _dd_within_limit(dd, criteria):
        return True
    if not _dd_within_limit(ledger.drawdown_cur, criteria):
        return True
    return False


def check_promotion(
    ledger: StrategyLedger,
    scorecard: dict,
    *,
    as_of: date | str | None = None,
    criteria: GraduationCriteria | None = None,
) -> str | None:
    """The stage ``ledger`` qualifies to advance to, or None.

    Requires: a next stage exists, enough days in the current stage (live stages), and a
    scorecard that clears the metric bar. A strategy that meets the retirement trigger is
    never promoted.
    """
    criteria = criteria or GraduationCriteria()
    if check_retirement(ledger, scorecard, criteria):
        return None
    nxt = next_stage(ledger.stage)
    if nxt is None:
        return None
    if _days_in_stage(ledger, as_of) < MIN_DAYS_IN_STAGE.get(ledger.stage, 0):
        return None
    if not _metrics_pass(scorecard, ledger, criteria):
        return None
    return nxt


def evaluate(
    ledger: StrategyLedger,
    scorecard: dict,
    *,
    as_of: date | str | None = None,
    criteria: GraduationCriteria | None = None,
) -> tuple[str, str | None]:
    """Decide the action for a strategy: ``("retire"|"promote"|"hold", target_stage)``."""
    criteria = criteria or GraduationCriteria()
    if check_retirement(ledger, scorecard, criteria):
        return ("retire", "retired")
    promote = check_promotion(ledger, scorecard, as_of=as_of, criteria=criteria)
    if promote is not None:
        return ("promote", promote)
    return ("hold", None)


def advance(
    ledger: StrategyLedger,
    scorecard: dict,
    *,
    as_of: date | str | None = None,
    criteria: GraduationCriteria | None = None,
) -> str:
    """Evaluate and *apply* the stage transition on the ledger; return the action taken.

    Mutates ``ledger.stage``/``stage_entered`` (stage state is this module's to own); it
    does not touch ``capital_alloc`` (allocation's job).
    """
    action, target = evaluate(ledger, scorecard, as_of=as_of, criteria=criteria)
    if target is not None:
        on = None if as_of is None else (as_of if isinstance(as_of, str) else as_of.isoformat())
        ledger.set_stage(target, on=on)
    return action
