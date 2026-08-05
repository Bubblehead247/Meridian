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
from pathlib import Path

from meridian.portfolio.allocation import FAMILY_TO_SLEEVE, SLEEVE_ALLOCATIONS
from meridian.portfolio.ledger import STAGES, StrategyLedger

#: Live-capital range (fraction of equity) per stage — policy table (PLAN.md §6).
STAGE_CAPITAL: dict[str, tuple[float, float]] = {
    "research": (0.0, 0.0), "backtest": (0.0, 0.0), "walk_forward": (0.0, 0.0),
    "oos": (0.0, 0.0), "paper": (0.0, 0.0),
    "pilot": (0.01, 0.03), "proven": (0.05, 0.10), "core": (0.10, 0.20),
    "elite": (0.20, 1.0), "retired": (0.0, 0.0),
}

#: Versioned criteria config (P1-C). See ``load_criteria_config`` for the fallback
#: used when the file is missing (e.g. a different working directory).
DEFAULT_CRITERIA_CONFIG = Path("configs") / "graduation_criteria.yaml"

#: Hardcoded fallback — identical to the shipped configs/graduation_criteria.yaml
#: v1.0.0 — used only when that file cannot be found, so graduation still works
#: outside a repo checkout (e.g. an installed package).
_FALLBACK_CONFIG: dict = {
    "version": "1.0.0-fallback",
    "min_sharpe": 0.35,
    "min_expectancy": 0.0,
    "max_drawdown_limit": 0.45,
    "max_portfolio_dd_contribution": 0.08,
    "min_periods": 1260,
    "min_days_in_stage": {"paper": 30, "pilot": 90, "proven": 180, "core": 365},
    "max_days_in_stage": {
        "research": 90, "backtest": 90, "walk_forward": 90, "oos": 90, "paper": 180,
    },
}


def load_criteria_config(path: str | Path | None = None) -> dict:
    """Read the versioned graduation-criteria config, or the hardcoded fallback.

    Re-reads the file on every call (it is tiny and rarely read in a hot loop) so a
    file edit is picked up immediately rather than needing a process restart — which
    matters for ``advance``'s version-mismatch check below.
    """
    from meridian.config import load_config

    p = Path(path) if path is not None else DEFAULT_CRITERIA_CONFIG
    try:
        return load_config(p)
    except FileNotFoundError:
        return dict(_FALLBACK_CONFIG)


#: Minimum days in the *current* stage before it may promote (live stages only).
#: Kept as a module-level constant for backward-compatible imports; reflects the
#: default config at import time. Prefer ``load_criteria_config()["min_days_in_stage"]``
#: for a value that respects a config edited after process start.
MIN_DAYS_IN_STAGE: dict[str, int] = dict(load_criteria_config()["min_days_in_stage"])

#: Maximum days in a *pre-live* stage before ``check_stale`` flags it (non-blocking).
MAX_DAYS_IN_STAGE: dict[str, int] = dict(load_criteria_config().get("max_days_in_stage", {}))


@dataclass(frozen=True)
class GraduationCriteria:
    """The metric bar a scorecard must clear to promote, and the retirement limits.

    Drawdown is evaluated as a portfolio contribution when ``allocation_weight`` is set:
    the strategy's max drawdown × sleeve weight must not exceed ``max_portfolio_dd_contribution``.
    Without a weight the raw ``max_drawdown_limit`` fallback is used instead.

    ``version`` identifies which edit of ``configs/graduation_criteria.yaml`` produced
    this instance (or ``"unversioned"`` for a criteria object built ad-hoc, e.g. in a
    test). ``pipeline.graduation.advance`` uses it to detect a strategy being judged
    under criteria that changed after it entered the pipeline — see that function.
    """

    min_sharpe: float = 0.35
    min_expectancy: float = 0.0
    max_drawdown_limit: float = 0.45                # fallback when allocation_weight is None
    max_portfolio_dd_contribution: float = 0.08     # max allowed portfolio drawdown impact
    allocation_weight: float | None = None          # sleeve's share of total equity
    min_periods: int = 1260                         # minimum bars required (~5 years)
    version: str = "unversioned"


def _criteria_from_config(cfg: dict, *, allocation_weight: float | None = None) -> GraduationCriteria:
    return GraduationCriteria(
        min_sharpe=cfg["min_sharpe"],
        min_expectancy=cfg["min_expectancy"],
        max_drawdown_limit=cfg["max_drawdown_limit"],
        max_portfolio_dd_contribution=cfg["max_portfolio_dd_contribution"],
        allocation_weight=allocation_weight,
        min_periods=cfg["min_periods"],
        version=cfg["version"],
    )


def criteria_for_family(
    family: str | None, *, config_path: str | Path | None = None
) -> GraduationCriteria:
    """Return the graduation criteria for ``family``, falling back to the global default.

    Resolves through ``FAMILY_TO_SLEEVE`` first: a family that shares a sleeve with
    another (e.g. ``breakouts`` -> ``trend_following``, ``volatility`` ->
    ``experimental_research``) is keyed by *sleeve* name in ``SLEEVE_ALLOCATIONS``, not
    its own family name, since that's what determines its allocation weight and
    portfolio drawdown-contribution limit. The threshold values themselves (min_sharpe,
    min_periods, etc.) come from ``configs/graduation_criteria.yaml`` and are the same
    across every family — only ``allocation_weight`` varies.
    """
    cfg = load_criteria_config(config_path)
    if not family:
        return _criteria_from_config(cfg)
    sleeve = FAMILY_TO_SLEEVE.get(family, family)
    weight = SLEEVE_ALLOCATIONS.get(sleeve)
    return _criteria_from_config(cfg, allocation_weight=weight)


#: Per-family criteria keyed by sleeve name, at the default config's current version.
#: Kept for backward-compatible imports; prefer ``criteria_for_family`` in new code,
#: since this snapshot does not reflect a config edited after import time.
FAMILY_CRITERIA: dict[str, GraduationCriteria] = {
    family: criteria_for_family(family) for family in SLEEVE_ALLOCATIONS
}


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
    if (
        expectancy is not None
        and expectancy == expectancy
        and expectancy <= criteria.min_expectancy
    ):
        return True
    dd = _get(scorecard, "max_drawdown", default=ledger.drawdown_max)
    if dd is not None and dd == dd and not _dd_within_limit(dd, criteria):
        return True
    if not _dd_within_limit(ledger.drawdown_cur, criteria):
        return True
    return False


def check_stale(ledger: StrategyLedger, as_of: date | str | None = None) -> bool:
    """True when a *pre-live* strategy has sat in its current stage past ``max_days_in_stage``.

    Non-blocking, mirroring ``pipeline/oos_guard.py``'s philosophy: this flags rather than
    refuses, since a legitimate bug-fix re-run also spends time in a pre-live stage. But an
    un-flagged strategy can otherwise be re-tried in ``research``/``backtest``/etc.
    indefinitely with no visible signal that it is happening — this is that signal (see
    ``research_integrity_gap_analysis.md`` §3.4). Live stages are excluded: their promotion
    gate already has an explicit *minimum* dwell (``MIN_DAYS_IN_STAGE``), and sitting in
    ``elite`` a long time is the intended outcome, not staleness.
    """
    if ledger.stage not in _PRE_LIVE_STAGES:
        return False
    limit = MAX_DAYS_IN_STAGE.get(ledger.stage)
    if limit is None:
        return False
    return _days_in_stage(ledger, as_of) > limit


class CriteriaVersionMismatch(ValueError):
    """Raised when a ledger already bound to one criteria version is judged under another.

    See ``advance``'s docstring: pass ``allow_criteria_override=True`` and an
    ``override_reason`` to proceed deliberately (logged to the append-only run log)
    instead of catching this and ignoring it.
    """


def _log_criteria_override(
    ledger: StrategyLedger, criteria: GraduationCriteria, reason: str
) -> None:
    """Append a record of a deliberate criteria-version override to the run log.

    Reuses ``experiments/run_log.py``'s append-only JSONL rather than adding a second
    logging mechanism — an override is exactly the kind of "what changed and why"
    research-process event that log already exists to capture.
    """
    from meridian.experiments import run_log as run_log_mod

    run_log_mod.append_run(
        run_log_mod.new_run_record(
            config_path=str(DEFAULT_CRITERIA_CONFIG),
            raw_config_text=None,
            kind="graduation_criteria_override",
            meta={
                "strategy": ledger.name, "family": ledger.family, "stage": ledger.stage,
                "from_version": ledger.criteria_version, "to_version": criteria.version,
                "reason": reason,
            },
            report_path=None,
            summary={},
        ),
        # Looked up on the module (not bound as a default-arg value) so tests that
        # monkeypatch run_log_mod.DEFAULT_RUN_LOG actually redirect this write.
        path=run_log_mod.DEFAULT_RUN_LOG,
    )


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
    allow_criteria_override: bool = False,
    override_reason: str | None = None,
) -> str:
    """Evaluate and *apply* the stage transition on the ledger; return the action taken.

    Mutates ``ledger.stage``/``stage_entered`` (stage state is this module's to own); it
    does not touch ``capital_alloc`` (allocation's job).

    The first call on a ledger binds ``ledger.criteria_version`` to ``criteria.version``
    (P1-C). A later call with a *different* version means ``configs/graduation_criteria.yaml``
    was edited while this strategy was still in the pipeline — i.e. it would be judged by a
    rule it didn't start under. That is refused by default (``CriteriaVersionMismatch``) so
    a threshold change can't silently re-judge an in-flight strategy. To proceed
    deliberately, pass ``allow_criteria_override=True`` with a non-empty ``override_reason``;
    the override is appended to the run log (``reports/run_log.jsonl``) before it takes
    effect, and the ledger is then rebound to the new version.
    """
    criteria = criteria or GraduationCriteria()
    if ledger.criteria_version is None:
        ledger.criteria_version = criteria.version
    elif ledger.criteria_version != criteria.version:
        if not allow_criteria_override:
            raise CriteriaVersionMismatch(
                f"{ledger.name!r} was bound to graduation criteria version "
                f"{ledger.criteria_version!r} but is now being evaluated under "
                f"{criteria.version!r}. Pass allow_criteria_override=True with a reason "
                "to re-judge it under the new criteria deliberately, or resolve "
                "criteria_for_family(...) at the original version to keep evaluating it "
                "under the criteria it entered the pipeline under."
            )
        if not override_reason:
            raise ValueError("allow_criteria_override=True requires a non-empty override_reason")
        _log_criteria_override(ledger, criteria, override_reason)
        ledger.criteria_version = criteria.version
    action, target = evaluate(ledger, scorecard, as_of=as_of, criteria=criteria)
    if target is not None:
        on = None if as_of is None else (as_of if isinstance(as_of, str) else as_of.isoformat())
        ledger.set_stage(target, on=on)
    return action
