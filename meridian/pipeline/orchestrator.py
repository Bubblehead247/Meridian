"""Top-level driver that walks a model through the pipeline stages.

This module owns sequencing a model through the stage runners and (optionally) advancing
its ledger; it does NOT own the stage math (reuses pipeline/backtest|walk_forward|oos).

``run_pipeline`` runs the chosen stages in order (backtest → walk_forward → oos),
advancing the strategy's graduation stage from each stage's scorecard and stopping early
when a stage fails — the single call that ties model → pipeline → ledger/graduation
together for one strategy on one price series.
"""

from __future__ import annotations

import pandas as pd

from meridian.families.base import Model
from meridian.pipeline.backtest import StageResult, run_backtest_stage
import dataclasses

from meridian.pipeline.graduation import GraduationCriteria, advance, criteria_for_family
from meridian.pipeline.oos import run_oos_stage
from meridian.pipeline.walk_forward import run_walk_forward_stage
from meridian.portfolio.ledger import StrategyLedger

DEFAULT_STAGES = ("backtest", "walk_forward", "oos")


def run_pipeline(
    model: Model,
    prices: pd.Series,
    *,
    ledger: StrategyLedger | None = None,
    stages: tuple[str, ...] = DEFAULT_STAGES,
    cost_bps: float = 1.0,
    bars: pd.DataFrame | None = None,
    regime_frame: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    criteria: GraduationCriteria | None = None,
    wf_spec=None,
    oos_spec=None,
    as_of: str | None = None,
    stop_on_fail: bool = True,
) -> dict[str, StageResult]:
    """Walk ``model`` through ``stages`` in order; advance ``ledger`` from each scorecard.

    Returns ``{stage: StageResult}``. When a ``ledger`` is given, each stage's scorecard is
    fed to ``graduation.advance`` (the resulting action is stored on the StageResult's
    ``detail['ledger_action']``). With ``stop_on_fail`` the walk halts at the first stage
    that does not clear the metric bar.
    """
    if criteria is None:
        criteria = criteria_for_family(ledger.family) if ledger is not None else GraduationCriteria()
    common = dict(
        cost_bps=cost_bps, bars=bars, regime_frame=regime_frame,
        periods_per_year=periods_per_year, criteria=criteria,
    )
    runners = {
        "backtest": (run_backtest_stage, {}),
        "walk_forward": (run_walk_forward_stage, {"spec": wf_spec} if wf_spec else {}),
        "oos": (run_oos_stage, {"spec": oos_spec} if oos_spec else {}),
    }

    results: dict[str, StageResult] = {}
    for stage in stages:
        if stage not in runners:
            raise KeyError(f"unknown stage {stage!r}. Known: {', '.join(runners)}")
        runner, extra = runners[stage]
        res = runner(model, prices, **common, **extra)
        if ledger is not None:
            # WF/OOS windows are intentionally short — skip the full-history min_periods
            # guard when advancing the ledger for those stages (matches the stage runners).
            adv_criteria = (
                dataclasses.replace(criteria, min_periods=0)
                if stage in ("walk_forward", "oos") else criteria
            )
            res.detail["ledger_action"] = advance(
                ledger, res.scorecard, as_of=as_of, criteria=adv_criteria
            )
        results[stage] = res
        if stop_on_fail and not res.passed:
            break
    return results
