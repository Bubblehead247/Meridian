"""Stage-runner that drives a model through the backtest stage of the pipeline.

This module owns running a model through the backtest stage and returning results;
it does NOT reimplement backtest math (reuses signals/backtest.py and portfolio/).

``StageResult`` is the shared result type for all three stage runners (backtest /
walk-forward / OOS): a stage name, the model, its scorecard, and whether it cleared the
metric bar (``graduation.passes_metric_bar``) — exactly the evidence the graduation state
machine consumes to promote a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from meridian.families.base import Model
from meridian.pipeline.graduation import GraduationCriteria, passes_metric_bar
from meridian.scoring import scorecard_from_backtest


@dataclass
class StageResult:
    """Outcome of running a model through one pipeline stage."""

    stage: str
    model: str | None
    scorecard: dict
    passed: bool
    detail: dict = field(default_factory=dict)


def run_backtest_stage(
    model: Model,
    prices: pd.Series,
    *,
    cost_bps: float = 1.0,
    bars: pd.DataFrame | None = None,
    regime_frame: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    criteria: GraduationCriteria | None = None,
) -> StageResult:
    """Full-history backtest of ``model`` → scorecard → pass/fail (the ``backtest`` gate).

    Reuses ``Model.backtest`` (which reuses ``signals.backtest``) and
    ``scoring.scorecard_from_backtest`` — no backtest math here.
    """
    result = model.backtest(prices, cost_bps=cost_bps, bars=bars, regime_frame=regime_frame)
    card = scorecard_from_backtest(
        result, regime_frame=regime_frame, periods_per_year=periods_per_year
    )
    return StageResult(
        stage="backtest",
        model=model.name,
        scorecard=card,
        passed=passes_metric_bar(card, criteria),
        detail={
            "n_periods": card.get("n_periods", 0), "n_trades": card.get("n_trades", 0),
            "fill_realism": result.meta.get("fill_realism", "close_approx"),
        },
    )
