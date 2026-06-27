"""Stage-runner that drives a model through the walk-forward validation stage.

This module owns running a model through the walk-forward stage and returning results;
it does NOT reimplement walk-forward math (reuses validation/walkforward.py).

Anchored walk-forward: a causal model's position path is fold-independent, so the model
is backtested once over full history and the disjoint out-of-sample test windows
(``make_folds``) are stitched into one OOS track record, then scored. This is the same
slice-once pattern the universe validator uses — no walk-forward math is duplicated here.
"""

from __future__ import annotations

import pandas as pd

from meridian.families.base import Model
from meridian.pipeline.backtest import StageResult
import dataclasses

from meridian.pipeline.graduation import GraduationCriteria, passes_metric_bar
from meridian.scoring import scorecard
from meridian.validation.walkforward import WalkForwardSpec, make_folds


def run_walk_forward_stage(
    model: Model,
    prices: pd.Series,
    *,
    spec: WalkForwardSpec | None = None,
    cost_bps: float = 1.0,
    bars: pd.DataFrame | None = None,
    regime_frame: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    criteria: GraduationCriteria | None = None,
) -> StageResult:
    """Walk-forward ``model`` → stitched OOS scorecard → pass/fail (the ``walk_forward`` gate)."""
    spec = spec or WalkForwardSpec()
    folds = make_folds(len(prices), spec)
    if not folds:
        return StageResult("walk_forward", model.name, {}, False, {"n_folds": 0})

    result = model.backtest(prices, cost_bps=cost_bps, bars=bars, regime_frame=regime_frame)
    oos_ret = pd.concat([result.returns.iloc[f.test_start:f.test_end] for f in folds])
    oos_pos = pd.concat([result.positions.iloc[f.test_start:f.test_end] for f in folds])
    card = scorecard(
        oos_ret, positions=oos_pos, regime_frame=regime_frame, periods_per_year=periods_per_year
    )
    return StageResult(
        stage="walk_forward",
        model=model.name,
        scorecard=card,
        passed=passes_metric_bar(card, dataclasses.replace(criteria, min_periods=0) if criteria else GraduationCriteria(min_periods=0)),
        detail={"n_folds": len(folds), "oos_periods": int(len(oos_ret))},
    )
