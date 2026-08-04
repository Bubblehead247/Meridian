"""Stage-runner that drives a model through the out-of-sample validation stage.

This module owns running a model through the OOS stage and returning results;
it does NOT reimplement OOS math (reuses data/splits.py and validation/).

The fixed OOS holdout is evaluated *once* (the discipline is the caller's to keep): the
model is run on the held-out window from ``data.splits`` and scored. ``run_oos_stage``
also scores the in-sample window so it can flag profile degradation — the "paper/OOS
results match the backtest" check the graduation pilot gate needs.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from meridian.data.splits import SplitSpec, split
from meridian.families.base import Model
from meridian.pipeline.backtest import StageResult
from meridian.pipeline.graduation import GraduationCriteria, passes_metric_bar
from meridian.pipeline.oos_guard import OOSGuard
from meridian.scoring import scorecard_from_backtest


def _slice_bars(bars: pd.DataFrame | None, index: pd.Index) -> pd.DataFrame | None:
    return None if bars is None else bars.reindex(index)


def compare_oos_to_is(
    oos_card: dict, is_card: dict, *, tolerance: float = 0.5
) -> dict:
    """Flag out-of-sample degradation vs in-sample (Sharpe retention below ``tolerance``)."""
    is_sharpe = is_card.get("sharpe")
    oos_sharpe = oos_card.get("sharpe")
    degraded = (
        is_sharpe is not None and is_sharpe == is_sharpe and is_sharpe > 0
        and (oos_sharpe is None or oos_sharpe != oos_sharpe or oos_sharpe < is_sharpe * tolerance)
    )
    return {"is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe, "degraded": bool(degraded)}


def run_oos_stage(
    model: Model,
    prices: pd.Series,
    *,
    spec: SplitSpec | None = None,
    cost_bps: float = 1.0,
    bars: pd.DataFrame | None = None,
    regime_frame: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    criteria: GraduationCriteria | None = None,
    tolerance: float = 0.5,
    guard: OOSGuard | None = None,
    symbol: str | None = None,
) -> StageResult:
    """Run ``model`` on the fixed OOS holdout → scorecard → pass/fail (the ``oos`` gate).

    Passes only if the OOS scorecard clears the metric bar AND its profile has not degraded
    relative to in-sample.

    When ``guard`` is supplied, each call increments a persisted run counter for
    (``model.family``, ``model.name``, ``symbol``) and stamps the resulting count on
    ``detail["oos_run_count"]`` — a count above 1 means this "final" OOS pass followed
    one or more prior attempts against the same holdout (see ``pipeline/oos_guard.py``).
    Non-blocking: a repeated run still executes and returns normally, just visibly flagged.
    """
    spec = spec or SplitSpec()
    parts = split(prices, spec)
    oos, ins = parts["out_of_sample"], parts["in_sample"]
    if len(oos) == 0:
        return StageResult("oos", model.name, {}, False, {"n_oos": 0})

    oos_res = model.backtest(
        oos, cost_bps=cost_bps, bars=_slice_bars(bars, oos.index), regime_frame=regime_frame
    )
    oos_card = scorecard_from_backtest(
        oos_res, regime_frame=regime_frame, periods_per_year=periods_per_year
    )

    comparison = {"degraded": False}
    if len(ins):
        is_res = model.backtest(
            ins, cost_bps=cost_bps, bars=_slice_bars(bars, ins.index), regime_frame=regime_frame
        )
        is_card = scorecard_from_backtest(is_res, periods_per_year=periods_per_year)
        comparison = compare_oos_to_is(oos_card, is_card, tolerance=tolerance)

    # OOS window is intentionally short (~756 bars); skip the full-history min_periods guard
    oos_criteria = (
        dataclasses.replace(criteria, min_periods=0) if criteria
        else GraduationCriteria(min_periods=0)
    )
    passed = passes_metric_bar(oos_card, oos_criteria) and not comparison["degraded"]
    detail: dict = {"n_oos": int(len(oos)), **comparison}
    if guard is not None:
        run_rec = guard.record_run(model.family or "unknown", model.name or "unknown", symbol or "unknown")
        detail["oos_run_count"] = run_rec.run_count
    return StageResult(
        stage="oos",
        model=model.name,
        scorecard=oos_card,
        passed=passed,
        detail=detail,
    )
