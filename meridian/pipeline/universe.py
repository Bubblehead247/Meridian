"""Universe-level stage runners for cross-sectional models.

This module owns sequencing a *cross-sectional* model (one whose ``backtest`` takes a
``{symbol: price}`` universe) through the pipeline stages; it does NOT own the cross-
sectional P&L (that stays in the model / portfolio.backtest_portfolio). It mirrors the
single-asset runners (pipeline/backtest.py) but scores the portfolio return stream.

Models are duck-typed: anything with ``.backtest(prices_by_symbol)`` returning a result
with a ``.returns`` series and a ``.name`` attribute works (i.e. CrossSectionalModel).
"""

from __future__ import annotations

import pandas as pd

import dataclasses

from meridian.pipeline.backtest import StageResult
from meridian.pipeline.graduation import GraduationCriteria, advance, criteria_for_family, passes_metric_bar
from meridian.scoring import scorecard
from meridian.validation.walkforward import WalkForwardSpec, make_folds


def _slice_universe(
    prices_by_symbol: dict[str, pd.Series],
    start: str | None,
    end: str | None,
) -> dict[str, pd.Series]:
    """Return the subset of each series that falls within [start, end]."""
    out = {}
    for sym, s in prices_by_symbol.items():
        sl = s
        if start:
            sl = sl[sl.index >= pd.Timestamp(start)]
        if end:
            sl = sl[sl.index <= pd.Timestamp(end)]
        if len(sl):
            out[sym] = sl
    return out


def run_universe_backtest_stage(
    model,
    prices_by_symbol: dict[str, pd.Series],
    *,
    cost_bps: float = 1.0,
    regime_frame: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    criteria: GraduationCriteria | None = None,
) -> StageResult:
    """Backtest a cross-sectional model over the whole universe → scorecard → pass/fail."""
    result = model.backtest(prices_by_symbol, cost_bps=cost_bps)
    card = scorecard(result.returns, regime_frame=regime_frame, periods_per_year=periods_per_year)
    return StageResult(
        stage="backtest",
        model=model.name,
        scorecard=card,
        passed=passes_metric_bar(card, criteria),
        detail={"n_symbols": len(prices_by_symbol), "n_periods": card.get("n_periods", 0)},
    )


def run_universe_walk_forward_stage(
    model,
    prices_by_symbol: dict[str, pd.Series],
    *,
    spec: WalkForwardSpec | None = None,
    cost_bps: float = 1.0,
    regime_frame: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    criteria: GraduationCriteria | None = None,
) -> StageResult:
    """Anchored walk-forward of a cross-sectional model: stitch the OOS portfolio returns."""
    spec = spec or WalkForwardSpec()
    result = model.backtest(prices_by_symbol, cost_bps=cost_bps)
    folds = make_folds(len(result.returns), spec)
    if not folds:
        return StageResult("walk_forward", model.name, {}, False, {"n_folds": 0})
    oos = pd.concat([result.returns.iloc[f.test_start:f.test_end] for f in folds])
    card = scorecard(oos, regime_frame=regime_frame, periods_per_year=periods_per_year)
    return StageResult(
        stage="walk_forward",
        model=model.name,
        scorecard=card,
        passed=passes_metric_bar(card, dataclasses.replace(criteria, min_periods=0) if criteria else GraduationCriteria(min_periods=0)),
        detail={"n_folds": len(folds), "oos_periods": int(len(oos)),
                "n_symbols": len(prices_by_symbol)},
    )


def run_universe_oos_stage(
    model,
    prices_by_symbol: dict[str, pd.Series],
    *,
    spec=None,
    cost_bps: float = 1.0,
    regime_frame: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    criteria: GraduationCriteria | None = None,
    tolerance: float = 0.5,
) -> StageResult:
    """Run a cross-sectional model on the fixed OOS holdout → scorecard → pass/fail.

    Slices every series in the universe to the OOS date window, backtests, and
    checks for degradation relative to in-sample — mirroring ``run_oos_stage`` for
    single-asset models.
    """
    from meridian.data.splits import SplitSpec
    from meridian.pipeline.oos import compare_oos_to_is

    spec = spec or SplitSpec()
    oos = _slice_universe(prices_by_symbol, *spec.out_of_sample)
    ins = _slice_universe(prices_by_symbol, *spec.in_sample)

    if not oos:
        return StageResult("oos", model.name, {}, False, {"n_oos": 0})

    oos_res = model.backtest(oos, cost_bps=cost_bps)
    oos_card = scorecard(oos_res.returns, regime_frame=regime_frame, periods_per_year=periods_per_year)

    comparison = {"degraded": False}
    if ins:
        is_res = model.backtest(ins, cost_bps=cost_bps)
        is_card = scorecard(is_res.returns, periods_per_year=periods_per_year)
        comparison = compare_oos_to_is(oos_card, is_card, tolerance=tolerance)

    oos_criteria = dataclasses.replace(criteria, min_periods=0) if criteria else GraduationCriteria(min_periods=0)
    passed = passes_metric_bar(oos_card, oos_criteria) and not comparison["degraded"]
    return StageResult(
        stage="oos",
        model=model.name,
        scorecard=oos_card,
        passed=passed,
        detail={"n_oos": int(len(oos_res.returns)), **comparison, "n_symbols": len(oos)},
    )


UNIVERSE_STAGE_RUNNERS = {
    "backtest": run_universe_backtest_stage,
    "walk_forward": run_universe_walk_forward_stage,
    "oos": run_universe_oos_stage,
}


def run_universe_pipeline(
    model,
    prices_by_symbol: dict[str, pd.Series],
    *,
    ledger=None,
    stages: tuple[str, ...] = ("backtest", "walk_forward", "oos"),
    cost_bps: float = 1.0,
    regime_frame: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    criteria: GraduationCriteria | None = None,
    wf_spec: WalkForwardSpec | None = None,
    oos_spec=None,
    as_of: str | None = None,
    stop_on_fail: bool = True,
) -> dict[str, StageResult]:
    """Walk a cross-sectional model through all three pipeline stages.

    Mirrors ``run_pipeline`` for single-asset models. Slices the universe dict by
    date for the OOS stage, so a single ``prices_by_symbol`` covering the full
    history is all that is needed. Returns ``{stage: StageResult}``.
    """
    if criteria is None:
        criteria = criteria_for_family(ledger.family) if ledger is not None else GraduationCriteria()
    common = dict(
        cost_bps=cost_bps, regime_frame=regime_frame,
        periods_per_year=periods_per_year, criteria=criteria,
    )
    results: dict[str, StageResult] = {}
    for stage in stages:
        if stage not in UNIVERSE_STAGE_RUNNERS:
            known = ", ".join(UNIVERSE_STAGE_RUNNERS)
            raise KeyError(f"unknown universe stage {stage!r}. Known: {known}")
        extra: dict = {}
        if stage == "walk_forward" and wf_spec:
            extra = {"spec": wf_spec}
        elif stage == "oos" and oos_spec:
            extra = {"spec": oos_spec}
        res = UNIVERSE_STAGE_RUNNERS[stage](model, prices_by_symbol, **common, **extra)
        if ledger is not None:
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
