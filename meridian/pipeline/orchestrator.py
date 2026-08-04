"""Top-level driver that walks a model through the pipeline stages.

This module owns sequencing a model through the stage runners and (optionally) advancing
its ledger; it does NOT own the stage math (reuses pipeline/backtest|walk_forward|oos).

``run_pipeline`` runs the chosen stages in order (backtest → walk_forward → oos),
advancing the strategy's graduation stage from each stage's scorecard and stopping early
when a stage fails — the single call that ties model → pipeline → ledger/graduation
together for one strategy on one price series.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from meridian.families.base import Model
from meridian.pipeline.backtest import StageResult, run_backtest_stage
from meridian.pipeline.graduation import (
    GraduationCriteria,
    advance,
    criteria_for_family,
    passes_metric_bar,
)
from meridian.pipeline.oos import compare_oos_to_is, run_oos_stage
from meridian.pipeline.oos_guard import OOSGuard
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
    oos_guard: OOSGuard | None = None,
    symbol: str | None = None,
) -> dict[str, StageResult]:
    """Walk ``model`` through ``stages`` in order; advance ``ledger`` from each scorecard.

    Returns ``{stage: StageResult}``. When a ``ledger`` is given, each stage's scorecard is
    fed to ``graduation.advance`` (the resulting action is stored on the StageResult's
    ``detail['ledger_action']``). With ``stop_on_fail`` the walk halts at the first stage
    that does not clear the metric bar. ``oos_guard``/``symbol`` are forwarded to the OOS
    stage's run-counter (see ``pipeline/oos_guard.py``); both optional and non-blocking.
    """
    if criteria is None:
        criteria = (
            criteria_for_family(ledger.family) if ledger is not None else GraduationCriteria()
        )
    common = dict(
        cost_bps=cost_bps, bars=bars, regime_frame=regime_frame,
        periods_per_year=periods_per_year, criteria=criteria,
    )
    runners = {
        "backtest": (run_backtest_stage, {}),
        "walk_forward": (run_walk_forward_stage, {"spec": wf_spec} if wf_spec else {}),
        "oos": (
            run_oos_stage,
            {**({"spec": oos_spec} if oos_spec else {}), "guard": oos_guard, "symbol": symbol},
        ),
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


def run_cross_sectional_pipeline(
    model,
    prices_by_symbol: dict[str, pd.Series],
    *,
    ledger: StrategyLedger | None = None,
    stages: tuple[str, ...] = DEFAULT_STAGES,
    cost_bps: float = 1.0,
    periods_per_year: int = 252,
    criteria: GraduationCriteria | None = None,
    wf_spec=None,
    oos_spec=None,
    as_of: str | None = None,
    stop_on_fail: bool = True,
) -> dict[str, StageResult]:
    """Walk a CrossSectionalModel through pipeline stages on a basket of price series.

    Parallels ``run_pipeline`` but accepts a ``{symbol: prices}`` dict instead of a
    single-asset Series.  The model is backtested once over full history (causal, no
    lookahead) and the walk-forward / OOS windows are sliced from that one result, so
    no stage re-runs the model — the same pattern as single-asset anchored walk-forward.

    Returns ``{stage: StageResult}`` with proper turnover, n_trades, and
    avg_holding_period metrics (derived from the portfolio weight frame).
    """
    from meridian.data.splits import SplitSpec, split
    from meridian.scoring import scorecard_from_portfolio
    from meridian.validation.walkforward import WalkForwardSpec, make_folds

    if criteria is None:
        criteria = (
            criteria_for_family(ledger.family) if ledger is not None else GraduationCriteria()
        )

    result = model.backtest(prices_by_symbol, cost_bps=cost_bps)
    n = len(result.returns)

    def _score(index) -> dict:
        return scorecard_from_portfolio(result, index=index, periods_per_year=periods_per_year)

    def _stage_result(stage_name: str, card: dict, passed: bool, **detail) -> StageResult:
        res = StageResult(
            stage=stage_name, model=model.name, scorecard=card, passed=passed, detail=detail
        )
        if ledger is not None:
            adv_criteria = (
                dataclasses.replace(criteria, min_periods=0)
                if stage_name in ("walk_forward", "oos") else criteria
            )
            res.detail["ledger_action"] = advance(ledger, card, as_of=as_of, criteria=adv_criteria)
        return res

    results: dict[str, StageResult] = {}
    for stage in stages:
        if stage == "backtest":
            card = _score(result.returns.index)
            res = _stage_result("backtest", card, passes_metric_bar(card, criteria), n_periods=n)

        elif stage == "walk_forward":
            spec = wf_spec or WalkForwardSpec()
            folds = make_folds(n, spec)
            if not folds:
                res = _stage_result("walk_forward", {}, False, n_folds=0)
            else:
                oos_idx = result.returns.index[
                    [i for f in folds for i in range(f.test_start, f.test_end)]
                ]
                card = _score(oos_idx)
                wf_crit = dataclasses.replace(criteria, min_periods=0)
                res = _stage_result(
                    "walk_forward", card, passes_metric_bar(card, wf_crit),
                    n_folds=len(folds), oos_periods=len(oos_idx),
                )

        elif stage == "oos":
            spec = oos_spec or SplitSpec()
            parts = split(result.returns, spec)
            oos_ret = parts.get("out_of_sample", pd.Series(dtype=float))
            is_ret = parts.get("in_sample", pd.Series(dtype=float))
            if len(oos_ret) == 0:
                res = _stage_result("oos", {}, False, n_oos=0)
            else:
                oos_card = _score(oos_ret.index)
                oos_crit = dataclasses.replace(criteria, min_periods=0)
                comparison = {"degraded": False}
                if len(is_ret):
                    is_card = _score(is_ret.index)
                    comparison = compare_oos_to_is(oos_card, is_card)
                passed = passes_metric_bar(oos_card, oos_crit) and not comparison["degraded"]
                res = _stage_result("oos", oos_card, passed, n_oos=len(oos_ret), **comparison)

        else:
            raise KeyError(f"unknown stage {stage!r}. Known: backtest, walk_forward, oos")

        results[stage] = res
        if stop_on_fail and not res.passed:
            break

    return results
