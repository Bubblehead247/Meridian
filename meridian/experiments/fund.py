"""Reusable fund-lifecycle runner (shared by the CLI and the interactive menu).

Runs the whole-fund lifecycle for one underlying: seed the 8 sleeves, drive a representative
single-asset model per family through the pipeline (advancing each ledger's graduation
stage), and produce the month-end review. This module owns the orchestration only — it does
no network I/O (callers pass loaded prices) and no rendering (callers render/print the
review), so it is offline-testable and used by both ``cli`` and ``interactive``.
"""

from __future__ import annotations

import pandas as pd

from meridian.families import create_model, list_models
from meridian.pipeline import run_pipeline
from meridian.pipeline.records import load_records
from meridian.portfolio import MonthlyReview, StrategyLedger, run_monthly_review, seed_ledgers
from meridian.portfolio.ledger import STAGES

_STAGE_RANK: dict[str, int] = {s: i for i, s in enumerate(STAGES)}


def _current_regime(regime_frame):
    """Extract the most recent RegimeLabel from a regime frame, or None."""
    if regime_frame is None or regime_frame.empty:
        return None
    from meridian.regimes.labeler import RegimeLabel
    row = regime_frame.iloc[-1]
    return RegimeLabel(
        trend=row.get("trend", "unknown"),
        volatility=row.get("volatility", "unknown"),
        breadth=row.get("breadth", "unknown"),
    )


def _first_single_asset(family: str) -> str | None:
    for name in list_models(family):
        if not getattr(create_model(family, name), "cross_sectional", False):
            return name
    return None


def _best_single_asset(family: str, symbol: str | None) -> str | None:
    """Best model for this family: highest stage, then highest Sharpe from saved records.

    Falls back to first registered when no saved record exists for the symbol.
    """
    if symbol:
        candidates = [
            r for r in load_records()
            if r.family == family and r.symbol == symbol
            and not getattr(create_model(family, r.model), "cross_sectional", False)
        ]
        if candidates:
            best = max(candidates, key=lambda r: (
                _STAGE_RANK.get(r.stage_passed, 0),
                r.scorecard.get("sharpe") or 0.0,
            ))
            return best.model
    return _first_single_asset(family)


def _first_cs_model(family: str) -> str | None:
    for name in list_models(family):
        if getattr(create_model(family, name), "cross_sectional", False):
            return name
    return None


def run_fund(
    prices: pd.Series,
    bars: pd.DataFrame | None = None,
    *,
    symbol: str | None = None,
    equity: float = 100_000.0,
    cost_bps: float = 1.0,
    regime_frame: pd.DataFrame | None = None,
) -> tuple[list[StrategyLedger], MonthlyReview]:
    """Run the fund lifecycle on one price series; return (ledgers, month-end review).

    Sleeves whose family has no single-asset model (cash_reserve, experimental_research) are
    held at their seeded research stage. When ``regime_frame`` is supplied, each model's
    positions are gated to its permission rule before scoring.
    """
    ledgers = seed_ledgers(equity)
    scorecards: dict[str, dict] = {}
    sleeve_returns: dict[str, pd.Series] = {}
    for led in ledgers:
        model_name = _best_single_asset(led.family, symbol)
        if model_name is None:
            continue
        model = create_model(led.family, model_name)
        results = run_pipeline(
            model, prices, ledger=led, bars=bars, cost_bps=cost_bps,
            regime_frame=regime_frame, symbol=symbol,
        )
        scorecards[led.name] = list(results.values())[-1].scorecard
        # Same backtest run_pipeline's backtest stage already performed; kept here
        # purely for its return series, which StageResult doesn't carry — that
        # series is what makes inter-sleeve correlation (below) real instead of
        # silently empty.
        sleeve_returns[led.name] = model.backtest(
            prices, cost_bps=cost_bps, bars=bars, regime_frame=regime_frame
        ).returns

    current_regime = _current_regime(regime_frame)
    review = run_monthly_review(
        ledgers, scorecards, account_equity=equity, current_regime=current_regime,
        sleeve_returns=sleeve_returns,
    )
    return ledgers, review


def run_cs_fund(
    prices_by_symbol: dict[str, pd.Series],
    *,
    equity: float = 100_000.0,
    cost_bps: float = 1.0,
    regime_frame: pd.DataFrame | None = None,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> tuple[list[StrategyLedger], MonthlyReview]:
    """Fund lifecycle on a cross-sectional basket; only CS models from each family run.

    Families with no CS model (mean_reversion, trend_following, breakouts, pullback,
    long_term_etf, cash_reserve, experimental_research) stay at their seeded stage.
    Currently only momentum and sector_rotation register CS models.

    ``bars_by_symbol`` (optional OHLC per symbol) enables next-open fills — see
    ``CrossSectionalModel.backtest``. Without it, fills fall back to the same
    close the signal was computed from.
    """
    from meridian.pipeline import run_universe_pipeline

    ledgers = seed_ledgers(equity)
    scorecards: dict[str, dict] = {}
    sleeve_returns: dict[str, pd.Series] = {}
    basket = "_".join(sorted(prices_by_symbol))
    for led in ledgers:
        cs_name = _first_cs_model(led.family)
        if cs_name is None:
            continue
        model = create_model(led.family, cs_name)
        results = run_universe_pipeline(
            model, prices_by_symbol, ledger=led, cost_bps=cost_bps,
            bars_by_symbol=bars_by_symbol, regime_frame=regime_frame, basket=basket,
        )
        if results:
            scorecards[led.name] = list(results.values())[-1].scorecard
            sleeve_returns[led.name] = model.backtest(
                prices_by_symbol, cost_bps=cost_bps, bars_by_symbol=bars_by_symbol
            ).returns
    current_regime = _current_regime(regime_frame)
    review = run_monthly_review(
        ledgers, scorecards, account_equity=equity, current_regime=current_regime,
        sleeve_returns=sleeve_returns,
    )
    return ledgers, review
