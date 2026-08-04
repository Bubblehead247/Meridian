"""Build the monthly review from live paper-stage records.

This module is the bridge between the saved-strategy records (backtest
scorecards) and the MonthlyReview / SleeveAction machinery in
portfolio/monthly_review.py.

Since paper trading just started, realized P&L is zero. The review
uses backtest scorecard metrics to populate per-sleeve scorecards so
that graduation checks and regime compliance can still run.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import pandas as pd

from meridian.data import load_ohlcv
from meridian.pipeline.records import load_records
from meridian.portfolio.allocation import SLEEVE_ALLOCATIONS, seed_ledgers
from meridian.portfolio.monthly_review import MonthlyReview, run_monthly_review
from meridian.reporting.monthly_report import render_monthly_report


def _benchmark_return(lookback_days: int = 30) -> float:
    """SPY total return over the last ``lookback_days`` calendar days.

    lookback_days is calendar days; we fetch 3× that many to ensure enough
    trading bars, then compare the last bar to the bar closest to the target
    date rather than a fixed negative index.
    """
    try:
        target = date.today() - timedelta(days=lookback_days)
        start = (target - timedelta(days=10)).isoformat()
        prices = load_ohlcv("SPY", start)["close"].dropna()
        if len(prices) < 2:
            return 0.0
        # Find the first bar on or after the target date
        target_str = target.isoformat()
        idx = prices.index.astype(str)
        past = prices[idx <= target_str]
        ref = past.iloc[-1] if not past.empty else prices.iloc[0]
        return float(prices.iloc[-1] / ref - 1)
    except Exception:
        return 0.0


def _aggregate_scorecard(records: list) -> dict:
    """Collapse multiple strategy records into one sleeve-level scorecard.

    Takes the average Sharpe/CAGR and the worst (most negative) max drawdown.
    """
    if not records:
        return {}
    sharpes = [r.scorecard.get("sharpe") or 0.0 for r in records]
    cagrs   = [r.scorecard.get("cagr")   or 0.0 for r in records]
    dds     = [r.scorecard.get("max_drawdown") or 0.0 for r in records]
    return {
        "sharpe":       sum(sharpes) / len(sharpes),
        "cagr":         sum(cagrs)   / len(cagrs),
        "max_drawdown": min(dds),     # worst across strategies in the sleeve
        "n_strategies": len(records),
    }


def _sleeve_returns_from_records(
    paper_records: list, price_start: str = "2023-01-01"
) -> dict[str, pd.Series]:
    """Reconstruct one return series per family for the correlation/MCTR blocks.

    For each family, re-backtests the best-Sharpe *single-asset* record's
    model on its saved symbol (cached prices — this reconstructs a historical
    view for reporting, not a live trading decision, so the on-disk cache is
    fine here unlike ``_fetch_prices``'s live path). Cross-sectional records
    are skipped: they need a basket, not one symbol, to backtest — same
    constraint as ``run_cs_fund``'s callers that don't have OHLC baskets handy.
    Failures for one family (bad/missing cached data, model errors) are
    swallowed so one broken record doesn't blank out the whole review.
    """
    from meridian.families import create_model

    by_family: dict[str, list] = defaultdict(list)
    for r in paper_records:
        by_family[r.family].append(r)

    out: dict[str, pd.Series] = {}
    for family, recs in by_family.items():
        candidates = sorted(recs, key=lambda r: r.scorecard.get("sharpe") or float("-inf"), reverse=True)
        for rec in candidates:
            try:
                model = create_model(family, rec.model)
                if getattr(model, "cross_sectional", False):
                    continue
                frame = load_ohlcv(rec.symbol, price_start)
                result = model.backtest(frame["close"], bars=frame)
                out[family] = result.returns
                break
            except Exception:
                continue
    return out


def _dry_run_decisions(equity: float = 10_000.0, price_start: str = "2023-01-01") -> list:
    """Run today's live picks in dry-run mode (no orders); return the decisions."""
    from meridian.execution.broker import SimulatedBroker
    from meridian.execution.live_runner import run_paper_session

    broker = SimulatedBroker(cash=equity)
    return run_paper_session(broker, account_equity=equity,
                             price_start=price_start, dry_run=True)


def _today_signals_brief(decisions: list) -> dict[str, dict]:
    """Return {family: {model/symbol: {sym: signal}}} for reporting."""
    result: dict[str, dict] = defaultdict(dict)
    for d in decisions:
        if not d.skipped:
            key = f"{d.model}/{d.symbol[:30]}"
            result[d.family][key] = d.signals
    return dict(result)


def _render_sector_exposure(decisions: list) -> str:
    """Markdown section: look-through sector exposure of today's held positions."""
    from meridian.portfolio.sectors import (
        DEFAULT_SECTOR_LIMIT,
        flag_concentration,
        lookthrough_exposures,
        position_weights_from_decisions,
    )

    weights = position_weights_from_decisions(decisions)
    exposures = lookthrough_exposures(weights)
    flagged = set(flag_concentration(exposures))

    lines = ["## Sector exposure (look-through)", ""]
    if not exposures:
        lines.append("No open positions.")
        return "\n".join(lines)

    lines.append("| Sector | % of account | |")
    lines.append("|---|---:|---|")
    for sector in sorted(exposures, key=exposures.get, reverse=True):
        mark = "⚠️ over limit" if sector in flagged else ""
        lines.append(f"| {sector} | {exposures[sector]:.1%} | {mark} |")
    lines.append("")
    lines.append(
        f"Concentration limit: {DEFAULT_SECTOR_LIMIT:.0%} of account per sector. "
        "ETF exposure is looked through to sectors using static approximate weights "
        "(see `portfolio/sectors.py`)."
    )
    return "\n".join(lines)


def _current_regime():
    """Fetch today's regime label, or None if unavailable or fully unknown.

    Returns None when all three dimensions are unknown so that the permissions
    check treats all families as permitted (conservative for reporting — unknown
    regime should not flag sleeves as non-compliant in the review output).
    """
    try:
        from meridian.regimes.labeler import RegimeLabel, build_regime_frame
        frame = build_regime_frame("2024-01-01")
        if frame.empty:
            return None
        row = frame.iloc[-1]
        trend     = row.get("trend", "unknown")
        volatility = row.get("volatility", "unknown")
        breadth   = row.get("breadth", "unknown")
        if trend == "unknown" and breadth == "unknown":
            return None  # not enough data — skip permission gating in review
        return RegimeLabel(trend=trend, volatility=volatility, breadth=breadth)
    except Exception:
        return None


def _render_strategy_inventory(paper_records: list, signals_by_family: dict) -> str:
    """Markdown section listing every paper strategy with its scorecard and today's signal."""
    by_family: dict[str, list] = defaultdict(list)
    for r in paper_records:
        by_family[r.family].append(r)

    lines = ["## Strategy inventory", ""]
    for family in sorted(by_family):
        alloc = SLEEVE_ALLOCATIONS.get(family, 0.0)
        lines.append(f"### {family}  ({alloc:.0%} sleeve)")
        lines.append("")
        lines.append("| Strategy | Symbol | Sharpe | CAGR | MaxDD | Signal |")
        lines.append("|---|---|---:|---:|---:|---|")
        fam_signals = signals_by_family.get(family, {})
        for r in sorted(by_family[family], key=lambda x: x.model):
            sc = r.scorecard
            sharpe = f"{sc.get('sharpe', 0):.2f}" if sc.get("sharpe") is not None else "—"
            cagr   = f"{sc.get('cagr', 0):.1%}"   if sc.get("cagr")   is not None else "—"
            mdd    = sc.get("max_drawdown")
            dd     = f"{mdd:.1%}" if mdd is not None else "—"
            key = f"{r.model}/{r.symbol[:30]}"
            sigs = fam_signals.get(key, {})
            long_syms = [s for s, v in sigs.items() if v > 0]
            sig_str = ", ".join(long_syms) if long_syms else "flat"
            sym_short = r.symbol[:35] + ("…" if len(r.symbol) > 35 else "")
            lines.append(f"| {r.model} | {sym_short} | {sharpe} | {cagr} | {dd} | {sig_str} |")
        lines.append("")

    return "\n".join(lines)


def build_paper_review(
    *,
    equity: float = 10_000.0,
    price_start: str = "2023-01-01",
    lookback_days: int = 30,
) -> tuple[MonthlyReview, str]:
    """Build the monthly review for all paper-stage strategies.

    Returns (MonthlyReview, extra_markdown) where extra_markdown is the
    strategy inventory section to append after the standard report.
    """
    paper_records = [r for r in load_records() if r.stage_passed == "paper"]

    # Group by family for per-sleeve scorecard aggregation
    by_family: dict[str, list] = defaultdict(list)
    for r in paper_records:
        by_family[r.family].append(r)

    scorecards = {fam: _aggregate_scorecard(recs) for fam, recs in by_family.items()}
    ledgers    = seed_ledgers(equity, stage="paper")

    benchmark_ret  = _benchmark_return(lookback_days)
    regime         = _current_regime()
    decisions      = _dry_run_decisions(equity=equity, price_start=price_start)
    signals        = _today_signals_brief(decisions)
    sleeve_returns = _sleeve_returns_from_records(paper_records, price_start=price_start)

    review = run_monthly_review(
        ledgers,
        scorecards,
        account_equity=equity,
        benchmark_return=benchmark_ret,
        current_regime=regime,
        sleeve_returns=sleeve_returns,
    )

    sector_md    = _render_sector_exposure(decisions)
    inventory_md = _render_strategy_inventory(paper_records, signals)
    return review, sector_md + "\n\n" + inventory_md


def render_full_review(review: MonthlyReview, inventory_md: str) -> str:
    """Combine the standard monthly report with the strategy inventory section."""
    base = render_monthly_report(review)
    note = (
        "\n> **Note:** realized P&L is zero — paper trading started "
        f"{date.today().isoformat()}. Scorecard metrics are from backtests.\n"
    )
    return base + note + "\n" + inventory_md
