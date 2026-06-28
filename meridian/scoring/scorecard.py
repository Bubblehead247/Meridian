"""Full per-strategy metric scorecard including regime splits and sleeve correlation.

This module owns the complete per-strategy metric set including regime splits and
sleeve correlation; it does NOT own raw metric formulas (reuses analytics/metrics).

The base risk/return block (CAGR, Sharpe, Sortino, Calmar, max drawdown, profit factor,
win rate, avg win/loss, trade stats) comes straight from
``analytics.metrics.performance_metrics``. This module *adds* the fields that the fund
scorecard needs on top: Ulcer Index, turnover, average holding period, slippage
sensitivity, regime-conditional splits (via ``regimes.labeler.attach_regimes``), and
correlation to each other active sleeve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.analytics.metrics import drawdown_series, max_drawdown, performance_metrics
from meridian.regimes.labeler import UNKNOWN, attach_regimes
from meridian.signals.backtest import PERIODS_PER_YEAR
from meridian.validation.stats import sharpe, total_return

_REGIME_DIMS = ("trend", "volatility", "breadth")
DEFAULT_SLIPPAGE_BPS = (1, 5, 10, 20)


def _ulcer_index(returns: pd.Series) -> float:
    """Ulcer Index — RMS of the percentage drawdown path (a depth-and-duration risk gauge).

    Lower is better. Expressed in percentage points (a 10% mean-square drawdown ≈ 10).
    """
    dd = drawdown_series(returns) * 100.0
    if len(dd) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(dd.to_numpy()))))


def _avg_holding_period(trades: pd.DataFrame | None) -> float:
    """Mean bars held per trade (NaN if no trades)."""
    if trades is None or len(trades) == 0 or "bars_held" not in trades:
        return float("nan")
    return float(trades["bars_held"].mean())


def _turnover(
    positions: pd.Series | None,
    trades: pd.DataFrame | None,
    *,
    n: int,
    periods_per_year: int,
) -> float:
    """Annualized two-sided turnover (units of notional traded per year).

    Exact from ``positions`` (Σ|Δposition| / years) when available; otherwise estimated
    from the trade count (each round trip ≈ 2 units of turnover).
    """
    years = n / periods_per_year if n else float("nan")
    if years in (0, float("nan")) or not years:
        return float("nan")
    if positions is not None and len(positions):
        held = positions.astype(float)
        traded = float(held.diff().abs().fillna(held.abs()).sum())
        return traded / years
    if trades is not None and len(trades):
        return 2.0 * len(trades) / years
    return float("nan")


def _slippage_sensitivity(
    returns: pd.Series,
    positions: pd.Series | None,
    *,
    bps_grid: tuple[int, ...],
    periods_per_year: int,
) -> dict[int, float]:
    """Sharpe after charging each extra ``bps`` of slippage on per-bar turnover.

    Shows how fast the edge erodes with worse fills. Needs ``positions`` to know the
    per-bar turnover; returns an empty dict when positions are unavailable.
    """
    if positions is None or not len(positions):
        return {}
    held = positions.astype(float)
    turn = held.diff().abs().fillna(held.abs())
    out: dict[int, float] = {}
    for bps in bps_grid:
        adj = returns - turn * (bps / 1e4)
        out[int(bps)] = sharpe(adj, periods_per_year)
    return out


def _compact_metrics(returns: pd.Series, periods_per_year: int) -> dict:
    """Small per-bucket block (used for regime splits)."""
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        return {"n_periods": 0}
    arr = r.to_numpy()
    return {
        "n_periods": int(r.size),
        "total_return": total_return(r),
        "sharpe": sharpe(r, periods_per_year),
        "max_drawdown": max_drawdown(arr),
        "hit_rate": float((arr > 0).mean()),
    }


def _regime_conditional(
    returns: pd.Series,
    regime_frame: pd.DataFrame | None,
    *,
    periods_per_year: int,
) -> dict[str, dict[str, dict]]:
    """Metrics recomputed within each bucket of each of the three regime dimensions.

    ``{dim: {label: compact_metrics}}`` for ``trend``/``volatility``/``breadth``.
    ``unknown`` (warm-up) buckets are skipped. Empty when no regime frame is supplied.
    """
    if regime_frame is None:
        return {}
    aligned = attach_regimes(returns.index, regime_frame)
    out: dict[str, dict[str, dict]] = {}
    for dim in _REGIME_DIMS:
        if dim not in aligned:
            continue
        labels = aligned[dim]
        buckets: dict[str, dict] = {}
        for label, grp in returns.groupby(labels.to_numpy()):
            if label == UNKNOWN:
                continue
            buckets[str(label)] = _compact_metrics(grp, periods_per_year)
        out[dim] = buckets
    return out


def _sleeve_correlation(
    returns: pd.Series, sleeve_returns: dict[str, pd.Series] | None
) -> dict[str, float]:
    """Pearson correlation of this strategy's returns to each other active sleeve."""
    if not sleeve_returns:
        return {}
    out: dict[str, float] = {}
    for name, other in sleeve_returns.items():
        joined = pd.concat([returns, other], axis=1, join="inner").dropna()
        out[name] = (
            float(joined.iloc[:, 0].corr(joined.iloc[:, 1])) if len(joined) > 1 else float("nan")
        )
    return out


def scorecard(
    returns: pd.Series,
    *,
    trades: pd.DataFrame | None = None,
    positions: pd.Series | None = None,
    regime_frame: pd.DataFrame | None = None,
    sleeve_returns: dict[str, pd.Series] | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
    slippage_bps: tuple[int, ...] = DEFAULT_SLIPPAGE_BPS,
) -> dict:
    """Full per-strategy scorecard.

    Args:
        returns: Per-bar net return series.
        trades: Optional trade ledger (for trade stats + holding period).
        positions: Optional held-position path (for exact turnover + slippage sensitivity).
        regime_frame: Optional labeler frame (``trend``/``volatility``/``breadth`` columns)
            for regime-conditional splits.
        sleeve_returns: Optional ``{sleeve: returns}`` for cross-sleeve correlation.
        periods_per_year / slippage_bps: annualization + the extra-cost grid.

    Returns:
        The base ``performance_metrics`` block plus ``ulcer_index``, ``turnover``,
        ``avg_holding_period``, ``slippage_sensitivity``, ``regime_conditional`` and
        ``sleeve_correlation``.
    """
    card = performance_metrics(returns, periods_per_year=periods_per_year, trades=trades)
    n = int(card.get("n_periods", len(pd.Series(returns).dropna())))
    card.update(
        {
            "ulcer_index": _ulcer_index(returns),
            "turnover": _turnover(positions, trades, n=n, periods_per_year=periods_per_year),
            "avg_holding_period": _avg_holding_period(trades),
            "slippage_sensitivity": _slippage_sensitivity(
                returns, positions, bps_grid=slippage_bps, periods_per_year=periods_per_year
            ),
            "regime_conditional": _regime_conditional(
                returns, regime_frame, periods_per_year=periods_per_year
            ),
            "sleeve_correlation": _sleeve_correlation(returns, sleeve_returns),
        }
    )
    return card


def scorecard_from_backtest(
    result,
    *,
    regime_frame: pd.DataFrame | None = None,
    sleeve_returns: dict[str, pd.Series] | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
    slippage_bps: tuple[int, ...] = DEFAULT_SLIPPAGE_BPS,
) -> dict:
    """Convenience: score a ``signals.backtest.BacktestResult`` directly."""
    return scorecard(
        result.returns,
        trades=result.trades,
        positions=result.positions,
        regime_frame=regime_frame,
        sleeve_returns=sleeve_returns,
        periods_per_year=periods_per_year,
        slippage_bps=slippage_bps,
    )


def _avg_holding_period_cs(weights: pd.DataFrame) -> float:
    """Mean bars held per rotation position (non-cash runs) in a cross-sectional model."""
    if weights is None or weights.empty:
        return float("nan")
    held = weights.idxmax(axis=1).where(weights.max(axis=1) > 1e-10)
    held_str = held.fillna("__cash__")
    transitions = held_str.ne(held_str.shift()).fillna(True)
    run_id = transitions.cumsum()
    periods = []
    for rid in run_id.unique():
        grp = held[run_id == rid]
        if len(grp) and not pd.isna(grp.iloc[0]):
            periods.append(len(grp))
    return float(np.mean(periods)) if periods else float("nan")


def scorecard_from_portfolio(
    result,
    *,
    index=None,
    regime_frame: pd.DataFrame | None = None,
    sleeve_returns: dict[str, pd.Series] | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
    slippage_bps: tuple[int, ...] = DEFAULT_SLIPPAGE_BPS,
) -> dict:
    """Score a ``PortfolioResult``; restrict to ``index`` for walk-forward / OOS windows.

    Derives turnover, n_trades, avg_holding_period from the portfolio's weight frame and
    pre-computed ``turnover_series`` — metrics that ``scorecard_from_backtest`` cannot
    compute for cross-sectional models.
    """
    if index is not None:
        returns = result.returns.reindex(index).fillna(0.0)
        turn = result.turnover_series.reindex(index).fillna(0.0)
        weights = (
            result.weights.reindex(index)
            if isinstance(result.weights, pd.DataFrame)
            else None
        )
    else:
        returns = result.returns
        turn = result.turnover_series
        weights = result.weights if isinstance(result.weights, pd.DataFrame) else None

    card = performance_metrics(returns, periods_per_year=periods_per_year)
    n = len(returns)
    years = n / periods_per_year if n else float("nan")

    ann_turn = float(turn.sum() / years) if years and years > 0 else float("nan")
    n_trades_val = int((turn > 0).sum()) if len(turn) else 0
    card["n_trades"] = n_trades_val

    slip: dict[int, float] = {}
    if len(turn) and len(returns):
        for bps in slippage_bps:
            adj = returns - turn * (bps / 1e4)
            slip[int(bps)] = sharpe(adj, periods_per_year)

    card.update(
        {
            "ulcer_index": _ulcer_index(returns),
            "turnover": ann_turn,
            "avg_holding_period": _avg_holding_period_cs(weights),
            "slippage_sensitivity": slip,
            "regime_conditional": _regime_conditional(
                returns, regime_frame, periods_per_year=periods_per_year
            ),
            "sleeve_correlation": _sleeve_correlation(returns, sleeve_returns),
        }
    )
    return card
