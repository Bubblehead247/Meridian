"""Transaction-cost stress testing and capacity/liquidity analysis (P2-A).

This module owns two related but distinct checks the platform previously had no way to
run:

1. **Cost-stress sweep** — re-run a backtest at several flat cost levels (0/5/10/25/50/
   100 bps) and see whether the conclusion (sign, significance, ranking) survives. This
   is deliberately the *simple* first step the audit brief calls for, not a market-impact
   model — see ``research_integrity_gap_analysis.md`` §3.9/P2-A.
2. **Capacity/participation analysis** — given real bar volume, how much capital can a
   strategy deploy before its trades would represent an implausible share of a day's
   volume? This does NOT synthesize a capacity-adjusted Sharpe/return number from an
   unvalidated impact-cost formula (real market impact is closer to a square-root-of-
   participation law, not linear, and calibrating either requires data this platform
   doesn't have). Instead it reports participation rate directly — the same "at minimum"
   metric the brief asks for — so a reviewer can see exactly when a backtest's
   frictionless-fill assumption stops being plausible, without being handed a fabricated
   precision it can't back up. A calibrated impact model is future work, deliberately not
   attempted here (Rule 4/5 in the audit brief: don't fabricate certainty, don't implement
   unverified formulas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The audit brief's specified cost-stress grid.
DEFAULT_COST_GRID: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0, 50.0, 100.0)

#: The audit brief's specified participation-level grid, as fractions of ADV.
DEFAULT_PARTICIPATION_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25)


# --- cost-stress sweep -------------------------------------------------------

def cost_stress_sweep(
    prices: pd.Series,
    positions: pd.Series,
    *,
    bars: pd.DataFrame | None = None,
    bps_grid: tuple[float, ...] = DEFAULT_COST_GRID,
) -> pd.DataFrame:
    """Re-run the single-asset backtest at each cost level in ``bps_grid``.

    Returns one row per cost level: ``cost_bps``, ``sharpe``, ``total_return``,
    ``max_drawdown``, ``n_trades``. Reuses ``signals.backtest.backtest`` unmodified —
    this is a sweep over its existing ``cost_bps`` parameter, not a new cost model.
    """
    from meridian.signals.backtest import backtest

    rows = []
    for bps in bps_grid:
        res = backtest(prices, positions, cost_bps=bps, bars=bars)
        s = res.summary()
        rows.append({
            "cost_bps": bps, "sharpe": s["sharpe"], "total_return": s["total_return"],
            "max_drawdown": s["max_drawdown"], "n_trades": s["n_trades"],
        })
    return pd.DataFrame(rows)


def cost_stress_sweep_cross_sectional(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    open_prices: pd.DataFrame | None = None,
    sizing: str = "equal_weight",
    lookback: int = 20,
    bps_grid: tuple[float, ...] = DEFAULT_COST_GRID,
) -> pd.DataFrame:
    """Cross-sectional counterpart of ``cost_stress_sweep``, via ``backtest_portfolio``."""
    from meridian.portfolio.portfolio import backtest_portfolio
    from meridian.validation.stats import sharpe as _sharpe, total_return

    rows = []
    for bps in bps_grid:
        res = backtest_portfolio(
            signals, prices, sizing=sizing, cost_bps=bps, lookback=lookback,
            open_prices=open_prices,
        )
        running_max = res.equity.cummax()
        max_dd = float((res.equity / running_max - 1.0).min()) if len(res.equity) else 0.0
        rows.append({
            "cost_bps": bps, "sharpe": _sharpe(res.returns),
            "total_return": total_return(res.returns), "max_drawdown": max_dd,
        })
    return pd.DataFrame(rows)


def cost_stress_conclusion_stable(sweep: pd.DataFrame, *, sign_only: bool = True) -> bool:
    """Whether the Sharpe sign (or, with ``sign_only=False``, positivity) holds across the
    whole grid — the "does the conclusion survive realistic costs" check the brief asks
    for. NaN Sharpes (e.g. a degenerate zero-turnover run) are ignored, not treated as a
    flip.
    """
    s = sweep["sharpe"].dropna()
    if s.empty:
        return False
    if sign_only:
        return bool((s > 0).all() or (s < 0).all())
    return bool((s > 0).all())


# --- capacity / participation ------------------------------------------------

def average_daily_volume(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling mean traded volume — the ADV denominator for participation rate."""
    return bars["volume"].rolling(window).mean()


def implied_shares_traded(
    capital: float, prices: pd.Series, positions: pd.Series
) -> pd.Series:
    """Shares bought/sold each bar a ``capital``-sized account would need, to move its
    position from one bar's target weight to the next. ``positions`` is the {-1,0,+1}
    (or fractional weight) exposure path *before* the 1-bar lag ``backtest`` applies —
    the turnover between consecutive target positions is what actually gets traded.
    """
    turnover = positions.astype(float).diff().abs()
    turnover.iloc[0] = abs(float(positions.iloc[0])) if len(positions) else 0.0
    return capital * turnover / prices.astype(float)


def participation_rate(
    capital: float, prices: pd.Series, positions: pd.Series, bars: pd.DataFrame,
    *, adv_window: int = 20,
) -> pd.Series:
    """Fraction of a day's ADV a ``capital``-sized account's trade would represent.

    Only meaningful on bars where a trade actually occurs (turnover > 0); other bars are
    0/NaN and should be excluded from any "worst case" read (use ``.dropna()`` or filter
    ``!= 0`` first — see ``max_capacity``).
    """
    shares = implied_shares_traded(capital, prices, positions)
    adv = average_daily_volume(bars, adv_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = shares / adv
    return rate.replace([np.inf, -np.inf], np.nan)


def max_capacity(
    prices: pd.Series, positions: pd.Series, bars: pd.DataFrame,
    *, participation_limit: float = 0.10, adv_window: int = 20,
) -> float:
    """The largest capital allocation that keeps every trade's participation rate at or
    below ``participation_limit``, i.e. the binding constraint is the single worst
    (lowest-ADV-relative-to-trade-size) turnover day, not the average one — capacity is
    set by the day you can't execute, not the typical day.

    Returns ``inf`` if the strategy never trades (no turnover -> no capacity constraint
    from this analysis) and ``0.0`` if any turnover day has zero/unknown ADV (can't size
    into an illiquid name at all under this constraint).
    """
    turnover = positions.astype(float).diff().abs()
    turnover.iloc[0] = abs(float(positions.iloc[0])) if len(positions) else 0.0
    adv = average_daily_volume(bars, adv_window)
    trading_days = turnover > 0
    if not trading_days.any():
        return float("inf")
    shares_per_dollar = (turnover / prices.astype(float))[trading_days]
    adv_on_trading_days = adv[trading_days]
    if (adv_on_trading_days <= 0).any() or adv_on_trading_days.isna().any():
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        capital_bound = participation_limit * adv_on_trading_days / shares_per_dollar
    capital_bound = capital_bound.replace([np.inf, -np.inf], np.nan).dropna()
    if capital_bound.empty:
        return float("inf")
    return float(capital_bound.min())


def capacity_stress_sweep(
    prices: pd.Series, positions: pd.Series, bars: pd.DataFrame,
    *, capital_levels: tuple[float, ...], adv_window: int = 20,
) -> pd.DataFrame:
    """Participation rate reached at each capital level — the "degradation as capital
    increases" evidence the brief asks for, reported honestly (as participation, not as
    a fabricated capacity-adjusted P&L) — see the module docstring for why.

    Returns one row per capital level: ``capital``, ``worst_participation`` (the single
    highest trade-day participation rate — the binding constraint), ``mean_participation``
    (average over trading days only), ``pct_trading_days_over_10pct`` (share of trades
    that would represent more than 10% of that day's ADV — a common liquidity-risk rule
    of thumb, not a hard limit).
    """
    rows = []
    for capital in capital_levels:
        rate = participation_rate(capital, prices, positions, bars, adv_window=adv_window)
        turnover = positions.astype(float).diff().abs()
        turnover.iloc[0] = abs(float(positions.iloc[0])) if len(positions) else 0.0
        traded = rate[turnover > 0].dropna()
        if traded.empty:
            rows.append({
                "capital": capital, "worst_participation": float("nan"),
                "mean_participation": float("nan"), "pct_trading_days_over_10pct": float("nan"),
            })
            continue
        rows.append({
            "capital": capital,
            "worst_participation": float(traded.max()),
            "mean_participation": float(traded.mean()),
            "pct_trading_days_over_10pct": float((traded > 0.10).mean()),
        })
    return pd.DataFrame(rows)
