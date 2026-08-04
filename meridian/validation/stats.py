"""Small return-statistics helpers shared across the validation engine.

Kept deliberately minimal — the full performance-analytics suite is Phase 7.
These are just the few numbers the validation math needs (Sharpe, total return,
and the per-bar net return of a position path).
"""

from __future__ import annotations

import numpy as np

from meridian.analytics.metrics import sharpe_ratio
from meridian.signals.backtest import PERIODS_PER_YEAR


def sharpe(returns, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized Sharpe ratio of a per-bar return series (NaN if degenerate).

    Thin wrapper around ``analytics.metrics.sharpe_ratio`` at ``risk_free=0.0``
    (this module never adjusted for a risk-free rate) — the single source of
    truth for the formula lives there now, so it can't drift into two
    different answers across the two modules.
    """
    return sharpe_ratio(returns, risk_free=0.0, periods_per_year=periods_per_year)


def total_return(returns) -> float:
    """Compounded total return of a per-bar return series."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return 0.0
    return float(np.prod(1.0 + r) - 1.0)


def strategy_net(held, market_returns, cost_bps: float = 0.0) -> np.ndarray:
    """Net per-bar returns of a *held* (already-lagged) position path.

    ``held[t]`` is the position carried into bar t, earning ``market_returns[t]``.
    Costs are charged on the change in held position. This mirrors the Phase 4
    backtester exactly, but on plain arrays so Monte-Carlo can call it fast.
    """
    h = np.asarray(held, dtype=float)
    ret = np.asarray(market_returns, dtype=float)
    gross = h * ret
    turnover = np.abs(np.diff(h, prepend=0.0))
    cost = turnover * (cost_bps / 1e4)
    return gross - cost
