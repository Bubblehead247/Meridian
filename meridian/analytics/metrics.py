"""Performance metrics.

Turns a per-bar return series into the full set of risk and return numbers a
research report needs: compounded and annualized return, volatility, the
risk-adjusted ratios (Sharpe, Sortino, Calmar), drawdown, tail statistics, and
bar/trade hit rates. Phase 6 used a single Sharpe for significance testing; this
is the richer descriptive layer for reporting.

All functions are NaN-safe and operate on plain return series so they compose
with any `BacktestResult` or stitched walk-forward curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sps

from meridian.signals.backtest import PERIODS_PER_YEAR


def equity_curve(returns) -> pd.Series:
    """Cumulative equity (starts at 1.0) from per-bar returns."""
    r = pd.Series(returns, dtype=float).fillna(0.0)
    return (1.0 + r).cumprod()


def drawdown_series(returns) -> pd.Series:
    """Drawdown at each bar: equity / running peak - 1 (<= 0)."""
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


def max_drawdown(returns) -> float:
    """Worst peak-to-trough decline (a negative number, or 0.0)."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else 0.0


def max_drawdown_duration(returns) -> int:
    """Longest run of bars spent below a prior equity peak."""
    eq = equity_curve(returns)
    peak = eq.cummax()
    underwater = eq < peak
    longest = run = 0
    for u in underwater.to_numpy():
        run = run + 1 if u else 0
        longest = max(longest, run)
    return int(longest)


def performance_metrics(
    returns,
    *,
    periods_per_year: int = PERIODS_PER_YEAR,
    risk_free: float = 0.0,
    trades: pd.DataFrame | None = None,
) -> dict:
    """Compute a full metrics dict from a per-bar return series.

    Args:
        returns: Per-bar (net) returns.
        periods_per_year: Annualization factor (252 for daily).
        risk_free: Annual risk-free rate, converted to per-bar for the ratios.
        trades: Optional trade ledger (from `BacktestResult.trades`) for
            trade-level stats (win rate, profit factor, expectancy).

    Returns:
        Dict of metrics. Degenerate inputs yield NaN rather than raising.
    """
    r = pd.Series(returns, dtype=float).dropna()
    n = len(r)
    rf_bar = risk_free / periods_per_year

    out: dict[str, float] = {"n_periods": n}
    if n == 0:
        return out

    arr = r.to_numpy()
    total = float(np.prod(1.0 + arr) - 1.0)
    years = n / periods_per_year
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else float("nan")
    vol = float(arr.std(ddof=0))
    ann_vol = vol * np.sqrt(periods_per_year)

    excess = arr - rf_bar
    sharpe = float(excess.mean() / vol * np.sqrt(periods_per_year)) if vol > 0 else float("nan")
    downside = arr[arr < 0]
    dstd = float(downside.std(ddof=0)) if downside.size else 0.0
    sortino = (
        float(excess.mean() / dstd * np.sqrt(periods_per_year)) if dstd > 0 else float("nan")
    )
    mdd = max_drawdown(arr)
    calmar = float(cagr / abs(mdd)) if mdd < 0 and not np.isnan(cagr) else float("nan")

    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())

    out.update(
        {
            "total_return": total,
            "cagr": cagr,
            "ann_volatility": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": mdd,
            "max_drawdown_duration": max_drawdown_duration(arr),
            "calmar": calmar,
            "hit_rate": float((arr > 0).mean()),
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "avg_win": float(wins.mean()) if wins.size else 0.0,
            "avg_loss": float(losses.mean()) if losses.size else 0.0,
            "skew": float(_sps.skew(arr)) if n > 2 else float("nan"),
            "kurtosis": float(_sps.kurtosis(arr)) if n > 3 else float("nan"),
            "var_95": float(np.percentile(arr, 5)),
            "cvar_95": float(arr[arr <= np.percentile(arr, 5)].mean()),
            "best": float(arr.max()),
            "worst": float(arr.min()),
            "tail_ratio": _tail_ratio(arr),
        }
    )

    if trades is not None and len(trades):
        tr = trades["return"].to_numpy(dtype=float)
        twin = tr[tr > 0]
        tloss = tr[tr < 0]
        out.update(
            {
                "n_trades": int(len(tr)),
                "trade_win_rate": float((tr > 0).mean()),
                "trade_avg_win": float(twin.mean()) if twin.size else 0.0,
                "trade_avg_loss": float(tloss.mean()) if tloss.size else 0.0,
                "trade_expectancy": float(tr.mean()),
                "trade_profit_factor": (
                    float(twin.sum() / -tloss.sum()) if tloss.size and tloss.sum() < 0 else float("inf")
                ),
            }
        )

    return out


def _tail_ratio(arr: np.ndarray) -> float:
    """Ratio of the right tail to the left tail (>1 = fatter upside)."""
    left = abs(np.percentile(arr, 5))
    right = abs(np.percentile(arr, 95))
    return float(right / left) if left > 0 else float("nan")
