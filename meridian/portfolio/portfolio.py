"""Cross-sectional portfolio backtester.

Combines per-symbol signals into one portfolio return stream: size the signals
into weights, lag them one bar (no look-ahead), apply them to each symbol's next
return, and net out turnover costs. The output is a single return series that
plugs into the same analytics and validation machinery as a single-asset
backtest — but built from many names, so idiosyncratic noise averages out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from meridian.portfolio.sizing import get_sizing


@dataclass
class PortfolioResult:
    """Outcome of a cross-sectional portfolio backtest."""

    returns: pd.Series          # per-bar net portfolio return
    equity: pd.Series           # cumulative net equity (starts at 1.0)
    weights: pd.Series | pd.DataFrame  # held weights (already lagged), per symbol
    gross_exposure: pd.Series   # sum of |weight| held each bar
    returns_by_symbol: pd.DataFrame  # per-symbol returns (for Monte-Carlo)
    turnover_series: pd.Series = field(default_factory=pd.Series)  # per-bar two-sided turnover
    meta: dict = field(default_factory=dict)


def backtest_portfolio(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    sizing: str = "equal_weight",
    cost_bps: float = 1.0,
    lookback: int = 20,
    flatten_overnight: bool = False,
) -> PortfolioResult:
    """Backtest a cross-sectional portfolio.

    Args:
        signals: Rows = dates, columns = symbols, values {-1, 0, +1} decided at
            each bar's close.
        prices: Same shape; per-symbol prices (use adjusted close).
        sizing: Sizing scheme name (`equal_weight` or `inverse_vol`).
        cost_bps: Flat cost in bps charged on portfolio turnover.
        lookback: Lookback for vol-based sizing.
        flatten_overnight: For intraday bars — force the book flat on the last bar
            of each session (derived from the DatetimeIndex's date). Because the
            held position is lagged, the next session opens flat, so the overnight
            gap return is never booked and nothing is carried overnight (and the
            daily flatten shows up as real turnover/cost).

    Returns:
        A `PortfolioResult` whose `returns` is the net portfolio return series.
    """
    prices = prices.reindex(columns=signals.columns)
    # fill_method=None: a missing price gives a NaN (->0) return, never a padded
    # one — so membership gaps / removals don't fabricate carried-forward prices.
    returns = prices.pct_change(fill_method=None).fillna(0.0)

    weights = get_sizing(sizing)(signals, returns, lookback)
    if flatten_overnight and len(weights):
        dates = pd.Series(pd.DatetimeIndex(weights.index).normalize(), index=weights.index)
        session_close = dates.ne(dates.shift(-1))  # last bar of each session (and final bar)
        weights = weights.mask(session_close, 0.0)
    held = weights.shift(1).fillna(0.0)  # lag: weight decided at t earns t->t+1

    gross = (held * returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = held.iloc[0].abs().sum()  # cost of entering from flat
    cost = turnover * (cost_bps / 1e4)
    net = gross - cost

    equity = (1.0 + net).cumprod()
    return PortfolioResult(
        returns=net,
        equity=equity,
        weights=held,
        gross_exposure=held.abs().sum(axis=1),
        returns_by_symbol=returns,
        turnover_series=turnover,
        meta={"sizing": sizing, "cost_bps": cost_bps, "n_symbols": signals.shape[1]},
    )
