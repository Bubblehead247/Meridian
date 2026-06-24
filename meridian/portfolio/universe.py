"""Universe-wide backtest: one estimator across many symbols.

For each symbol the same causal pipeline as the single-asset case computes a
signal path; the paths are aligned on a common calendar and handed to the
cross-sectional backtester. This is the strongest test of a mean-reversion
estimator: it must work across a breadth of names, not just one.
"""

from __future__ import annotations

from functools import reduce

import pandas as pd

from meridian.portfolio.portfolio import PortfolioResult, backtest_portfolio
from meridian.signals import SignalConfig, compute_scores, generate_positions


def per_symbol_signals(
    prices_by_symbol: dict[str, pd.Series],
    estimator: str,
    deviation: str,
    signal: SignalConfig | None = None,
    *,
    window: int = 20,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Causal signal path per symbol, as a date × symbol DataFrame.

    Each symbol is scored on its own full history (so warm-up is correct), then
    the columns are aligned on the union calendar.
    """
    bars_by_symbol = bars_by_symbol or {}
    cols: dict[str, pd.Series] = {}
    for sym, px in prices_by_symbol.items():
        scores = compute_scores(px, estimator, deviation, window=window,
                                bars=bars_by_symbol.get(sym))
        # A name with no price that bar (e.g. before listing / after index removal)
        # is held flat — so removed names leave the portfolio rather than linger.
        cols[sym] = generate_positions(scores, signal).where(px.notna())
    return pd.DataFrame(cols)


def common_index(prices_by_symbol: dict[str, pd.Series]) -> pd.Index:
    """Intersection of all symbols' date indices (the *shared* calendar).

    Note: with staggered listings (recent IPOs) this collapses to the newest
    symbol's range. Prefer `union_index` for screened universes.
    """
    return reduce(lambda a, b: a.intersection(b),
                  (s.index for s in prices_by_symbol.values()))


def union_index(prices_by_symbol: dict[str, pd.Series]) -> pd.Index:
    """Union of all symbols' date indices (full study calendar).

    A symbol absent before its listing simply contributes no signal those bars
    (masked to weight 0 downstream), so staggered IPOs/delistings are handled
    without collapsing the study window.
    """
    idx = reduce(lambda a, b: a.union(b), (s.index for s in prices_by_symbol.values()))
    return idx.sort_values()


def run_universe_backtest(
    prices_by_symbol: dict[str, pd.Series],
    estimator: str,
    deviation: str = "zscore",
    signal: SignalConfig | None = None,
    *,
    sizing: str = "equal_weight",
    window: int = 20,
    cost_bps: float = 1.0,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    index: pd.Index | None = None,
    align: str = "union",
    signal_prices_by_symbol: dict[str, pd.Series] | None = None,
    flatten_overnight: bool = False,
) -> PortfolioResult:
    """Backtest one estimator across a universe into a single portfolio.

    Args:
        prices_by_symbol: ``{symbol: price Series}`` — the **tradable** prices,
            used for P&L.
        estimator/deviation/signal/window: the usual single-asset knobs, applied
            identically to every symbol (no per-symbol optimization).
        sizing: portfolio sizing scheme.
        cost_bps: turnover cost.
        bars_by_symbol: optional OHLC per symbol (for ``atr_norm``).
        index: restrict to these dates (defaults to the study calendar).
        align: "union" (full calendar, handles staggered listings) or
            "intersection" (shared calendar only). Ignored if ``index`` given.
        signal_prices_by_symbol: optional **signal** series (e.g. a
            cross-sectional relative series). When given, signals are computed
            from it while P&L still uses ``prices_by_symbol``. Defaults to using
            the tradable prices for both (absolute strategy).

    Returns:
        A `PortfolioResult` for the whole universe.
    """
    if index is not None:
        idx = index
    else:
        idx = union_index(prices_by_symbol) if align == "union" else common_index(prices_by_symbol)
    signal_src = (
        signal_prices_by_symbol if signal_prices_by_symbol is not None else prices_by_symbol
    )
    signals = per_symbol_signals(
        signal_src, estimator, deviation, signal,
        window=window, bars_by_symbol=bars_by_symbol,
    ).reindex(idx)
    prices = pd.DataFrame({s: p.reindex(idx) for s, p in prices_by_symbol.items()})
    return backtest_portfolio(
        signals, prices, sizing=sizing, cost_bps=cost_bps, lookback=window,
        flatten_overnight=flatten_overnight,
    )
