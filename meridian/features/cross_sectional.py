"""Cross-sectional signal builders for multi-asset (ranking) strategies.

This module owns turning a ``{symbol: price}`` universe into per-symbol momentum scores and
cross-sectional long/short rank signals; it does NOT own portfolio construction or P&L
(that stays in portfolio/). The signals are the {-1,0,+1} date×symbol frame the portfolio
backtester consumes.

Cross-sectional strategies (relative strength, dual momentum, sector rotation) compare each
name to its *peers* each bar, which the single-asset estimator pipeline cannot express.
"""

from __future__ import annotations

import pandas as pd


def momentum_scores(prices_by_symbol: dict[str, pd.Series], lookback: int = 60) -> pd.DataFrame:
    """Per-symbol momentum = trailing ``lookback``-bar return, as a date×symbol frame."""
    return pd.DataFrame(prices_by_symbol).astype(float).pct_change(lookback, fill_method=None)


def rank_signals(
    scores: pd.DataFrame, *, quantile: float = 0.3, long_only: bool = False
) -> pd.DataFrame:
    """Cross-sectional rank signals: long the top ``quantile`` of names each bar, short the bottom.

    Ranking is within each bar (row), so a name's signal depends on its peers. ``long_only``
    keeps only the longs (e.g. sector rotation). Bars with a missing score are flat for that
    name.
    """
    ranks = scores.rank(axis=1, pct=True)
    sig = pd.DataFrame(0, index=scores.index, columns=scores.columns, dtype=int)
    sig = sig.mask(ranks >= 1.0 - quantile, 1)
    if not long_only:
        sig = sig.mask(ranks <= quantile, -1)
    return sig.where(scores.notna(), 0).astype(int)


def rank_signals_buffered(
    scores: pd.DataFrame,
    *,
    quantile: float = 0.3,
    long_only: bool = False,
    buffer_pct: float = 0.01,
) -> pd.DataFrame:
    """Like rank_signals but with a hysteresis buffer to reduce rotation churn.

    Each bar, currently-held names receive a ``buffer_pct`` bonus on their momentum
    score before re-ranking. A challenger must beat a held name by more than this
    buffer to displace it — preventing daily flip-flops when marginal names are
    nearly tied in 6-month return. Path-dependent so computed bar-by-bar.
    """
    n = scores.shape[1]
    n_long = max(1, round(quantile * n))

    sig = pd.DataFrame(0, index=scores.index, columns=scores.columns, dtype=int)
    prev_long: set = set()

    for dt in scores.index:
        row = scores.loc[dt]
        if row.isna().all():
            prev_long = set()
            continue
        valid = row.dropna()
        adj = valid.copy()
        for sym in prev_long:
            if sym in adj.index:
                adj[sym] += buffer_pct

        top = set(adj.nlargest(n_long).index)
        sig.loc[dt, list(top)] = 1
        if not long_only:
            bottom = set(adj.nsmallest(n_long).index) - top
            sig.loc[dt, list(bottom)] = -1
        prev_long = top

    return sig.where(scores.notna(), 0).astype(int)
