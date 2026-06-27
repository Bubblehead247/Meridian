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
    return pd.DataFrame(prices_by_symbol).astype(float).pct_change(lookback)


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
