"""Relativization transforms for cross-sectional / relative-value strategies.

Absolute mean reversion asks "is this price far from its *own* recent average?".
Relative-value mean reversion asks "is this price far from *its peers*?" — which
is where short-term-reversal and statistical-arbitrage edges are documented.

This module turns a ``{symbol: price Series}`` dict into a ``{symbol: relative
Series}`` dict of the *same shape*. The relative series is fed to the existing
estimator → deviation → signal pipeline as the **signal** input, while P&L is
computed from the actual prices (see ``run_universe_backtest``'s
``signal_prices_by_symbol``) — a relative series crosses zero, so it has no
meaningful return of its own.

Only **linear** estimators (sma, ema, ou, lsma, kalman, …) are appropriate for a
relative series; positive-only estimators (geometric, harmonic, exp_reg) assume
price-like inputs and should not be used here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_demean(prices_by_symbol: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Cross-sectional log-price demeaning (the short-term-reversal transform).

    For each bar, ``rel_i = log(P_i) - mean_j(log P_j)`` over the names present
    that bar. A name far *above* the cross-sectional average is "rich" (positive
    relative value), far *below* is "cheap" (negative). Reverting that relative
    value to zero is market-neutral by construction (the relative values sum to
    zero across names each bar).

    Args:
        prices_by_symbol: ``{symbol: price Series}`` (positive prices).

    Returns:
        ``{symbol: relative Series}`` aligned on the union calendar; bars where a
        name has no price (or a non-positive price) are NaN, so the downstream
        pipeline treats it as untradable then.
    """
    prices = pd.DataFrame(prices_by_symbol).astype(float)
    log_prices = np.log(prices.where(prices > 0))
    relative = log_prices.sub(log_prices.mean(axis=1, skipna=True), axis=0)
    return {sym: relative[sym] for sym in relative.columns}
