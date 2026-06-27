"""Single-series technical indicators for strategy models.

This module owns causal, vectorized indicators (RSI, overnight gap, volume ratio) the
family models compose into entry/exit rules; it does NOT own fair-value estimation
(estimators/) or regime labeling (regimes/). ADX lives in ``regimes.labeler`` and is
reused there rather than duplicated here.

Every indicator uses only information up to and including each bar (no look-ahead); the
backtester applies the usual one-bar position lag downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (0-100): >70 overbought, <30 oversold.

    Uses Wilder smoothing (``ewm(alpha=1/window)``) of average gains and losses. NaN until
    enough history; 100 when there have been no losses.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)


def overnight_gap(bars: pd.DataFrame) -> pd.Series:
    """Overnight gap as a return: ``(open - prior close) / prior close``.

    Needs OHLC ``bars`` with ``open`` and ``close`` columns. NaN on the first bar.
    """
    prev_close = bars["close"].shift(1)
    return (bars["open"] - prev_close) / prev_close


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume relative to its ``window``-bar average (>1 = above-average participation)."""
    avg = volume.rolling(window).mean()
    return volume / avg.replace(0.0, np.nan)
