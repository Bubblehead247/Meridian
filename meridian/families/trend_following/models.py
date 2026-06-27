"""Concrete trend-following models built on the directional archetype.

This module owns trend following family concrete models; it does NOT own
portfolio construction or risk management (those stay in portfolio/).

Importing this module registers its models under the ``trend_following`` family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.families.base import Model, MovingAverageTrendModel, register_model
from meridian.regimes.labeler import adx


@register_model("trend_following", "ma_trend")
class MaTrendModel(MovingAverageTrendModel):
    """Long above / short below a 50-bar moving average (medium-term trend)."""

    window = 50


@register_model("trend_following", "ma_trend_long_only")
class MaTrendLongOnlyModel(Model):
    """Long when price is above the 200-bar MA; flat otherwise — no short leg."""

    window = 200

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        ma = prices.rolling(self.window).mean()
        return (prices > ma).astype(int)


@register_model("trend_following", "dual_ma_trend")
class DualMaTrendModel(Model):
    """50/200-bar dual MA crossover: long when fast > slow, flat otherwise.

    Uses the same filter as ``above_200ma`` (long_term_etf) but enters on the
    fast/slow crossover signal rather than holding continuously above the 200MA.
    """

    fast = 50
    slow = 200

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        fast_ma = prices.rolling(self.fast).mean()
        slow_ma = prices.rolling(self.slow).mean()
        return (fast_ma > slow_ma).astype(int)


@register_model("trend_following", "adx_trend")
class AdxTrendModel(MovingAverageTrendModel):
    """MA-trend direction, but only traded when ADX confirms a strong trend (> ``adx_min``).

    Reuses the ADX from ``regimes.labeler``; needs OHLC ``bars`` (high/low/close) — without
    them no trend strength can be measured, so it stays flat.
    """

    window = 50
    adx_window = 14
    adx_min = 25.0

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        direction = super().signals(prices)
        if bars is None or not {"high", "low", "close"} <= set(bars.columns):
            return pd.Series(0, index=prices.index, dtype=int)
        strength = adx(bars["high"], bars["low"], bars["close"], self.adx_window)
        return direction.where(strength > self.adx_min, 0).astype(int)


@register_model("trend_following", "chandelier_trend")
class ChandelierTrendModel(Model):
    """Chandelier exit trend following: enter on new high, trail with ATR stop.

    Enters long on a new ``entry_window``-bar closing high. Exits when price
    falls more than ``atr_mult`` × ATR(``atr_window``) below the rolling high
    (a chandelier exit). Targets 1–3 month holds that match actual trend duration
    — not the 200MA lag of ``ma_trend_long_only`` or ``dual_ma_trend``.
    """

    entry_window = 20
    atr_window   = 20
    atr_mult     = 2.0

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        if bars is not None and {"high", "low"} <= set(bars.columns):
            high, low = bars["high"], bars["low"]
        else:
            high = low = prices
        prev = prices.shift(1)
        tr  = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window).mean()
        roll_high = prices.rolling(self.entry_window).max()
        stop = roll_high - self.atr_mult * atr
        raw = pd.Series(np.nan, index=prices.index)
        raw[prices >= roll_high] = 1.0
        raw[prices <  stop]      = 0.0
        return raw.ffill().fillna(0).astype(int)


@register_model("trend_following", "channel_breakout")
class ChannelBreakoutTrendModel(Model):
    """Donchian channel breakout filtered by a long-MA trend.

    Long a new ``window``-bar high only while above the ``trend_window`` MA; short a new low
    only while below it — so breakouts are taken in the direction of the prevailing trend.
    """

    window = 20
    trend_window = 100

    def __init__(self, *, window: int | None = None, trend_window: int | None = None):
        if window is not None:
            self.window = window
        if trend_window is not None:
            self.trend_window = trend_window

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        hi = prices.rolling(self.window).max()
        lo = prices.rolling(self.window).min()
        ma = prices.rolling(self.trend_window).mean()
        raw = pd.Series(np.nan, index=prices.index)
        raw[(prices >= hi) & (prices > ma)] = 1.0
        raw[(prices <= lo) & (prices < ma)] = -1.0
        return raw.ffill().fillna(0).astype(int)

