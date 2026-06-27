"""Concrete breakout models built on the breakout archetype.

This module owns breakouts family concrete models; it does NOT own portfolio
construction or risk management (those stay in portfolio/).

Importing this module registers its models under the ``breakouts`` family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.families.base import BreakoutModel, Model, register_model
from meridian.features.indicators import volume_ratio


@register_model("breakouts", "donchian_breakout")
class DonchianBreakoutModel(BreakoutModel):
    """Long on a new 20-bar high, short on a new 20-bar low, held until reversed."""

    window = 20


@register_model("breakouts", "donchian_long_only")
class DonchianLongOnlyModel(Model):
    """Donchian breakout, long-only, gated by the 200-bar MA trend filter.

    Enters long on a new ``window``-bar high only when price is above the 200-bar MA
    (confirming the broader trend). Exits when price falls back below the ``window``-bar low.
    Never short — avoids the catastrophic drawdowns of the bidirectional Donchian.
    """

    window = 20
    trend_window = 200

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        hi = prices.rolling(self.window).max()
        lo = prices.rolling(self.window).min()
        ma = prices.rolling(self.trend_window).mean()
        in_trend = prices > ma
        raw = pd.Series(np.nan, index=prices.index)
        raw[(prices >= hi) & in_trend] = 1.0
        raw[prices <= lo] = 0.0
        return raw.ffill().fillna(0).astype(int)


@register_model("breakouts", "donchian_55_20")
class Donchian5520Model(Model):
    """Turtle Traders system: enter on 55-bar high, exit on 20-bar low, long-only.

    The original Turtle Trading entry rule (Richard Dennis, 1983). Long-only above
    the 200-bar MA to avoid fighting the secular trend.
    """

    entry_window = 55
    exit_window  = 20
    trend_window = 200

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        entry = prices.rolling(self.entry_window).max()
        exit_ = prices.rolling(self.exit_window).min()
        ma    = prices.rolling(self.trend_window).mean()
        in_trend = prices > ma
        raw = pd.Series(np.nan, index=prices.index)
        raw[(prices >= entry) & in_trend] = 1.0
        raw[prices <= exit_] = 0.0
        return raw.ffill().fillna(0).astype(int)


@register_model("breakouts", "volume_confirmed_breakout")
class VolumeConfirmedBreakoutModel(Model):
    """Donchian breakout taken only when volume expands (>= ``vol_mult``x its average).

    Needs OHLCV ``bars`` (volume); without them it produces no positions.
    """

    window = 20
    vol_window = 20
    vol_mult = 1.5

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        if bars is None or "volume" not in bars:
            return pd.Series(0, index=prices.index, dtype=int)
        hi = prices.rolling(self.window).max()
        lo = prices.rolling(self.window).min()
        confirmed = volume_ratio(bars["volume"], self.vol_window) >= self.vol_mult
        raw = pd.Series(np.nan, index=prices.index)
        raw[(prices >= hi) & confirmed] = 1.0
        raw[(prices <= lo) & confirmed] = -1.0
        return raw.ffill().fillna(0).astype(int)


@register_model("breakouts", "donchian_ma_exit")
class DonchianMaExitModel(Model):
    """Donchian long-only with 200MA exit: enter on new high above trend, exit when trend breaks.

    Enters long on a new ``window``-bar high while price is above the 200-bar MA.
    Exits on EITHER the 20-bar low crossing OR price falling below the 200MA — whichever
    comes first. The 200MA exit makes the strategy exit bear markets much faster than a
    slow Donchian channel alone.
    """

    window       = 20
    trend_window = 200

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        hi = prices.rolling(self.window).max()
        lo = prices.rolling(self.window).min()
        ma = prices.rolling(self.trend_window).mean()
        in_trend = prices > ma
        raw = pd.Series(np.nan, index=prices.index)
        raw[(prices >= hi) & in_trend] = 1.0
        raw[(prices <= lo) | ~in_trend] = 0.0          # exit on channel low OR trend break
        return raw.ffill().fillna(0).astype(int)


@register_model("breakouts", "turtle_ma_exit")
class TurtleMaExitModel(Model):
    """Turtle 55/20 system with 200MA exit added: fastest of three exits wins.

    Enters on a new 55-bar high while above the 200MA. Exits on the first of:
      1. Price falls below the 20-bar low (original Turtle exit)
      2. Price falls below the 200-bar MA (trend breakdown)
    The dual exit dramatically reduces drawdown in persistent bear markets.
    """

    entry_window = 55
    exit_window  = 20
    trend_window = 200

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        entry    = prices.rolling(self.entry_window).max()
        exit_ch  = prices.rolling(self.exit_window).min()
        ma       = prices.rolling(self.trend_window).mean()
        in_trend = prices > ma
        raw = pd.Series(np.nan, index=prices.index)
        raw[(prices >= entry) & in_trend]          = 1.0
        raw[(prices <= exit_ch) | ~in_trend]       = 0.0
        return raw.ffill().fillna(0).astype(int)


@register_model("breakouts", "nr7_breakout")
class NarrowRangeBreakoutModel(Model):
    """Trade the breakout of the narrowest-range bar of the last ``window`` (a volatility squeeze).

    Needs OHLC ``bars`` (high/low). Long when price clears the squeeze bar's high, short below
    its low, held until reversed; without bars it produces no positions.
    """

    window = 7

    def __init__(self, *, window: int | None = None):
        if window is not None:
            self.window = window

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        if bars is None or not {"high", "low"} <= set(bars.columns):
            return pd.Series(0, index=prices.index, dtype=int)
        rng = bars["high"] - bars["low"]
        squeeze = rng == rng.rolling(self.window).min()       # narrowest range in the window
        trigger_high = bars["high"].where(squeeze).ffill()
        trigger_low = bars["low"].where(squeeze).ffill()
        raw = pd.Series(np.nan, index=prices.index)
        raw[prices > trigger_high] = 1.0
        raw[prices < trigger_low] = -1.0
        return raw.ffill().fillna(0).astype(int)
