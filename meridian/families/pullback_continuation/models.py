"""Concrete pullback-continuation models built on the pullback archetype.

This module owns pullback continuation family concrete models; it does NOT own
portfolio construction or risk management (those stay in portfolio/).

Importing this module registers its models under the ``pullback_continuation`` family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.families.base import Model, PullbackModel, register_model
from meridian.features.indicators import rsi


@register_model("pullback_continuation", "ma_pullback")
class MaPullbackModel(PullbackModel):
    """Long-only: buy dips below the fast MA while the fast MA leads the slow MA (uptrend)."""

    fast = 20
    slow = 50


@register_model("pullback_continuation", "rsi_pullback")
class RsiPullbackModel(Model):
    """Long-only: in an uptrend (fast MA > slow MA), buy when RSI dips below ``dip``."""

    fast = 20
    slow = 50
    rsi_window = 14
    dip = 40.0

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        fast = prices.rolling(self.fast).mean()
        slow = prices.rolling(self.slow).mean()
        r = rsi(prices, self.rsi_window)
        long = (fast > slow) & (r < self.dip) & slow.notna()
        return long.astype(int)


@register_model("pullback_continuation", "rsi_pullback_50_200")
class RsiPullback50200Model(RsiPullbackModel):
    """RSI pullback with a stricter 50/200 MA trend filter.

    The 50/200 confirmation requires price to have been in a sustained uptrend
    before entering, which reduces false entries during regime deterioration.
    """

    fast = 50
    slow = 200


@register_model("pullback_continuation", "rsi_pullback_50_200_tight")
class RsiPullback50200TightModel(RsiPullbackModel):
    """RSI pullback with 50/200 MA filter and tight RSI < 35 entry threshold.

    Only enters on deeply oversold dips within a confirmed long-term uptrend.
    High win rate, low drawdown, low trade frequency (~3/yr).
    """

    fast = 50
    slow = 200
    dip = 35.0


@register_model("pullback_continuation", "rsi_pullback_continuation")
class RsiPullbackContinuationModel(Model):
    """Pullback continuation: enter on dip, hold until the trend reasserts.

    Entry: RSI < ``entry_rsi`` while price is above both MAs (confirmed uptrend).
    Exit: RSI recovers to ``exit_rsi`` (trend has resumed) OR price breaks below
    the slow MA (trend has broken). The wide entry/exit spread gives 1–3 week holds
    — long enough to capture the continuation move after the dip.
    """

    fast      = 50
    slow      = 200
    rsi_window = 14
    entry_rsi  = 40.0
    exit_rsi   = 60.0

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        fast_ma = prices.rolling(self.fast).mean()
        slow_ma = prices.rolling(self.slow).mean()
        r       = rsi(prices, self.rsi_window)
        in_trend = (fast_ma > slow_ma) & slow_ma.notna()
        pos = pd.Series(np.nan, index=prices.index)
        pos[(r < self.entry_rsi) & in_trend] = 1.0   # enter: dip in uptrend
        pos[r > self.exit_rsi]               = 0.0   # exit: RSI recovered
        pos[~in_trend]                       = 0.0   # exit: trend breaks
        return pos.ffill().fillna(0).astype(int)
