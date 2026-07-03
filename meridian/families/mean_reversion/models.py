"""Concrete mean-reversion models composing existing estimators/deviations/signals.

This module owns mean reversion family concrete models; it does NOT own
portfolio construction or risk management (those stay in portfolio/).

Importing this module registers its models under the ``mean_reversion`` family. Only
``ZScoreReversionModel`` is implemented so far (it proves the multi-model pattern by
reusing the sma/zscore pipeline end to end); the other four are TODO.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.families.base import EstimatorModel, Model, register_model
from meridian.features.indicators import overnight_gap, rsi


@register_model("mean_reversion", "zscore_reversion")
class ZScoreReversionModel(EstimatorModel):
    """Revert when price deviates from its moving-average fair value (z-score bands)."""

    estimator = "sma"
    deviation = "zscore"
    window = 20


@register_model("mean_reversion", "bollinger_reversion")
class BollingerReversionModel(EstimatorModel):
    """Bollinger-style reversion: z-score of price vs an EMA fair value."""

    estimator = "ema"
    deviation = "zscore"
    window = 20


@register_model("mean_reversion", "atr_extension")
class AtrExtensionModel(EstimatorModel):
    """Revert when price extends far from fair value measured in ATR units."""

    estimator = "sma"
    deviation = "atr_norm"
    window = 20


@register_model("mean_reversion", "rsi_exhaustion")
class RsiExhaustionModel(Model):
    """Fade exhaustion: long when RSI is oversold (<low), short when overbought (>high).

    Holds the position until RSI reverts to the neutral band (exit toward 50).
    """

    window = 14
    low = 30.0
    high = 70.0

    def __init__(self, *, window: int | None = None):
        if window is not None:
            self.window = window

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        r = rsi(prices, self.window)
        pos = pd.Series(np.nan, index=prices.index)
        pos[r < self.low] = 1.0
        pos[r > self.high] = -1.0
        pos[(r >= 45) & (r <= 55)] = 0.0          # exit near neutral
        return pos.ffill().fillna(0).astype(int)


@register_model("mean_reversion", "zscore_neutral_regime")
class ZScoreNeutralRegimeModel(Model):
    """Symmetric z-score mean reversion, active only in neutral intermediate regimes.

    Same entry/exit logic as ``zscore_reversion`` but skips entries whenever the
    60-day return is outside the ±``trend_band`` range. A reading beyond +8% flags
    a sustained bull run (don't short); below -8% flags a bear run (don't long).
    In those trending periods the model is flat. This preserves the symmetric,
    2–5 day nature of mean reversion while avoiding the persistent-trend entries
    that destroy symmetric strategies on directional markets (2020–21 bull and
    2022 bear each devastate the contra-trend leg without this gate).
    """

    window       = 20
    entry_z      = 1.5
    trend_window = 60
    trend_band   = 0.08     # flat when |60-day return| > 8 %

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        mean  = prices.rolling(self.window).mean()
        std   = prices.rolling(self.window).std()
        z     = (prices - mean) / std.replace(0, np.nan)
        roc60 = prices / prices.shift(self.trend_window) - 1.0
        pos = pd.Series(np.nan, index=prices.index)
        pos[(z < -self.entry_z) & (roc60 > -self.trend_band)] =  1.0   # long: oversold, not in bear
        pos[(z >  self.entry_z) & (roc60 <  self.trend_band)] = -1.0   # short: overbought, not bull
        pos[z.abs() < 0.5]                                     =  0.0   # exit near mean
        return pos.ffill().fillna(0).astype(int)


@register_model("mean_reversion", "zscore_reversion_long_only")
class ZScoreReversionLongOnlyModel(EstimatorModel):
    """Z-score reversion, long-only: buy oversold, sell at fair value — never short.

    Enters long when the z-score falls below -threshold (price unusually cheap vs SMA).
    Exits when z-score reverts above 0 (back to fair value). Flat when overbought.
    Avoids the catastrophic short-leg drawdowns in persistent bull markets.
    """

    estimator = "sma"
    deviation = "zscore"
    window = 20
    entry_z = -1.5

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        mean = prices.rolling(self.window).mean()
        std  = prices.rolling(self.window).std()
        z = (prices - mean) / std.replace(0, np.nan)
        pos = pd.Series(np.nan, index=prices.index)
        pos[z < self.entry_z] = 1.0
        pos[z >= 0.0]         = 0.0
        return pos.ffill().fillna(0).astype(int)


@register_model("mean_reversion", "rsi_reversion_ma_filter")
class RsiReversionMaFilterModel(Model):
    """RSI mean reversion, long-only, gated by the 200-bar MA trend filter.

    Enters long when RSI < 35 AND price is above the 200-bar MA (healthy dip in an
    uptrend). Exits when RSI reverts above 50. Never enters short — avoids catching
    falling knives in bear markets, which is the dominant failure mode for symmetric
    mean-reversion strategies on index ETFs.
    """

    rsi_window = 14
    ma_window  = 200
    entry_rsi  = 35.0
    exit_rsi   = 50.0

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        r  = rsi(prices, self.rsi_window)
        ma = prices.rolling(self.ma_window).mean()
        in_trend = prices > ma
        pos = pd.Series(np.nan, index=prices.index)
        pos[(r < self.entry_rsi) & in_trend] = 1.0
        pos[r > self.exit_rsi]               = 0.0
        return pos.ffill().fillna(0).astype(int)


@register_model("mean_reversion", "bollinger_reversion_long_only")
class BollingerReversionLongOnlyModel(EstimatorModel):
    """Bollinger reversion, long-only: buy below lower band, exit at middle band.

    Same as bollinger_reversion but never shorts the upper band. The EMA fair value
    is the middle; lower band = mean − 2σ. Exit when price returns to fair value.
    """

    estimator = "ema"
    deviation = "zscore"
    window    = 20
    entry_z   = -2.0

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        ema = prices.ewm(span=self.window, adjust=False).mean()
        std = prices.rolling(self.window).std()
        z   = (prices - ema) / std.replace(0, np.nan)
        pos = pd.Series(np.nan, index=prices.index)
        pos[z < self.entry_z] = 1.0
        pos[z >= 0.0]         = 0.0
        return pos.ffill().fillna(0).astype(int)


@register_model("mean_reversion", "gap_fill")
class GapFillModel(Model):
    """Fade overnight gaps: short a gap-up, long a gap-down (betting the gap fills).

    Needs OHLC ``bars`` (open + close); without them it produces no positions.
    """

    threshold = 0.01

    def __init__(self, *, threshold: float | None = None):
        if threshold is not None:
            self.threshold = threshold

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        pos = pd.Series(0, index=prices.index, dtype=int)
        if bars is None or "open" not in bars or "close" not in bars:
            return pos
        gap = overnight_gap(bars)
        pos[gap > self.threshold] = -1
        pos[gap < -self.threshold] = 1
        return pos
