"""Concrete momentum models — single-asset and cross-sectional.

This module owns momentum family concrete models; it does NOT own portfolio
construction or risk management (those stay in portfolio/).

Importing this module registers its models under the ``momentum`` family.
"""

from __future__ import annotations

import numpy as np

import pandas as pd

from meridian.families.base import CrossSectionalModel, Model, MomentumModel, register_model
from meridian.features.cross_sectional import momentum_scores, rank_signals


@register_model("momentum", "roc_momentum")
class RocMomentumModel(MomentumModel):
    """Single-asset: long after a positive 60-bar return, short after a negative one."""

    window = 60


@register_model("momentum", "roc_momentum_long_only")
class RocMomentumLongOnlyModel(Model):
    """Long when 252-bar (12-month) return is positive; flat otherwise — no short leg.

    12-month lookback is the industry standard for cross-sectional momentum (Jegadeesh &
    Titman 1993). Long-only avoids the catastrophic short drawdowns on index ETFs.
    """

    window = 252

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        roc = prices / prices.shift(self.window) - 1.0
        return (roc > 0).astype(int)


@register_model("momentum", "roc_momentum_200ma")
class RocMomentum200MaModel(Model):
    """12-month ROC long-only, additionally gated by the 200-bar MA trend filter.

    Requires both positive 12-month momentum AND price above its 200-bar MA before entering.
    This two-filter approach reduces whipsaws and false signals during corrections.
    """

    window = 252
    trend_window = 200

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        roc = prices / prices.shift(self.window) - 1.0
        ma  = prices.rolling(self.trend_window).mean()
        return ((roc > 0) & (prices > ma)).astype(int)


@register_model("momentum", "swing_momentum")
class SwingMomentumModel(Model):
    """Swing momentum: long when the 20-day MA has been rising over the last two weeks.

    Measures whether the short-term trend is accelerating: long when today's
    20-day MA is above its value ``slope_lag`` bars ago (MA slope positive).
    Exits when the slope turns negative. Targets 1–3 week holds with 8–15
    trades/yr — distinct from the slow 50/200 crossover trend-following models.
    """

    ma_window = 20
    slope_lag  = 10   # compare MA today vs MA 2 weeks ago

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        ma    = prices.rolling(self.ma_window).mean()
        slope = ma - ma.shift(self.slope_lag)
        pos = pd.Series(np.nan, index=prices.index)
        pos[slope > 0] = 1.0
        pos[slope < 0] = 0.0
        return pos.ffill().fillna(0).astype(int)


@register_model("momentum", "relative_strength")
class RelativeStrengthModel(CrossSectionalModel):
    """Cross-sectional: long the strongest names by trailing return, short the weakest."""

    lookback = 60
    quantile = 0.3


@register_model("momentum", "relative_strength_long_only")
class RelativeStrengthLongOnlyModel(CrossSectionalModel):
    """Cross-sectional: long the strongest names by trailing return; no short leg.

    Long-only removes the catastrophic short drawdowns that occur in bull markets when
    weak names still rise. Practical for a managed account where shorting is unavailable.
    """

    lookback = 60
    quantile = 0.3
    long_only = True


@register_model("momentum", "dual_momentum")
class DualMomentumModel(CrossSectionalModel):
    """Dual momentum: relative rank gated by absolute momentum.

    Long only top-ranked names that *also* have positive absolute momentum; short only
    bottom-ranked names with negative absolute momentum (so a falling-but-least-bad name is
    not bought, and a rising-but-weakest name is not shorted).
    """

    lookback = 120
    quantile = 0.3

    def signals(self, prices_by_symbol):
        scores = momentum_scores(prices_by_symbol, self.lookback)
        rel = rank_signals(scores, quantile=self.quantile)
        absmom = np.sign(scores)
        sig = rel.where(~((rel > 0) & (absmom <= 0)), 0)   # drop longs w/o positive abs mom
        sig = sig.where(~((rel < 0) & (absmom >= 0)), 0)   # drop shorts w/o negative abs mom
        return sig.astype(int)


@register_model("momentum", "dual_momentum_long_only")
class DualMomentumLongOnlyModel(CrossSectionalModel):
    """Dual momentum, long-only: long top-ranked names with positive absolute momentum only.

    The absolute momentum gate (positive 120-day return required to enter long) acts as a
    bear-market filter — the portfolio moves to cash when all names have falling momentum,
    which dramatically limits drawdown vs. the long/short version.
    """

    lookback = 120
    quantile = 0.3
    long_only = True

    def signals(self, prices_by_symbol):
        scores = momentum_scores(prices_by_symbol, self.lookback)
        rel = rank_signals(scores, quantile=self.quantile, long_only=True)
        absmom = np.sign(scores)
        # Drop longs where absolute momentum is non-positive (cash those positions)
        sig = rel.where(~((rel > 0) & (absmom <= 0)), 0)
        return sig.astype(int)
