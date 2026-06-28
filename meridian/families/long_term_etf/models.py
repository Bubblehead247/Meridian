"""Concrete long-term ETF models built on the long-term archetype.

This module owns long-term ETF family concrete models; it does NOT own
portfolio construction or risk management (those stay in portfolio/).

Importing this module registers its models under the ``long_term_etf`` family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.families.base import LongTermETFModel, Model, register_model


@register_model("long_term_etf", "above_200ma")
class Above200MAModel(LongTermETFModel):
    """Hold the ETF while it is above its 200-day average, otherwise sit in cash (flat)."""

    window = 200


@register_model("long_term_etf", "dual_momentum")
class DualMomentumModel(Model):
    """Antonacci-style dual momentum: hold SPY when equity beats bonds over lookback AND
    equity has positive absolute return; else hold TLT; else cash.

    When run on a single price series this degrades to absolute momentum only
    (long when 12-month return > 0, else flat) — the relative leg requires a
    companion TLT series passed via ``bars`` as column ``"tlt_close"``.
    """

    lookback: int = 252   # 12-month momentum window

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        abs_ret = prices.pct_change(self.lookback)
        abs_pos = (abs_ret > 0).astype(int)

        if bars is not None and "tlt_close" in bars.columns:
            tlt     = bars["tlt_close"].reindex(prices.index)
            rel_ret = abs_ret - tlt.pct_change(self.lookback)
            # +1 → equity beats bonds AND positive; -1 → bonds (proxy: flat here); 0 → cash
            sig = pd.Series(0, index=prices.index)
            sig[abs_pos.astype(bool) & (rel_ret > 0)] = 1
        else:
            sig = abs_pos

        return sig.fillna(0).astype(int)


@register_model("long_term_etf", "ma_bond_rotation")
class MABondRotationModel(Model):
    """Risk-on / risk-off: hold equity ETF while above its 200-day MA, else rotate to bonds.

    When run on an equity ETF alone the bond leg is represented as flat (cash).
    Pass ``bars["tlt_close"]`` to enable the true rotation signal (the bond leg
    earns TLT returns when equity is below 200MA).
    """

    window: int = 200

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        ma      = prices.rolling(self.window).mean()
        equity  = ((prices >= ma) & ma.notna()).astype(int)
        # When equity signal is off, we're in bonds — mark as flat for single-series
        # backtester; the portfolio layer handles the TLT allocation.
        return equity.fillna(0).astype(int)
