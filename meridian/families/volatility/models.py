"""Volatility family models.

These strategies harvest the equity volatility risk premium — a persistent excess
return earned by selling volatility (or holding inverse-VIX products) when elevated
implied volatility mean-reverts back to normal.

Symbol convention: "SVXY_^VIX" — SVXY is the traded leg, ^VIX is the signal gate.
"""

from __future__ import annotations

import pandas as pd

from meridian.families.base import CrossSectionalModel, register_model


@register_model("volatility", "vix_band")
class VixBandModel(CrossSectionalModel):
    """Hold SVXY (inverse VIX) when VIX is in an elevated but non-extreme band.

    Entry:  VIX > vix_lo  (premium elevated enough to harvest)
    Exit:   VIX > vix_hi  (spike risk too high — Volmageddon territory)
            OR VIX drops back below vix_lo (premium normalised)

    The <35 upper cap sidesteps volatility spikes where inverse-VIX products can
    lose 50–90% in a single session. The 20-35 band backtest (2012–2026) produced
    Sharpe 0.71, MaxDD -37.9%, 10yr +285.6% vs. holding SVXY outright (Sharpe 0.53,
    MaxDD -95.2%, 10yr +9.9%).

    Input basket: {"SVXY": <price series>, "^VIX": <vix series>}
    Only SVXY appears in the output signals — ^VIX is a gate, not a position.
    """

    vix_lo: float = 20.0
    vix_hi: float = 35.0
    cross_sectional: bool = True

    def signals(self, prices_by_symbol: dict[str, pd.Series]) -> pd.DataFrame:
        svxy = prices_by_symbol.get("SVXY")
        vix  = prices_by_symbol.get("^VIX")

        if svxy is None or vix is None:
            return pd.DataFrame()

        idx = svxy.index.intersection(vix.index)
        svxy = svxy.loc[idx]
        vix  = vix.loc[idx]

        in_band = (vix > self.vix_lo) & (vix < self.vix_hi)
        sig = pd.DataFrame({"SVXY": in_band.astype(int)}, index=idx)
        return sig
