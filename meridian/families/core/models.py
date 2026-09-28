"""Core-book models: buy and hold.

The 2026-09 research found no honest timing skill in Meridian's sleeves, and
the pre-registered structure test (research/plans/meridian_structure_2026_10.json)
picked a simple held core: 60/40 SPY/IEF, the long-term ETF rotation and
T-bills. Live sizing splits a sleeve equally across its longs, so the 60/40
is two sleeves (SPY, IEF) and T-bills a third (SGOV), each holding one ETF.
"""

from __future__ import annotations

import pandas as pd

from meridian.families.base import Model, register_model


class _BuyAndHold(Model):
    """Always long the sleeve's one symbol."""

    def signals(self, prices: pd.Series, bars: pd.DataFrame | None = None) -> pd.Series:
        return pd.Series(1, index=prices.index)


@register_model("core_equity", "buy_and_hold")
class CoreEquityHold(_BuyAndHold):
    """US equities for the core book (SPY)."""


@register_model("core_bonds", "buy_and_hold")
class CoreBondsHold(_BuyAndHold):
    """Intermediate Treasuries for the core book (IEF)."""


@register_model("core_tbills", "buy_and_hold")
class CoreTbillsHold(_BuyAndHold):
    """T-bills for the core book (SGOV)."""
