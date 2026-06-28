"""Concrete long-term ETF models built on the long-term archetype.

This module owns long-term ETF family concrete models; it does NOT own
portfolio construction or risk management (those stay in portfolio/).

Importing this module registers its models under the ``long_term_etf`` family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.families.base import CrossSectionalModel, LongTermETFModel, Model, register_model


@register_model("long_term_etf", "dual_momentum_cs")
class DualMomentumCSModel(CrossSectionalModel):
    """Antonacci dual momentum over a basket (e.g. SPY / TLT / GLD).

    Each bar, rank all assets by their trailing ``lookback``-bar return (12-month
    default).  Hold the top-ranked asset only when it has a positive absolute
    return (absolute-momentum gate); otherwise go to cash.  One asset at a time.

    ``buffer_pct`` implements hysteresis: the challenger must beat the current
    holding by at least this margin in 12-month return before a rotation fires,
    preventing daily flip-flopping when two assets rank nearly identically.
    """

    lookback: int = 252      # 12-month ranking window
    buffer_pct: float = 0.02 # challenger must beat current holding by ≥ 2 pct-points
    long_only: bool = True

    def signals(self, prices_by_symbol: dict[str, pd.Series]) -> pd.DataFrame:
        from meridian.features.cross_sectional import momentum_scores

        scores = momentum_scores(prices_by_symbol, self.lookback)
        sig = pd.DataFrame(0, index=scores.index, columns=scores.columns, dtype=int)

        current: str | None = None      # currently held asset

        for dt in scores.index:
            row = scores.loc[dt].dropna()
            if row.empty:
                current = None
                continue

            best = row.idxmax()
            cur_in = current is not None and current in row.index

            if not cur_in:
                # Entering from flat — enter best only if it has positive absolute return
                current = best if row[best] > 0 else None
            elif best == current:
                # Still the top performer; exit only if 12-month went negative
                if row[current] <= 0:
                    current = None
            else:
                # Rotation candidate: only switch if challenger leads by buffer_pct AND is positive
                if row[best] > 0 and row[best] - row[current] > self.buffer_pct:
                    current = best
                elif row[current] <= 0:
                    # Current's 12-month went negative; no valid replacement clears buffer → flat
                    current = None
                # else: keep current (buffer prevents flip-flopping)

            if current is not None:
                sig.loc[dt, current] = 1

        return sig


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


@register_model("long_term_etf", "ma_bond_rotation_cs")
class MABondRotationCSModel(CrossSectionalModel):
    """MA-based risk-on/off rotation: hold equity while above its long MA, else hold bonds.

    The first symbol in the basket is the equity (the 200-day MA is applied to it).
    Remaining symbols are bond alternatives — held when equity is below its MA.
    When multiple bond alternatives are present, the one with the best trailing
    ``bond_lookback``-bar return is selected each bar.  One asset held at a time.

    Convention: pass symbols as equity-first, e.g. --symbols SPY IEF TLT.
    """

    window: int = 200
    bond_lookback: int = 63  # trailing window for ranking bond alternatives
    long_only: bool = True

    def signals(self, prices_by_symbol: dict[str, pd.Series]) -> pd.DataFrame:
        symbols = list(prices_by_symbol.keys())
        equity = symbols[0]
        bonds  = symbols[1:]

        prices = pd.DataFrame(prices_by_symbol).ffill().bfill()
        ma      = prices[equity].rolling(self.window).mean()
        risk_on = (prices[equity] >= ma) & ma.notna()

        sig = pd.DataFrame(0, index=prices.index, columns=symbols, dtype=int)
        sig.loc[risk_on, equity] = 1

        risk_off = (~risk_on) & ma.notna()
        if not bonds:
            pass  # no bond leg — cash when risk-off
        elif len(bonds) == 1:
            sig.loc[risk_off, bonds[0]] = 1
        else:
            bond_rets = prices[bonds].pct_change(self.bond_lookback, fill_method=None)
            for dt in prices.index[risk_off]:
                row = bond_rets.loc[dt].dropna()
                if not row.empty:
                    sig.loc[dt, row.idxmax()] = 1

        return sig
