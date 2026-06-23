"""Position-sizing schemes for cross-sectional portfolios.

A sizing scheme turns a grid of per-symbol signals (rows = dates, columns =
symbols, values in {-1, 0, +1}) into a grid of portfolio *weights* — how much
capital each name gets, in its signal's direction. Schemes target a gross
exposure of ~1 so portfolios are comparable.

All schemes share the signature ``(signals, returns, lookback) -> weights`` so
they are interchangeable; schemes that ignore ``returns`` simply don't use it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(signals: pd.DataFrame, returns: pd.DataFrame | None = None,
                 lookback: int = 20) -> pd.DataFrame:
    """Each active name gets equal capital in its signal direction.

    Gross exposure is 1 whenever any name is active (weights are
    ``sign / n_active``); a flat row is all zeros.
    """
    s = signals.fillna(0.0)
    n_active = (s != 0).sum(axis=1)
    w = s.div(n_active.replace(0, np.nan), axis=0)
    return w.fillna(0.0)


def inverse_vol(signals: pd.DataFrame, returns: pd.DataFrame,
                lookback: int = 20) -> pd.DataFrame:
    """Weight active names inversely to recent volatility (risk parity-ish).

    Quieter names get more capital; each row is normalized so gross exposure is
    1. Reduces the chance that one volatile name dominates portfolio risk.
    """
    vol = returns.rolling(lookback, min_periods=2).std()
    raw = signals.fillna(0.0) / vol.replace(0.0, np.nan)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = raw.abs().sum(axis=1)
    w = raw.div(gross.replace(0, np.nan), axis=0)
    return w.fillna(0.0)


SIZING = {
    "equal_weight": equal_weight,
    "inverse_vol": inverse_vol,
}


def get_sizing(name: str):
    if name not in SIZING:
        raise KeyError(f"unknown sizing {name!r}. Known: {', '.join(SIZING)}")
    return SIZING[name]
