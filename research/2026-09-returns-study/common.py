"""Shared pieces for the 2026-09 returns study (see README.md in this folder).

All returns are TOTAL returns (adj_close, dividends in), 10 bps per side,
fills at the signal close, long-only (the engine default since fe9fca1).
T-bills = ^IRX 13-week yield / 252, lagged a day.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from meridian.data import load_ohlcv
from meridian.execution.live_runner import _expand_symbol
from meridian.families import create_model

COST = 10.0
START = "2010-01-01"
PERIODS = [("2011-2015", "2011-01-01", "2015-12-31"),
           ("2016-2020", "2016-01-01", "2020-12-31"),
           ("2021-2026/06", "2021-01-01", "2026-06-26"),
           ("2011-2026/06", "2011-01-01", "2026-06-26")]
SUB = PERIODS[:3]

_px: dict[str, pd.Series] = {}


def tr(sym: str) -> pd.Series:
    """Total-return price series (dividend-adjusted close)."""
    if sym not in _px:
        _px[sym] = load_ohlcv(sym, START, use_cache=False)["adj_close"].dropna()
    return _px[sym]


def index() -> pd.DatetimeIndex:
    return tr("SPY").loc["2010-06-01":].index


def tbill() -> pd.Series:
    irx = load_ohlcv("^IRX", START, use_cache=False)["close"]
    return (irx.reindex(index()).ffill() / 100 / 252).shift(1).fillna(0.0)


def hold(sym: str) -> pd.Series:
    return tr(sym).pct_change().reindex(index()).fillna(0.0)


def sleeve(family: str, model_name: str, symbol: str, cost: float = COST):
    """(daily net returns, exposure 0..1) for one family/model/symbol, on index()."""
    model = create_model(family, model_name)
    if getattr(model, "cross_sectional", False):
        px = {s: tr(s) for s in _expand_symbol(symbol)}
        bt = model.backtest(px, cost_bps=cost)
        r, e = bt.returns, bt.gross_exposure.clip(upper=1.0)
    else:
        bt = model.backtest(tr(symbol), cost_bps=cost)
        r, e = bt.returns, bt.positions.abs().clip(upper=1.0)
    idx = index()
    return r.reindex(idx).fillna(0.0), e.reindex(idx).fillna(0.0)


def stats(r: pd.Series, a: str, b: str) -> dict:
    r = r.loc[a:b].fillna(0.0)
    eq = (1 + r).cumprod()
    sd = r.std()
    return {"cagr": eq.iloc[-1] ** (252 / len(r)) - 1,
            "vol": sd * np.sqrt(252),
            "sharpe": r.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
            "maxdd": (eq / eq.cummax() - 1).min()}


def table(rows: dict[str, pd.Series], periods=PERIODS, baseline: str | None = None) -> str:
    out = []
    for name, a, b in periods:
        out.append(f"\n### {name}\n| Portfolio | CAGR | Vol | Sharpe | Max DD |\n|---|---:|---:|---:|---:|")
        for label, r in rows.items():
            s = stats(r, a, b)
            lab = f"**{label}**" if label == baseline else label
            out.append(f"| {lab} | {s['cagr']:+.1%} | {s['vol']:.1%} | {s['sharpe']:.2f} | {s['maxdd']:+.1%} |")
    return "\n".join(out)


def regress(y: pd.Series, xs: dict[str, pd.Series], a: str, b: str) -> dict:
    """OLS of daily returns on factor returns; alpha annualized."""
    df = pd.concat([y.rename("y")] + [x.rename(k) for k, x in xs.items()], axis=1).loc[a:b].dropna()
    X = np.column_stack([np.ones(len(df))] + [df[k].to_numpy() for k in xs])
    coef, *_ = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)
    fit = X @ coef
    resid = df["y"].to_numpy() - fit
    r2 = 1 - resid.var() / df["y"].var()
    # t-stat of alpha (plain OLS s.e.)
    s2 = resid @ resid / (len(df) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {"alpha": coef[0] * 252, "t_alpha": coef[0] / se[0],
            **{f"beta_{k}": c for k, c in zip(xs, coef[1:])}, "r2": r2}
