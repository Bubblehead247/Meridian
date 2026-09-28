"""Descriptive follow-up to the sweep (no selection): the recommended deployable model and the
overall in-sample winner, by period. Re-derives both exactly as run_sweep.py does and first checks
it reproduces the sweep's full-sample figures.

    PYTHONPATH=. python research/2026-10-sweep/periods.py
"""
import sys
import warnings

sys.path[:0] = ["research/2026-09-returns-study"]
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402

common.START = "2006-01-01"
from common import tr  # noqa: E402
from meridian.data import load_ohlcv  # noqa: E402

U8 = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "IEF", "TLT"]
COST = 10 / 1e4
FULL = tr("SPY").index
P = pd.DataFrame({s: tr(s).reindex(FULL).ffill(limit=5) for s in U8})
rf = (load_ohlcv("^IRX", "2006-01-01", use_cache=False)["close"].reindex(FULL).ffill() / 100 / 252).shift(1).fillna(0)
ret = P.pct_change()
ME = pd.DatetimeIndex(pd.Series(FULL, index=FULL).groupby([FULL.year, FULL.month]).max().values)
M = P.loc[ME]
TBm = (1 + rf).cumprod().loc[ME]


def run(T):
    w = T.reindex(FULL).ffill().shift(1).fillna(0.0)
    tc = T.fillna(0).diff().abs().sum(axis=1).reindex(FULL).fillna(0).shift(1).fillna(0) * COST
    return ((w * ret.fillna(0)).sum(axis=1) + (1 - w.sum(axis=1)) * rf - tc).loc["2007-05-01":]


trend6 = run((M > M.rolling(6).mean()).astype(float) / 8)
r6, h6 = M / M.shift(6) - 1, TBm / TBm.shift(6) - 1
dual = run(((r6.rank(axis=1, ascending=False, method="first") <= 4) & r6.gt(h6, axis=0)).astype(float) / 4)


def st(r, a, b):
    r = r.loc[a:b]
    eq = (1 + r).cumprod()
    return (eq.iloc[-1] ** (252 / len(r)) - 1, (eq / eq.cummax() - 1).min())


periods = [("2008-26/06", "2008-01-01", "2026-06-26"), ("2008-10", "2008-01-01", "2010-12-31"),
           ("2011-15", "2011-01-01", "2015-12-31"), ("2016-20", "2016-01-01", "2020-12-31"),
           ("2021-26/06", "2021-01-01", "2026-06-26")]
print("| Model | " + " | ".join(p for p, _, _ in periods) + " |\n|---|" + "---:|" * len(periods))
for name, r in (("TREND6-EW-U8", trend6), ("DUALMOM6-K4-U8", dual)):
    print(f"| {name} | " + " | ".join(f"{st(r, a, b)[0]:+.1%} / {st(r, a, b)[1]:+.1%}" for _, a, b in periods) + " |")
avg_in = (dual.loc["2008":"2026-06-26"] != 0).mean()
held = (((r6.rank(axis=1, ascending=False, method="first") <= 4) & r6.gt(h6, axis=0)).loc["2008":"2026-06"].sum(axis=1) / 4)
print(f"\nDUALMOM6-K4-U8 average invested share 2008-26/06: {held.mean():.0%}; months fully in T-bills: {(held == 0).mean():.0%}")
tr_held = ((M > M.rolling(6).mean()).loc["2008":"2026-06"].sum(axis=1) / 8)
print(f"TREND6-EW-U8 average invested share: {tr_held.mean():.0%}; months fully in T-bills: {(tr_held == 0).mean():.0%}")
