"""Does it matter whether trend 6's signal uses plain close (what the live runner feeds models)
or the dividend-adjusted close (what the sweep used)? Returns are total returns either way.

    PYTHONPATH=. python research/plans/trend6_signal_input_check.py
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from meridian.data import load_ohlcv  # noqa: E402

U8 = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "IEF", "TLT"]
raw = {s: load_ohlcv(s, "2006-01-01", use_cache=False) for s in U8}
FULL = raw["SPY"].index
adj = pd.DataFrame({s: raw[s]["adj_close"] for s in U8}).reindex(FULL).ffill(limit=5)
close = pd.DataFrame({s: raw[s]["close"] for s in U8}).reindex(FULL).ffill(limit=5)
rf = (load_ohlcv("^IRX", "2006-01-01", use_cache=False)["close"].reindex(FULL).ffill() / 100 / 252).shift(1).fillna(0)
ret = adj.pct_change()
ME = pd.DatetimeIndex(pd.Series(FULL, index=FULL).groupby([FULL.year, FULL.month]).max().values)


def run(sig_prices):
    M = sig_prices.loc[ME]
    T = (M > M.rolling(6).mean()).astype(float) / 8
    w = T.reindex(FULL).ffill().shift(1).fillna(0.0)
    tc = T.diff().abs().sum(axis=1).reindex(FULL).fillna(0).shift(1).fillna(0) * 10 / 1e4
    return ((w * ret.fillna(0)).sum(axis=1) + (1 - w.sum(axis=1)) * rf - tc).loc["2008-01-01":"2026-06-26"], T


def st(r):
    eq = (1 + r).cumprod()
    return eq.iloc[-1] ** (252 / len(r)) - 1, (eq / eq.cummax() - 1).min()


ra, Ta = run(adj)
rc, Tc = run(close)
diff = (Ta.loc["2008":"2026-06"] != Tc.loc["2008":"2026-06"])
print(f"adj_close signals: CAGR {st(ra)[0]:+.2%}, max DD {st(ra)[1]:+.2%}")
print(f"close signals:     CAGR {st(rc)[0]:+.2%}, max DD {st(rc)[1]:+.2%}")
print(f"month-end decisions that differ: {int(diff.values.sum())} of {diff.size} "
      f"({diff.values.mean():.1%}); by ETF: {diff.sum().to_dict()}")
