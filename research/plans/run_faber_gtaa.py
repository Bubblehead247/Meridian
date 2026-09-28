"""Run the pre-registered plan research/plans/faber_gtaa_5etf.json through the skill lab.

Weights are held fixed at their month-end targets between rebalances (for the
strategy and the benchmark alike); costs are charged on each month's change.

    PYTHONPATH=. python research/plans/run_faber_gtaa.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from meridian.data import load_ohlcv
from meridian.validation import skill_lab as lab

PLAN = "research/plans/faber_gtaa_5etf.json"
ETFS = ["SPY", "EFA", "IEF", "GSG", "VNQ"]
COST = 10 / 1e4


def load():
    px = {s: load_ohlcv(s, "2006-01-01", use_cache=False)["adj_close"].dropna() for s in ETFS}
    irx = load_ohlcv("^IRX", "2006-01-01", use_cache=False)["close"]
    idx = px["SPY"].index
    rf = (irx.reindex(idx).ffill() / 100 / 252).shift(1).fillna(0.0)
    return px, rf


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def strategy_factory(rf: pd.Series):
    def strategy(prices: dict[str, pd.Series], cfg: dict) -> pd.Series:
        n = int(cfg["sma_months"])
        df = pd.DataFrame(prices).dropna()
        rets = df.pct_change().fillna(0.0)
        me = month_ends(df.index)
        monthly = df.loc[me]
        sma = monthly.rolling(n).mean()
        target = ((monthly > sma) & sma.notna()).astype(float) * (1.0 / len(df.columns))
        # Weights decided at a month-end close apply from the next day.
        w = target.reindex(df.index).ffill().shift(1).fillna(0.0)
        cash = 1.0 - w.sum(axis=1)
        turnover = target.diff().abs().sum(axis=1).reindex(df.index).shift(1).fillna(0.0)
        return (w * rets).sum(axis=1) + cash * rf.reindex(df.index).fillna(0.0) - turnover * COST
    return strategy


def ew_hold(prices: dict[str, pd.Series]) -> pd.Series:
    df = pd.DataFrame(prices).dropna()
    return df.pct_change().fillna(0.0).mean(axis=1)


def stats(r: pd.Series) -> str:
    eq = (1 + r).cumprod()
    return (f"CAGR {eq.iloc[-1] ** (252 / len(r)) - 1:+.1%}  vol {r.std() * np.sqrt(252):.1%}  "
            f"Sharpe {r.mean() / r.std() * np.sqrt(252):.2f}  maxDD {(eq / eq.cummax() - 1).min():+.1%}")


def main() -> int:
    plan = lab.load_plan(PLAN)  # refuses an uncommitted or edited plan
    px, rf = load()
    strat = strategy_factory(rf)
    rep = lab.run(plan, px, strat, lambda p: {"EW hold": ew_hold(p)}, rf, final=True)

    a, b = rep["window"]
    best = rep["best"]
    r = best["returns"]
    ew = ew_hold(px).loc[a:b]
    spy = px["SPY"].pct_change().loc[a:b].fillna(0)
    ief = px["IEF"].pct_change().loc[a:b].fillna(0)
    print(f"plan {rep['plan']}  fingerprint {rep['fingerprint'][:12]}  window {a}..{b}  "
          f"({rep['years']:.1f} years)")
    print(f"power: detects an information ratio >= {rep['mde_information_ratio']:.2f} at the plan's t")
    al = best["alpha"]
    print(f"\nPRIMARY (vs EW hold of the same 5 ETFs): alpha {al['alpha']:+.2%}/yr, "
          f"NW t {al['t_alpha']:+.2f}, beta {al['betas']['EW hold']:.2f}, "
          f"IR {al['information_ratio']:+.2f}, R2 {al['r2']:.2f}")
    print(f"VERDICT: {'PASS' if rep['passed'] else 'not passed'}")

    sec = lab.alpha_test(r, {"SPY": spy, "IEF": ief}, rf.loc[a:b])
    print(f"\nsecondary (vs SPY + IEF): alpha {sec['alpha']:+.2%}/yr, NW t {sec['t_alpha']:+.2f}, "
          f"betas SPY {sec['betas']['SPY']:.2f} IEF {sec['betas']['IEF']:.2f}")
    print("\n| Portfolio | Stats |\n|---|---|")
    print(f"| **EW hold of the 5 ETFs (baseline)** | {stats(ew)} |")
    print(f"| Faber 10-month rule | {stats(r)} |")
    print(f"| SPY | {stats(spy)} |")
    print(f"| 60/40 SPY/IEF | {stats(0.6 * spy + 0.4 * ief)} |")
    print("\nBy period (CAGR): strategy / EW hold")
    for lo, hi in (("2007-05-01", "2012-12-31"), ("2013-01-01", "2019-12-31"), ("2020-01-01", b)):
        s, e = r.loc[lo:hi], ew.loc[lo:hi]
        f = lambda x: (1 + x).prod() ** (252 / len(x)) - 1  # noqa: E731
        print(f"  {lo[:4]}-{hi[:4]}: {f(s):+.1%} / {f(e):+.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
