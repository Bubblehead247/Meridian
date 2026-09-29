"""Run research/plans/trend6_small_mid_caps.json once through the skill lab.

    PYTHONPATH=. python research/plans/run_trend6_small_mid_caps.py
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402

from meridian.data import load_ohlcv  # noqa: E402
from meridian.validation import skill_lab as lab  # noqa: E402

PLAN = "research/plans/trend6_small_mid_caps.json"
A, B = "2008-01-01", "2026-06-26"
BUDGET = -0.15


def stats(r):
    eq = (1 + r).cumprod()
    return eq.iloc[-1] ** (252 / len(r)) - 1, (eq / eq.cummax() - 1).min()


def at_budget(r, t):
    """CAGR after scaling toward T-bills (k <= 1) so the worst drawdown is no deeper than the budget."""
    if stats(r)[1] >= BUDGET:
        return stats(r)[0], 1.0
    lo, hi = 0.0, 1.0
    for _ in range(40):
        k = (lo + hi) / 2
        lo, hi = (k, hi) if stats(k * r + (1 - k) * t)[1] >= BUDGET else (lo, k)
    return stats(lo * r + (1 - lo) * t)[0], lo


def main() -> int:
    plan = lab.load_plan(PLAN)
    tickers = sorted({s for c in plan.data["configs"] for s in c["etfs"]})
    idx = load_ohlcv("SPY", "2006-01-01", use_cache=False).index
    px = pd.DataFrame({s: load_ohlcv(s, "2006-01-01", use_cache=False)["adj_close"] for s in tickers}) \
        .reindex(idx).ffill(limit=5)
    rf = (load_ohlcv("^IRX", "2006-01-01", use_cache=False)["close"].reindex(idx).ffill() / 100 / 252) \
        .shift(1).fillna(0.0)
    ret = px.pct_change()
    ends = pd.DatetimeIndex(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values)

    def trend(etfs):
        M = px.loc[ends, etfs]
        T = (M > M.rolling(6).mean()).astype(float) / len(etfs)
        w = T.reindex(idx).ffill().shift(1).fillna(0.0)
        tc = T.diff().abs().sum(axis=1).reindex(idx).fillna(0.0).shift(1).fillna(0.0) * 10 / 1e4
        return (w * ret[etfs].fillna(0.0)).sum(axis=1) + (1 - w.sum(axis=1)) * rf - tc

    R = {c["id"]: trend(c["etfs"]) for c in plan.data["configs"]}
    c0, d0 = stats(R["BASE"].loc[A:B])
    if not (abs(c0 - 0.0702) <= 0.0015 and abs(d0 + 0.0964) <= 0.003):
        print(f"GATE FAIL: BASE {c0:+.2%} / {d0:+.2%}")
        return 1
    print(f"GATE BASE reproduces trend 6: {c0:+.2%} / {d0:+.2%} PASS\n")

    t = rf.loc[A:B]
    periods = plan.data["periods"]
    print("| Portfolio | CAGR | Max DD | At -15% (exposure) | " +
          " | ".join(f"{a[:4]}-{b[2:4]}" for a, b in periods) + " |\n|---|---:|---:|---:|" + "---:|" * len(periods))
    res = {}
    for cid, r in R.items():
        rr = r.loc[A:B]
        c, d = stats(rr)
        ob, k = at_budget(rr, t)
        per = [stats(r.loc[a:b])[0] for a, b in periods]
        res[cid] = (c, d, ob, per)
        lab_ = f"**{cid} (baseline)**" if cid == "BASE" else cid
        print(f"| {lab_} | {c:+.1%} | {d:+.1%} | {ob:+.1%} ({k:.2f}) | " + " | ".join(f"{p:+.1%}" for p in per) + " |")

    b, u = res["BASE"], res["U10"]
    wins = sum(x > y for x, y in zip(u[3], b[3]))
    adopt = (u[2] - b[2] > 0.005) and wins >= 3 and (u[1] >= b[1] - 0.02)
    print(f"\nDECISION: U10 minus BASE at -15% = {u[2] - b[2]:+.2%}/yr; higher CAGR in {wins} of 4 periods; "
          f"DD {u[1]:+.1%} vs {b[1]:+.1%} -> {'ADOPT' if adopt else 'KEEP U8'}")

    lab.run(plan, {"SPY": px["SPY"]}, lambda p, cfg: R[cfg["id"]], lambda p: {"BASE": R["BASE"]}, rf, final=True)
    print("final run logged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
