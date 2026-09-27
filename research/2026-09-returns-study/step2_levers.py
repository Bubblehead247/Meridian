"""Step 3: return levers, tested on the HONEST baseline (fund B from step 1).

Pass rule, fixed before running (2026-09-27):
  * "free"       CAGR >= baseline AND Sharpe >= baseline in all three
                 sub-periods (2011-15, 2016-20, 2021-26/06).
  * "risk trade" CAGR > baseline in all three sub-periods; its extra drawdown
                 is reported and the user decides whether it is worth it.
  * otherwise    "fails".
Borrowing cost for anything above 100% invested: T-bill + 1.5%/yr
(ASSUMPTION — Alpaca's actual margin rate not checked; paper charges none).

Levers:
  L1  free sweeps: 5% cash reserve and the volatility sleeve's idle share -> T-bills
  L2  move the three no-edge sleeves (32.5%, T-bills in the baseline) to:
        a SPY   b SPY with 200-day trend filter (ma_trend_long_only)
        c long_term_etf strategy   d honest-ETF momentum   e 60/40 SPY/IEF
  L3  volatility targeting on the whole fund: 10% and 12% target, 60-day
      trailing vol, leverage capped at 1.5x, rebalanced daily with 10 bps
      on the change in exposure
  L4  per-sleeve value added: each timing sleeve vs holding what it trades
  L5  volatility sleeve after SVXY's Feb-2018 change (my recollection: to
      -0.5x; checked here only by the change in its daily volatility)

    PYTHONPATH=. python research/2026-09-returns-study/step2_levers.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "research/2026-09-returns-study")
from common import PERIODS, SUB, hold, index, sleeve, stats, table, tbill, tr  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from meridian.portfolio.live_picks import live_pick_weight, load_live_picks  # noqa: E402

BORROW_SPREAD = 0.015 / 252


def verdict(r: pd.Series, base: pd.Series) -> str:
    c = [stats(r, a, b)["cagr"] - stats(base, a, b)["cagr"] for _, a, b in SUB]
    s = [stats(r, a, b)["sharpe"] - stats(base, a, b)["sharpe"] for _, a, b in SUB]
    if all(x >= 0 for x in c) and all(x >= 0 for x in s):
        return "free"
    if all(x > 0 for x in c):
        return "risk trade"
    return "fails"


def vol_target(r: pd.Series, tb: pd.Series, target: float, cap: float = 1.5) -> pd.Series:
    realized = r.rolling(60).std().shift(1) * np.sqrt(252)
    lev = (target / realized).clip(upper=cap).fillna(1.0)
    borrow = (lev - 1).clip(lower=0) * (tb + BORROW_SPREAD)
    idle = (1 - lev).clip(lower=0) * tb  # under-invested share earns T-bills
    cost = lev.diff().abs().fillna(0) * 10 / 1e4
    return lev * r + idle - borrow - cost


def main() -> int:
    df = pd.read_csv("research/2026-09-returns-study/step1_returns.csv", index_col=0, parse_dates=True)
    picks = load_live_picks()
    fams = set(picks)
    w = {f: live_pick_weight(f, fams) for f in picks}
    tb = df["tbill"]
    parts = {f: df[f"b_{f}"] for f in picks}
    base = df["fund_b"]
    no_edge = ["breakouts", "mean_reversion", "pullback_continuation"]
    w_ne = sum(w[f] for f in no_edge)

    # L1
    _, vol_exp = sleeve("volatility", picks["volatility"]["model"], picks["volatility"]["symbol"])
    l1 = base + 0.05 * tb + w["volatility"] * (1 - vol_exp) * tb

    spy, ief = hold("SPY"), hold("IEF")
    spy_trend, _ = sleeve("trend_following", "ma_trend_long_only", "SPY")
    subs = {"a SPY": spy, "b SPY 200-day trend": spy_trend, "c long_term_etf": parts["long_term_etf"],
            "d honest momentum": parts["momentum"], "e 60/40 SPY/IEF": 0.6 * spy + 0.4 * ief}
    rows = {"Baseline (honest fund B)": base, "L1 free sweeps": l1}
    for k, s in subs.items():
        rows[f"L1 + L2{k}"] = l1 - w_ne * tb + w_ne * s
    for t in (0.10, 0.12):
        rows[f"L1 + vol target {t:.0%}"] = vol_target(l1, tb, t)
        rows[f"L1 + L2b + vol target {t:.0%}"] = vol_target(rows["L1 + L2b SPY 200-day trend"], tb, t)

    print(table(rows, baseline="Baseline (honest fund B)"))
    print("\n### Verdicts (pass rule in the docstring)")
    print("| Lever | Verdict | Extra max DD 2011-26/06 |\n|---|---|---:|")
    b_dd = stats(base, "2011-01-01", "2026-06-26")["maxdd"]
    for k, r in rows.items():
        if k.startswith("Baseline"):
            continue
        dd = stats(r, "2011-01-01", "2026-06-26")["maxdd"] - b_dd
        print(f"| {k} | {verdict(r, base)} | {dd:+.1%} |")

    # L4: does each timing sleeve beat holding what it trades?
    print("\n### L4 per-sleeve value added (timing minus holding the same instruments)")
    print("| Sleeve | 2011-15 | 2016-20 | 2021-26/06 |\n|---|---:|---:|---:|")
    holds = {
        "long_term_etf": ("QQQ_IEF_KMLM", "long_term_etf"),
        "sector_rotation": ("XLK_XLV_XLF_XLE_XLI_XLY_XLP_XLU_XLRE_XLB_XLC", "sector_rotation"),
        "momentum (honest ETFs)": ("SPY_QQQ_IWM_EFA_EEM_GLD_IEF_TLT", "momentum"),
        "trend_following (SPY)": ("SPY", "trend_following"),
    }
    for label, (syms, fam) in holds.items():
        r = parts[fam] if fam in parts else None
        if label.startswith("trend"):
            r = spy_trend
        members = syms.split("_")
        # Equal-weight hold of the members that exist on each day.
        h = pd.concat([hold(s).where(tr(s).reindex(index()).notna()) for s in members], axis=1).mean(axis=1).fillna(0)
        cells = [stats(r, a, b)["cagr"] - stats(h, a, b)["cagr"] for _, a, b in SUB]
        print(f"| {label} | " + " | ".join(f"{c:+.1%}" for c in cells) + " |")

    # L5: volatility sleeve before/after SVXY's 2018 change
    sv = tr("SVXY").pct_change()
    print("\n### L5 SVXY daily vol: 2012-2017 "
          f"{sv.loc['2012':'2017'].std() * np.sqrt(252):.0%}, 2018-04..2026 "
          f"{sv.loc['2018-04':'2026'].std() * np.sqrt(252):.0%}")
    v = parts["volatility"]
    for lab, a, b in (("vix_band sleeve 2018-04..2026/06", "2018-04-01", "2026-06-26"),
                      ("T-bills same window", "2018-04-01", "2026-06-26")):
        s = stats(v if lab.startswith("vix") else tb, a, b)
        print(f"  {lab}: CAGR {s['cagr']:+.1%}, Sharpe {s['sharpe']:.2f}, maxDD {s['maxdd']:+.1%}")
    s = stats(spy, "2018-04-01", "2026-06-26")
    print(f"  SPY same window: CAGR {s['cagr']:+.1%}, Sharpe {s['sharpe']:.2f}, maxDD {s['maxdd']:+.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
