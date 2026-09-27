"""Step 1-2: how much of Meridian's backtest is hindsight, and what drives the rest.

Fund A = the live picks as they run today (incl. the SGOV sweep, 12553fb).
Fund B = the same structure with every hindsight choice replaced by one fixed
before looking (research/2026-09-returns-study/README.md, "Honest baseline"):
  momentum      megacap15 (NVDA/AVGO/LLY picked in 2026) -> same model on a
                fixed broad-ETF set SPY QQQ IWM EFA EEM GLD IEF TLT
  trend         XLK -> SPY (same model)
  breakouts / mean_reversion / pullback -> T-bills (option C: no edge found
                that survives an honest split; the live sweep already parks
                them in SGOV while flat)
  volatility    SVXY vix_band kept (instrument, not a stock pick)
  long_term_etf, sector_rotation kept (ETF sets)
  cash reserve  0 in both (as live)
Then decomposition: regress each fund's daily returns on SPY, and SPY + IEF.

    PYTHONPATH=. python research/2026-09-returns-study/step1_baseline.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "research/2026-09-returns-study")
from common import PERIODS, SUB, hold, index, regress, sleeve, stats, table, tbill  # noqa: E402

import pandas as pd  # noqa: E402

from meridian.portfolio.live_picks import live_pick_weight, load_live_picks  # noqa: E402

SWEPT = {"breakouts", "mean_reversion", "pullback_continuation"}
HONEST_ETFS = "SPY_QQQ_IWM_EFA_EEM_GLD_IEF_TLT"


def main() -> int:
    picks = load_live_picks()
    fams = set(picks)
    w = {f: live_pick_weight(f, fams) for f in picks}
    tb = tbill()

    parts_a, parts_b = {}, {}
    for f, p in picks.items():
        r, e = sleeve(f, p["model"], p["symbol"])
        parts_a[f] = r + (1 - e) * tb if f in SWEPT else r
    parts_b = dict(parts_a)
    parts_b["momentum"], _ = sleeve("momentum", picks["momentum"]["model"], HONEST_ETFS)
    parts_b["trend_following"], _ = sleeve("trend_following", picks["trend_following"]["model"], "SPY")
    for f in SWEPT:
        parts_b[f] = tb

    fund_a = sum(w[f] * parts_a[f] for f in picks)
    fund_b = sum(w[f] * parts_b[f] for f in picks)
    spy, qqq, ief = hold("SPY"), hold("QQQ"), hold("IEF")
    print("weights:", {f: round(v, 3) for f, v in w.items()})
    print(table({"A: live picks as run (hindsight)": fund_a,
                 "B: honest baseline": fund_b,
                 "SPY": spy, "QQQ": qqq, "60/40 SPY/IEF": 0.6 * spy + 0.4 * ief},
                baseline="B: honest baseline"))

    print("\n### Per-sleeve, 2011-2026/06 (A = live pick, B = honest substitute)")
    print("| Sleeve | Weight | A CAGR | A Sharpe | B CAGR | B Sharpe |\n|---|---:|---:|---:|---:|---:|")
    for f in sorted(picks):
        sa, sb = stats(parts_a[f], "2011-01-01", "2026-06-26"), stats(parts_b[f], "2011-01-01", "2026-06-26")
        print(f"| {f} | {w[f]:.3f} | {sa['cagr']:+.1%} | {sa['sharpe']:.2f} | {sb['cagr']:+.1%} | {sb['sharpe']:.2f} |")

    print("\n### Decomposition (daily OLS; alpha annualized)")
    print("| Fund | Period | Alpha | t(alpha) | Beta SPY | Beta IEF | R² |\n|---|---|---:|---:|---:|---:|---:|")
    for label, y in (("A live", fund_a), ("B honest", fund_b)):
        for name, a, b in PERIODS:
            g = regress(y - tb, {"SPY": spy - tb, "IEF": ief - tb}, a, b)
            print(f"| {label} | {name} | {g['alpha']:+.1%} | {g['t_alpha']:+.1f} | "
                  f"{g['beta_SPY']:.2f} | {g['beta_IEF']:.2f} | {g['r2']:.2f} |")

    out = pd.DataFrame({"fund_a": fund_a, "fund_b": fund_b, **{f"b_{k}": v for k, v in parts_b.items()},
                        "tbill": tb})
    out.to_csv("research/2026-09-returns-study/step1_returns.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
