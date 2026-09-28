"""Step 4: the most return each standard portfolio gives inside a -15% drawdown.

Each portfolio is scaled with T-bills (k < 1) or margin (k > 1, borrowing at
T-bill + 1.5%/yr — an ASSUMPTION; Alpaca's rate not checked, and paper may
charge none) so that its worst drawdown 2008-01..2026-06 is exactly -15%.
Calibrating k on the same history it is judged on flatters every row equally:
compare rows with each other, not with a promise.

Portfolios (standard, chosen before running): SPY; QQQ; 60/40 SPY/IEF;
the C1 core; Faber 5-ETF; Faber on SPY alone (10-month SMA); an "All
Weather"-style mix 30% SPY / 40% TLT / 15% IEF / 7.5% GLD / 7.5% GSG (the widely
published retail approximation — not Bridgewater's actual fund).
Total returns, daily fixed weights, 10 bps on Faber turnover.

    PYTHONPATH=. python research/2026-09-returns-study/step4_return_at_budget.py
"""

from __future__ import annotations

import sys

sys.path[:0] = ["research/2026-09-returns-study", "research/plans"]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402

# The shared helpers start in mid-2010. This step must include 2008, so load
# from 2006 and index from 2007-06 (Faber's first full signal is 2007-04-30).
common.START = "2006-01-01"
common.index = lambda: common.tr("SPY").loc["2007-06-01":].index
from common import hold, index, sleeve, tbill  # noqa: E402
import run_faber_gtaa as faber  # noqa: E402

A, B = "2008-01-01", "2026-06-26"
BUDGET = -0.15
SPREAD = 0.015 / 252


def scaled(r: pd.Series, tb: pd.Series, k: float) -> pd.Series:
    if k <= 1:
        return k * r + (1 - k) * tb
    return k * r - (k - 1) * (tb + SPREAD)


def maxdd(r: pd.Series) -> float:
    eq = (1 + r.loc[A:B].fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def solve_k(r, tb) -> float:
    lo, hi = 0.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if maxdd(scaled(r, tb, mid)) < BUDGET:
            hi = mid
        else:
            lo = mid
    return lo


def cagr(r, a=A, b=B):
    r = r.loc[a:b].fillna(0)
    return (1 + r).prod() ** (252 / len(r)) - 1


def main() -> int:
    tb = tbill()
    spy, qqq, ief, tlt, gld, gsg = (hold(s) for s in ("SPY", "QQQ", "IEF", "TLT", "GLD", "GSG"))
    lte, _ = sleeve("long_term_etf", "ma_bond_rotation_cs_b126", "QQQ_IEF_KMLM")
    px, rf = faber.load()
    fab5 = faber.strategy_factory(rf)(px, {"sma_months": 10}).reindex(index()).fillna(0)
    fab_spy = faber.strategy_factory(rf)({"SPY": px["SPY"]}, {"sma_months": 10}).reindex(index()).fillna(0)
    sixty = 0.6 * spy + 0.4 * ief
    ports = {
        "SPY": spy, "QQQ": qqq, "60/40 SPY/IEF": sixty,
        "C1 core (50% 60/40, 25% LTE, 25% T-bills)": 0.5 * sixty + 0.25 * lte + 0.25 * tb,
        "Faber 5-ETF": fab5, "Faber on SPY alone": fab_spy,
        "All-Weather-style 30/40/15/7.5/7.5": 0.30 * spy + 0.40 * tlt + 0.15 * ief + 0.075 * gld + 0.075 * gsg,
    }
    rows = []
    for name, r in ports.items():
        k = solve_k(r, tb)
        s = scaled(r, tb, k)
        rows.append((cagr(s), name, k, maxdd(r), cagr(s, "2008-01-01", "2015-12-31"),
                     cagr(s, "2016-01-01", "2020-12-31"), cagr(s, "2021-01-01", B)))
    print(f"Scaled to a {BUDGET:.0%} max drawdown over {A[:4]}-2026/06 (k = exposure; >1 is margin)\n")
    print("| Portfolio | Exposure k | Unscaled max DD | CAGR at -15% | 2008-15 | 2016-20 | 2021-26/06 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for c, name, k, dd, p1, p2, p3 in sorted(rows, reverse=True):
        lab = f"**{name}**" if name.startswith("C1") else name
        print(f"| {lab} | {k:.2f} | {dd:+.1%} | {c:+.1%} | {p1:+.1%} | {p2:+.1%} | {p3:+.1%} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
