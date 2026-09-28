"""Run research/plans/structure_2008_check.json once through the skill lab.

Does the C1 structure choice survive 2008? The baseline weights are hard-coded
(the allocation table on the C1 branch is different); the idle share of every
timed sleeve earns T-bills on both sides; long_term_etf rotates QQQ/IEF only
until KMLM exists (2021) in both the baseline and C1.

    PYTHONPATH=. python research/plans/run_structure_2008_check.py
"""

from __future__ import annotations

import sys

sys.path[:0] = ["research/2026-09-returns-study", "research/plans"]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402

common.START = "2006-01-01"
common.index = lambda: common.tr("SPY").loc["2007-06-01":].index
from common import hold, index, sleeve, tbill, tr  # noqa: E402

from meridian.validation import skill_lab as lab  # noqa: E402

PLAN = "research/plans/structure_2008_check.json"
A, B = "2008-01-01", "2026-06-26"
PERIODS = [("2008-10", "2008-01-01", "2010-12-31"), ("2011-15", "2011-01-01", "2015-12-31"),
           ("2016-20", "2016-01-01", "2020-12-31"), ("2021-26/06", "2021-01-01", B)]
HONEST_ETFS = "SPY_QQQ_IWM_EFA_EEM_GLD_IEF_TLT"
SECTORS = "XLK XLV XLF XLE XLI XLY XLP XLU XLRE XLB XLC".split()

# Original 9-sleeve weights, hard-coded: (weight, family, model, symbol); None = T-bills.
BASELINE = [
    (0.25, "long_term_etf", "ma_bond_rotation_cs_b126", "QQQ_IEF_KMLM"),
    (0.15, "momentum", "dual_momentum_long_only", HONEST_ETFS),
    (0.10, "sector_rotation", "relative_strength_b05", "SECTORS"),
    (0.075, "trend_following", "ma_trend_long_only", "SPY"),
    (0.05, "volatility", "vix_band", "SVXY_^VIX"),
    (0.375, None, None, None),   # mean_reversion 15 + pullback 10 + breakouts 7.5 + cash 5
]


def stats(r, a=A, b=B):
    r = r.loc[a:b].fillna(0.0)
    eq = (1 + r).cumprod()
    sd = r.std()
    return {"cagr": eq.iloc[-1] ** (252 / len(r)) - 1, "sharpe": r.mean() / sd * np.sqrt(252) if sd else np.nan,
            "vol": sd * np.sqrt(252), "maxdd": (eq / eq.cummax() - 1).min()}


def scaled(r, tb, k, spread):
    return k * r + (1 - k) * tb if k <= 1 else k * r - (k - 1) * (tb + spread / 252)


def solve_k(r, tb, budget, spread):
    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if stats(scaled(r, tb, mid, spread))["maxdd"] < budget:
            hi = mid
        else:
            lo = mid
    return lo


def ew(syms):
    return pd.concat([hold(s).where(tr(s).reindex(index()).notna()) for s in syms], axis=1).mean(axis=1).fillna(0)


def main() -> int:
    plan = lab.load_plan(PLAN)
    tb = tbill()
    parts, timed = {}, {}
    base = pd.Series(0.0, index=index())
    for w, fam, model, sym in BASELINE:
        if fam is None:
            base += w * tb
            continue
        r, e = sleeve(fam, model, sym)
        parts[fam] = r + (1 - e) * tb          # idle share in T-bills
        timed[fam] = (r, e)
        base += w * parts[fam]

    spy, ief = hold("SPY"), hold("IEF")
    sixty = 0.6 * spy + 0.4 * ief
    c1 = 0.5 * sixty + 0.25 * parts["long_term_etf"] + 0.25 * tb
    swap = {"sector_rotation": ew(SECTORS), "momentum": ew(HONEST_ETFS.split("_")), "trend_following": spy}
    wmap = {fam: w for w, fam, *_ in BASELINE if fam}
    c3 = base + sum(wmap[f] * (swap[f] - parts[f]) for f in swap)
    cands = {"C1": c1, "C3": c3}

    rep = lab.run(plan, {"SPY": tr("SPY")}, lambda p, cfg: cands[cfg["candidate"]],
                  lambda p: {"baseline": base}, tb, final=True)
    rule = plan.data["pass_rule"]

    rows = {"Baseline (8 sleeves, honest)": base, "C1 core": c1, "C3 (post-hoc holds)": c3,
            "SPY": spy, "60/40 SPY/IEF": sixty}
    print(f"plan {plan.id} fingerprint {plan.fingerprint[:12]}, final run logged; window {A}..{B}\n")
    print("### Unscaled, 2008-01..2026-06\n| Portfolio | CAGR | Vol | Sharpe | Max DD |\n|---|---:|---:|---:|---:|")
    for k, r in rows.items():
        s = stats(r)
        lab_ = f"**{k}**" if k.startswith("Baseline") else k
        print(f"| {lab_} | {s['cagr']:+.1%} | {s['vol']:.1%} | {s['sharpe']:.2f} | {s['maxdd']:+.1%} |")

    print("\n### By period: CAGR / max DD\n| Portfolio | " + " | ".join(p for p, _, _ in PERIODS) + " |\n|---|"
          + "---:|" * len(PERIODS))
    for k, r in rows.items():
        cells = [f"{stats(r, a, b)['cagr']:+.1%} / {stats(r, a, b)['maxdd']:+.1%}" for _, a, b in PERIODS]
        print(f"| {k} | " + " | ".join(cells) + " |")

    print(f"\n### Matched {rule['budget']:.0%} worst drawdown, 2008-01..2026-06 (k = exposure; >1 is margin)")
    print("| Portfolio | k | CAGR (borrow +1.5%) | k | CAGR (borrow +3.0%) |\n|---|---:|---:|---:|---:|")
    at = {}
    for k_, r in rows.items():
        k1 = solve_k(r, tb, rule["budget"], rule["borrow_spread_primary"])
        k2 = solve_k(r, tb, rule["budget"], rule["borrow_spread_sensitivity"])
        c_1 = stats(scaled(r, tb, k1, rule["borrow_spread_primary"]))["cagr"]
        c_2 = stats(scaled(r, tb, k2, rule["borrow_spread_sensitivity"]))["cagr"]
        at[k_] = c_1
        print(f"| {k_} | {k1:.2f} | {c_1:+.1%} | {k2:.2f} | {c_2:+.1%} |")

    gap = at["C1 core"] - at["Baseline (8 sleeves, honest)"]
    verdict = "REAFFIRMED" if gap >= -rule["cagr_margin"] else "REOPEN"
    print(f"\nDECISION (plan rule): C1 minus baseline at -15% = {gap:+.2%}/yr -> {verdict}")
    print("alpha vs baseline (NW, unscaled): " + ", ".join(
        f"{x['config']['candidate']} {x['alpha']['alpha']:+.2%} (t {x['alpha']['t_alpha']:+.2f})" for x in rep["results"]))

    print("\n### Each timed sleeve vs holding its own instruments, 2008-01..2010-12 (CAGR / max DD)")
    print("| Sleeve | Timed (idle in T-bills) | Held |\n|---|---:|---:|")
    holds = {"long_term_etf": ew(["QQQ", "IEF"]), "momentum": swap["momentum"],
             "sector_rotation": swap["sector_rotation"], "trend_following": spy}
    for fam, h in holds.items():
        t_ = stats(parts[fam], "2008-01-01", "2010-12-31")
        h_ = stats(h, "2008-01-01", "2010-12-31")
        print(f"| {fam} | {t_['cagr']:+.1%} / {t_['maxdd']:+.1%} | {h_['cagr']:+.1%} / {h_['maxdd']:+.1%} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
