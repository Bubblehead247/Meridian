"""Run research/plans/meridian_structure_2026_10.json once through the skill lab.

Baseline and parts come from research/2026-09-returns-study/step1_returns.csv
(honest fund B). Mixes hold fixed weights daily (the same for all rows).

    PYTHONPATH=. python research/plans/run_structure_test.py
"""

from __future__ import annotations

import json
import sys

sys.path[:0] = ["research/2026-09-returns-study", "research/plans"]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import hold, index, tr  # noqa: E402
from meridian.portfolio.live_picks import live_pick_weight, load_live_picks  # noqa: E402
from meridian.validation import skill_lab as lab  # noqa: E402
import run_faber_gtaa as faber  # noqa: E402

PLAN = "research/plans/meridian_structure_2026_10.json"


def ew(syms):
    return pd.concat([hold(s).where(tr(s).reindex(index()).notna()) for s in syms], axis=1).mean(axis=1).fillna(0)


def stats(r, a, b):
    r = r.loc[a:b].fillna(0)
    eq = (1 + r).cumprod()
    return {"cagr": eq.iloc[-1] ** (252 / len(r)) - 1, "sharpe": r.mean() / r.std() * np.sqrt(252),
            "maxdd": (eq / eq.cummax() - 1).min()}


def main() -> int:
    plan = lab.load_plan(PLAN)
    df = pd.read_csv("research/2026-09-returns-study/step1_returns.csv", index_col=0, parse_dates=True)
    tb, base, lte = df["tbill"], df["fund_b"], df["b_long_term_etf"]
    spy, ief = hold("SPY"), hold("IEF")
    sixty = 0.6 * spy + 0.4 * ief

    px, rf = faber.load()
    faber_r = faber.strategy_factory(rf)(px, {"sma_months": 10}).reindex(df.index).fillna(0.0)
    # Faber needs 10 month-ends of GSG first: before 2007-05 it's all T-bills (irrelevant here, test starts 2011).

    picks = load_live_picks()
    w = {f: live_pick_weight(f, set(picks)) for f in picks}
    swap = {"sector_rotation": ew("XLK XLV XLF XLE XLI XLY XLP XLU XLRE XLB XLC".split()),
            "momentum": ew("SPY QQQ IWM EFA EEM GLD IEF TLT".split()),
            "trend_following": spy}
    c3 = base + 0.05 * tb + sum(w[f] * (swap[f] - df[f"b_{f}"]) for f in swap)

    cands = {"C1": 0.5 * sixty + 0.25 * lte + 0.25 * tb,
             "C2": 0.5 * faber_r + 0.25 * lte + 0.25 * sixty,
             "C3": c3}
    sleeves = {"C1": 2, "C2": 3, "C3": 5}

    rep = lab.run(plan, {"SPY": tr("SPY")}, lambda p, cfg: cands[cfg["candidate"]],
                  lambda p: {"baseline": base}, tb, final=True)

    rule = plan.data["pass_rule"]
    periods = plan.data["periods"]
    print(f"plan {plan.id} fingerprint {plan.fingerprint[:12]}, final run logged\n")
    print("| Portfolio | 2011-15 Sharpe / DD | 2016-20 Sharpe / DD | 2021-26/06 Sharpe / DD | Full CAGR / Sharpe / DD | Non-inferior? |")
    print("|---|---|---|---|---|---|")

    def cells(r):
        return [f"{stats(r, a, b)['sharpe']:.2f} / {stats(r, a, b)['maxdd']:+.1%}" for a, b in periods]

    full = lambda r: stats(r, "2011-01-01", "2026-06-26")  # noqa: E731
    fb = full(base)
    print(f"| **Baseline (honest 8 sleeves)** | " + " | ".join(cells(base)) +
          f" | {fb['cagr']:+.1%} / {fb['sharpe']:.2f} / {fb['maxdd']:+.1%} | — |")
    winners = []
    for k, r in cands.items():
        ok = all(stats(r, a, b)["sharpe"] >= stats(base, a, b)["sharpe"] - rule["sharpe_margin"]
                 and stats(r, a, b)["maxdd"] >= stats(base, a, b)["maxdd"] - rule["max_drawdown_margin"]
                 for a, b in periods)
        fr = full(r)
        if ok:
            winners.append((sleeves[k], -fr["sharpe"], k))
        label = f"{k}{' (post-hoc)' if k == 'C3' else ''}"
        print(f"| {label} | " + " | ".join(cells(r)) +
              f" | {fr['cagr']:+.1%} / {fr['sharpe']:.2f} / {fr['maxdd']:+.1%} | {'yes' if ok else 'no'} |")
    print("\nalpha vs baseline (NW): " + ", ".join(
        f"{x['config']['candidate']} {x['alpha']['alpha']:+.2%} (t {x['alpha']['t_alpha']:+.2f})"
        for x in rep["results"]))
    print(f"\nWINNER (fewest sleeves, then Sharpe): {sorted(winners)[0][2] if winners else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
