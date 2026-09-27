"""Option C: re-select the single-asset live picks with long-only backtests.

Method fixed before any result was seen (2026-09-27):
  * Families: breakouts, mean_reversion, pullback_continuation, trend_following
    (the sleeves whose live pick is one symbol + one single-asset model).
  * Candidates: every registered single-asset model of the family x every
    symbol the 6/28 selection tested for that family (saved_strategies/),
    crypto excluded (the bot trades US equities only).
  * Backtest: long-only (the engine default since fe9fca1), 10 bps per side,
    next-open fills, whole history, then sliced.
  * Select on in-sample 2011-01-01 .. 2020-12-31: best Sharpe among candidates
    with >= 5 years of in-sample history and >= 10 entries in-sample.
  * Judge on out-of-sample 2021-01-01 .. 2026-06-26 (the 6/28 pick date).
  * Also reported, labelled in-sample: the best full-sample (2011..2026/06)
    Sharpe, which is roughly how the 6/28 picks were made.

    PYTHONPATH=. python research/2026-09-strategy-review/reselect_picks.py
"""

from __future__ import annotations

import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from meridian.data import load_ohlcv
from meridian.experiments.gauntlet import _single_asset_models
from meridian.pipeline.records import load_records
from meridian.portfolio.live_picks import load_live_picks

FAMILIES = ["breakouts", "mean_reversion", "pullback_continuation", "trend_following"]
IS = ("2011-01-01", "2020-12-31")
OOS = ("2021-01-01", "2026-06-26")
FULL = ("2011-01-01", "2026-06-26")
COST = 10.0
MIN_IS_YEARS = 5
MIN_IS_ENTRIES = 10


def metrics(r: pd.Series, pos: pd.Series, a: str, b: str) -> dict:
    r = r.loc[a:b].dropna()
    p = pos.loc[a:b]
    if len(r) < 60:
        return {"years": len(r) / 252}
    eq = (1 + r).cumprod()
    sd = r.std()
    return {
        "years": len(r) / 252,
        "cagr": eq.iloc[-1] ** (252 / len(r)) - 1,
        "sharpe": r.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
        "maxdd": (eq / eq.cummax() - 1).min(),
        "entries": int(((p > 0) & (p.shift(1).fillna(0) <= 0)).sum()),
        "in_mkt": float((p > 0).mean()),
    }


def main() -> int:
    live = load_live_picks()
    symbols = defaultdict(set)
    for rec in load_records():
        if rec.family in FAMILIES and "_" not in rec.symbol:  # _USDT crypto, baskets
            symbols[rec.family].add(rec.symbol)
    models = defaultdict(list)
    for fam, name in _single_asset_models():
        if fam in FAMILIES:
            models[fam].append(name)

    from meridian.families import create_model

    rows = []
    for fam in FAMILIES:
        for sym in sorted(symbols[fam]):
            try:
                bars = load_ohlcv(sym, "2010-01-01")
            except Exception as exc:
                print(f"  skip {sym}: {exc}")
                continue
            for name in models[fam]:
                try:
                    res = create_model(fam, name).backtest(bars["close"], cost_bps=COST, bars=bars)
                except Exception as exc:
                    print(f"  {fam}/{name} on {sym} failed: {exc}")
                    continue
                bh = bars["close"].pct_change()
                row = {"family": fam, "model": name, "symbol": sym,
                       "live": live.get(fam, {}) == {"model": name, "symbol": sym}}
                for tag, (a, b) in (("is", IS), ("oos", OOS), ("full", FULL)):
                    for k, v in metrics(res.returns, res.positions, a, b).items():
                        row[f"{tag}_{k}"] = v
                    row[f"{tag}_bh_cagr"] = metrics(bh, pd.Series(1, index=bh.index), a, b).get("cagr")
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("research/2026-09-strategy-review/reselect_all.csv", index=False)

    pct = lambda v: "" if pd.isna(v) else f"{v:+.1%}"
    num = lambda v: "" if pd.isna(v) else f"{v:.2f}"

    def line(label, r):
        return (f"| {label} | {r['model']} / {r['symbol']} | {num(r.get('is_sharpe'))} | "
                f"{pct(r.get('oos_cagr'))} | {num(r.get('oos_sharpe'))} | {pct(r.get('oos_maxdd'))} | "
                f"{pct(r.get('oos_bh_cagr'))} | {'' if pd.isna(r.get('oos_in_mkt')) else f'{r['oos_in_mkt']:.0%}'} |")

    print(f"candidates: {len(df)} (model x symbol), cost {COST:g} bps, next-open, long-only\n")
    for fam in FAMILIES:
        f = df[df.family == fam]
        eligible = f[(f.is_years >= MIN_IS_YEARS) & (f.is_entries >= MIN_IS_ENTRIES)]
        print(f"## {fam}  ({len(f)} candidates, {len(eligible)} eligible in-sample; "
              f"{f.symbol.nunique()} symbols, {len(models[fam])} models)")
        print("| Row | Model / symbol | IS Sharpe 2011-20 | OOS CAGR 2021-26/06 | OOS Sharpe | OOS maxDD | Hold symbol OOS | OOS in market |")
        print("|---|---|---:|---:|---:|---:|---:|---:|")
        cur = f[f.live]
        if len(cur):
            print(line("**Current live pick**", cur.iloc[0]))
        if len(eligible):
            best = eligible.sort_values("is_sharpe", ascending=False)
            print(line("Selected on 2011-20", best.iloc[0]))
            for i in range(1, min(3, len(best))):
                print(line(f"  runner-up {i}", best.iloc[i]))
        fs = f[f.full_years >= MIN_IS_YEARS].sort_values("full_sharpe", ascending=False)
        if len(fs):
            r = fs.iloc[0]
            print(f"| Best full-sample (in-sample!) | {r['model']} / {r['symbol']} | full Sharpe "
                  f"{num(r['full_sharpe'])} | full CAGR {pct(r['full_cagr'])} | | | | |")
        # How the selection rule does on average: median OOS of the top-5 IS picks.
        if len(eligible) >= 5:
            top5 = eligible.sort_values("is_sharpe", ascending=False).head(5)
            print(f"\n  top-5 by IS Sharpe: median OOS CAGR {pct(top5.oos_cagr.median())}, "
                  f"median OOS Sharpe {num(top5.oos_sharpe.median())}; all eligible: median OOS "
                  f"Sharpe {num(eligible.oos_sharpe.median())}")
            rho = eligible[["is_sharpe", "oos_sharpe"]].corr(method="spearman").iloc[0, 1]
            print(f"  rank correlation IS Sharpe -> OOS Sharpe across eligible: {rho:+.2f}")
        late = sorted(set(f[f.is_years < MIN_IS_YEARS].symbol))
        if late:
            print(f"  not selectable (< {MIN_IS_YEARS}y before 2021): {', '.join(late)}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
