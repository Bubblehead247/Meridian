"""Option B: what to do with the single-stock sleeves' mostly idle capital.

The three single-stock sleeves (breakouts/TRGP 7.5%, mean_reversion/SNOW 15%,
pullback/WFRD 10% = 32.5% of the fund) are invested 3-37% of the time, and
option C found their timing rarely beats simply holding (research README §6).

Variants, all on TOTAL returns (adj_close, dividends in — the loader's `close`
is price-only, which would short-change anything held rather than timed),
10 bps per side, fills at the signal close, long-only. Idle cash earns 0
unless swept. T-bills = ^IRX 13-week yield / 252, lagged a day.

  V0  current                                   (baseline)
  V1  sweep idle S-sleeve cash into T-bills
  V2  sweep idle S-sleeve cash into SPY
  V3  sweep idle S-sleeve cash into the long_term_etf sleeve's strategy
  V4  replace the S sleeves with SPY buy-and-hold
  V5  replace the S sleeves with more long_term_etf (25% -> 57.5%)

The S-sleeve picks were chosen in 2026 with hindsight, so V0-V3 carry that
bias and V4/V5 do not — the comparison leans towards the current setup.

    PYTHONPATH=. python research/2026-09-strategy-review/option_b_idle_sleeves.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from meridian.data import load_ohlcv
from meridian.execution.live_runner import _expand_symbol
from meridian.families import create_model
from meridian.portfolio.live_picks import live_pick_weight, load_live_picks

S = ["breakouts", "mean_reversion", "pullback_continuation"]
COST = 10.0
PERIODS = [("2011-2015", "2011-01-01", "2015-12-31"),
           ("2016-2020", "2016-01-01", "2020-12-31"),
           ("2021-2026/06", "2021-01-01", "2026-06-26"),
           ("2011-2026/06", "2011-01-01", "2026-06-26")]

_px: dict[str, pd.Series] = {}


def tr(sym: str) -> pd.Series:
    """Total-return price series (dividend-adjusted close)."""
    if sym not in _px:
        _px[sym] = load_ohlcv(sym, "2010-01-01", use_cache=False)["adj_close"].dropna()
    return _px[sym]


def sleeve(family: str, model_name: str, symbol: str) -> tuple[pd.Series, pd.Series]:
    """(daily net returns, exposure 0..1 held that day) for one live pick."""
    model = create_model(family, model_name)
    if getattr(model, "cross_sectional", False):
        px = {s: tr(s) for s in _expand_symbol(symbol)}
        bt = model.backtest(px, cost_bps=COST)
        return bt.returns, bt.gross_exposure.clip(upper=1.0)
    bt = model.backtest(tr(symbol), cost_bps=COST)
    return bt.returns, bt.positions.abs().clip(upper=1.0)


def stats(r: pd.Series, a: str, b: str) -> dict:
    r = r.loc[a:b].fillna(0.0)
    eq = (1 + r).cumprod()
    return {"cagr": eq.iloc[-1] ** (252 / len(r)) - 1,
            "sharpe": r.mean() / r.std() * np.sqrt(252),
            "maxdd": (eq / eq.cummax() - 1).min()}


def main() -> int:
    picks = load_live_picks()
    fams = set(picks)
    w = {f: live_pick_weight(f, fams) for f in picks}
    ret, exp = {}, {}
    for f, p in picks.items():
        ret[f], exp[f] = sleeve(f, p["model"], p["symbol"])

    idx = tr("SPY").loc["2010-06-01":].index
    ret = {f: r.reindex(idx).fillna(0.0) for f, r in ret.items()}
    exp = {f: e.reindex(idx).fillna(0.0) for f, e in exp.items()}
    tbill = (load_ohlcv("^IRX", "2010-01-01", use_cache=False)["close"]
             .reindex(idx).ffill() / 100 / 252).shift(1).fillna(0.0)
    spy = tr("SPY").pct_change().reindex(idx).fillna(0.0)
    lte = ret["long_term_etf"]

    def fund(sweep: pd.Series | None = None, replace: pd.Series | None = None) -> pd.Series:
        total = pd.Series(0.0, index=idx)
        for f in picks:
            if f in S and replace is not None:
                total += w[f] * replace
            elif f in S and sweep is not None:
                # The idle share of the sleeve earns the sweep asset instead of 0.
                total += w[f] * (ret[f] + (1.0 - exp[f]) * sweep)
            else:
                total += w[f] * ret[f]
        return total

    variants = {
        "V0 current (baseline)": fund(),
        "V1 idle S cash -> T-bills": fund(sweep=tbill),
        "V2 idle S cash -> SPY": fund(sweep=spy),
        "V3 idle S cash -> long_term_etf": fund(sweep=lte),
        "V4 S sleeves -> SPY": fund(replace=spy),
        "V5 S sleeves -> long_term_etf": fund(replace=lte),
    }
    print(f"S sleeves {S}: weight {sum(w[f] for f in S):.3f}; average exposure 2011-26/06: "
          + ", ".join(f"{f} {exp[f].loc['2011':'2026-06-26'].mean():.0%}" for f in S))
    print(f"fund weights sum {sum(w.values()):.3f} (rest is the 5% cash reserve, earning 0)\n")

    for name, a, b in PERIODS:
        print(f"### {name}")
        print("| Variant | CAGR | Sharpe | Max DD |")
        print("|---|---:|---:|---:|")
        for label, r in variants.items():
            s = stats(r, a, b)
            print(f"| {label} | {s['cagr']:+.1%} | {s['sharpe']:.2f} | {s['maxdd']:+.1%} |")
        s = stats(spy, a, b)
        print(f"| SPY alone (reference) | {s['cagr']:+.1%} | {s['sharpe']:.2f} | {s['maxdd']:+.1%} |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
