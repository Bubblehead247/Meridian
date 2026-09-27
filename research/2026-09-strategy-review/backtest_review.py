"""Meridian strategy review, 2026-09-27: live picks backtested honestly.

Picks were chosen 2026-06-28 by 10-year backtest return, so everything before
that is in-sample. This measures, per sleeve and for the weighted fund:
  * time in market (the live account is ~40% idle)
  * CAGR / max drawdown at 1, 10, 25 bps per side, with fills at the signal
    close (what build_fund_returns does) and at the next open (what live does)
  * sub-period stability (2011-15, 2016-20, 2021-26/06)
  * benchmarks: SPY, QQQ, 60/40 SPY/IEF, and the same sleeve weights held
    buy-and-hold in each sleeve's own instruments ("same-weights B&H")

    python research/2026-09-strategy-review/backtest_review.py
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

START = "2010-01-01"
SELECTED = pd.Timestamp("2026-06-28")
PERIODS = [("2011-2015", "2011-01-01", "2015-12-31"),
           ("2016-2020", "2016-01-01", "2020-12-31"),
           ("2021-2026/06", "2021-01-01", "2026-06-26"),
           ("full to 2026/06", "2011-01-01", "2026-06-26")]

_bars: dict[str, pd.DataFrame] = {}


def bars(sym: str) -> pd.DataFrame:
    if sym not in _bars:
        _bars[sym] = load_ohlcv(sym, START, use_cache=False)
    return _bars[sym]


def stats(r: pd.Series, a: str, b: str) -> tuple[float, float]:
    r = r.loc[a:b].fillna(0.0)
    # A symbol listed partway through (SNOW 2020, WFRD 2021) would annualize a
    # few months into nonsense; require most of the period.
    if len(r) < 0.8 * len(pd.bdate_range(a, b)):
        return float("nan"), float("nan")
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    return cagr, dd


def sleeve_backtest(family: str, model_name: str, symbol: str, cost: float, next_open: bool):
    model = create_model(family, model_name)
    if getattr(model, "cross_sectional", False):
        syms = _expand_symbol(symbol)
        px, bb = {}, {}
        for s in syms:
            try:
                px[s] = bars(s)["close"]
                bb[s] = bars(s)
            except Exception:
                pass
        bt = model.backtest(px, cost_bps=cost, bars_by_symbol=bb if next_open else None)
        held = bt.gross_exposure  # fraction of the sleeve invested, 0..1
        # Buy-and-hold what the model can actually hold: its signal columns
        # (tradeable_universe also lists vix_band's ^VIX, an index).
        held_syms = list(model.filtered_signals(px).columns)
        bh = pd.DataFrame({s: px[s].pct_change() for s in held_syms}).mean(axis=1)
    else:
        b = bars(symbol)
        # Live is long-only (live_runner: "shorts ... go flat"); Model.backtest
        # would also book rsi_exhaustion's short signals, which never trade.
        from meridian.signals.backtest import backtest as run_bt
        pos = model.signals(b["close"], bars=b).clip(lower=0)
        bt = run_bt(b["close"], pos, cost_bps=cost, bars=b if next_open else None)
        held = bt.positions
        bh = b["close"].pct_change()
    in_mkt = held.abs().clip(upper=1.0)  # average = share of sleeve capital deployed
    return bt.returns, in_mkt, bh


def main() -> int:
    picks = load_live_picks()
    fams = set(picks)
    weights = {f: live_pick_weight(f, fams) for f in picks}
    print("weights:", {f: round(w, 3) for f, w in weights.items()},
          "sum", round(sum(weights.values()), 3))

    rows = []
    fund = {}
    live_window = ("2026-07-31", "2026-09-25")
    for cost in (1.0, 10.0, 25.0):
        for next_open in (False, True):
            key = (cost, next_open)
            combined = None
            bh_combined = None
            for f, p in sorted(picks.items()):
                try:
                    r, in_mkt, bh = sleeve_backtest(f, p["model"], p["symbol"], cost, next_open)
                except Exception as exc:
                    print(f"  {f}: backtest failed: {exc}")
                    continue
                w = weights[f]
                combined = r * w if combined is None else combined.add(r * w, fill_value=0.0)
                bh_combined = (bh * w if bh_combined is None
                               else bh_combined.add(bh * w, fill_value=0.0))
                if cost == 1.0:
                    im = in_mkt.loc["2011-01-01":"2026-06-26"].mean()
                    im_live = in_mkt.loc[live_window[0]:live_window[1]].mean()
                    row = {"sleeve": f, "fills": "next open" if next_open else "signal close",
                           "weight": w, "in_mkt": im, "in_mkt_live_window": im_live,
                           "last_bar": str(r.index[-1])[:10]}
                    for name, a, b in PERIODS:
                        c, d = stats(r, a, b)
                        row[name] = c
                    first = str(r.dropna().index[0])[:10]
                    c, d = stats(r, max(first, "2011-01-01"), "2026-06-26")
                    row["since start"] = c
                    row["start"] = max(first, "2011-01-01")
                    row["maxDD"] = d
                    cb, _ = stats(bh, max(first, "2011-01-01"), "2026-06-26")
                    row["B&H same instr."] = cb
                    rows.append(row)
            fund[key] = combined
            fund[("bh",)] = bh_combined

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    df = pd.DataFrame(rows)
    fmt = {c: "{:+.1%}".format for c in ["2011-2015", "2016-2020", "2021-2026/06",
                                          "full to 2026/06", "since start", "maxDD", "B&H same instr."]}
    fmt.update({"in_mkt": "{:.0%}".format, "in_mkt_live_window": "{:.0%}".format,
                "weight": "{:.3f}".format})
    print("\n=== Per sleeve (1 bps/side) ===")
    print(df.to_string(formatters=fmt, index=False))

    def rets(sym):
        return bars(sym)["close"].pct_change()

    bench = {
        "SPY": rets("SPY"),
        "QQQ": rets("QQQ"),
        "60/40 SPY/IEF": 0.6 * rets("SPY") + 0.4 * rets("IEF"),
        "same-weights B&H (95% invested)": fund[("bh",)],
    }
    out = []
    for (label, r) in bench.items():
        row = {"portfolio": label}
        for name, a, b in PERIODS:
            row[name] = stats(r, a, b)[0]
        row["maxDD"] = stats(r, "2011-01-01", "2026-06-26")[1]
        out.append(row)
    for (cost, nxt), r in fund.items() if False else [(k, v) for k, v in fund.items() if len(k) == 2]:
        row = {"portfolio": f"Meridian live picks, {cost:g} bps, "
                            f"{'next-open' if nxt else 'signal-close'} fills"}
        for name, a, b in PERIODS:
            row[name] = stats(r, a, b)[0]
        row["maxDD"] = stats(r, "2011-01-01", "2026-06-26")[1]
        out.append(row)
    print("\n=== Fund vs benchmarks (CAGR; maxDD over 2011-2026/06) ===")
    print(pd.DataFrame(out).to_string(
        formatters={c: "{:+.1%}".format for c in [p[0] for p in PERIODS] + ["maxDD"]},
        index=False))

    # Backtest over the live window, to compare with the account.
    print("\n=== Live window 2026-06-29 .. last bar (backtest, 1 bps next-open) ===")
    r = fund[(1.0, True)].loc["2026-06-29":]
    print(f"  backtest fund return: {(1 + r.fillna(0)).prod() - 1:+.2%}  "
          f"(last bar {str(r.index[-1])[:10]})")
    for label, s in (("SPY", rets("SPY")), ("QQQ", rets("QQQ")), ("same-weights B&H", fund[("bh",)])):
        s = s.loc["2026-06-29":]
        print(f"  {label:18} {(1 + s.fillna(0)).prod() - 1:+.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
