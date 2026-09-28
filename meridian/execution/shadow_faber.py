"""Faber GTAA shadow sleeve: logged month-end decisions, no orders.

The pre-registered plan research/plans/faber_gtaa_5etf.json came back
"positive, not significant" (alpha +2.07%/yr, t 1.68), and its committed
consequence is a forward test: run the frozen rule from 2026-10-01, judge it on
2027-09-30. Only the future is truly unseen data, so this is the real test.

The rule (Faber 2006/2007, as registered): at each month-end close, each of
SPY EFA IEF GSG VNQ holds 20% if its total-return price is above the mean of its
last 10 month-end prices, otherwise that 20% is in T-bills.

Each completed month-end's decision is appended once to an append-only log, so
the forward record can't be quietly rewritten. Performance is recomputed from
prices, starting at SHADOW_START, and compared with holding the five ETFs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ETFS = ("SPY", "EFA", "IEF", "GSG", "VNQ")
SMA_MONTHS = 10
SHADOW_START = pd.Timestamp("2026-10-01")
JUDGE_ON = "2027-09-30"
LOG = Path("ledger") / "shadow" / "faber_gtaa.jsonl"


def month_end_targets(prices: pd.DataFrame, n: int = SMA_MONTHS) -> pd.DataFrame:
    """Target weights decided at each month-end close (index = month-end dates)."""
    idx = prices.index
    me = pd.DatetimeIndex(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values)
    monthly = prices.loc[me]
    sma = monthly.rolling(n).mean()
    return ((monthly > sma) & sma.notna()).astype(float) / prices.shape[1]


def _completed_month_ends(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop the current month if it may not be over yet (the last bar is not a month-end we can confirm)."""
    targets = month_end_targets(prices)
    last_bar = prices.index[-1]
    if targets.index[-1] == last_bar:
        # The newest row is the month so far. It is complete only once a bar from the next month exists.
        targets = targets.iloc[:-1]
    return targets


def logged_months(log: Path = LOG) -> set[str]:
    if not log.exists():
        return set()
    return {json.loads(line)["month_end"] for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()}


def update(prices: pd.DataFrame, log: Path = LOG) -> list[dict]:
    """Append every completed month-end decision from SHADOW_START's month on, once each."""
    targets = _completed_month_ends(prices[list(ETFS)].dropna())
    first = SHADOW_START - pd.offsets.MonthBegin(1)  # decision at the end of the month before
    seen = logged_months(log)
    new = []
    for me, row in targets.loc[first:].iterrows():
        key = str(me.date())
        if key in seen:
            continue
        rec = {"month_end": key, "weights": {k: float(v) for k, v in row.items()},
               "tbills": float(1.0 - row.sum()),
               "logged_at": datetime.now(timezone.utc).isoformat()}
        new.append(rec)
    if new:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            for rec in new:
                f.write(json.dumps(rec) + "\n")
    return new


def performance(prices: pd.DataFrame, rf: pd.Series, log: Path = LOG,
                cost_bps: float = 10.0) -> dict | None:
    """Shadow NAV since SHADOW_START from the LOGGED weights, and holding the five ETFs."""
    if not log.exists():
        return None
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return None
    w = pd.DataFrame({pd.Timestamp(r["month_end"]): r["weights"] for r in rows}).T.sort_index()
    df = prices[list(ETFS)].dropna()
    rets = df.pct_change().fillna(0.0)
    daily_w = w.reindex(df.index).ffill().shift(1).loc[SHADOW_START:].fillna(0.0)
    rets = rets.loc[daily_w.index]
    if rets.empty:
        return None
    cash = 1.0 - daily_w.sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).reindex(df.index).shift(1).loc[daily_w.index].fillna(0.0)
    r = (daily_w * rets).sum(axis=1) + cash * rf.reindex(rets.index).fillna(0.0) - turnover * cost_bps / 1e4
    hold = rets.mean(axis=1)
    return {"days": len(r), "shadow_return": float((1 + r).prod() - 1),
            "hold_return": float((1 + hold).prod() - 1), "judge_on": JUDGE_ON,
            "latest_weights": rows[-1]}


def run_shadow() -> str:
    """Fetch prices, log any new month-end decision, and return a one-line status."""
    from meridian.data import load_ohlcv

    px = pd.DataFrame({s: load_ohlcv(s, "2025-01-01", use_cache=False)["adj_close"] for s in ETFS})
    irx = load_ohlcv("^IRX", "2025-01-01", use_cache=False)["close"]
    rf = (irx.reindex(px.index).ffill() / 100 / 252).shift(1).fillna(0.0)
    new = update(px)
    perf = performance(px, rf)
    msg = f"Faber shadow: {len(new)} new month-end decision(s) logged"
    if perf:
        held = [k for k, v in perf["latest_weights"]["weights"].items() if v > 0]
        msg += (f"; since {SHADOW_START.date()}: shadow {perf['shadow_return']:+.2%}, "
                f"hold {perf['hold_return']:+.2%}; holding {', '.join(held) or 'all T-bills'}; "
                f"judged {JUDGE_ON}")
    return msg
