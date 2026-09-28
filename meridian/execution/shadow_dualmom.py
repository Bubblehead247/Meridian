"""Dual momentum pod shadow: logged month-end decisions, no orders.

Pre-registered plan: research/plans/pod_dualmom_forward.json. The full sweep's
in-sample best (DUALMOM6-K4-U8, research/2026-10-sweep/) is a pod candidate,
not a core. Its parameters were picked from 96 on 2008-2026 data, so only
forward data is honest evidence. The first decision is the 2026-09-30 close.

Rule: at each month's last close, rank SPY QQQ IWM EFA EEM GLD IEF TLT by
6-month total return; the top 4 each get 25% if their 6-month return beats
T-bills over the same months, otherwise that slot is in T-bills.

Each completed month-end decision is appended once to an append-only log. The
report is built from the LOGGED weights: pod return, equal-weight hold of the
same 8 ETFs (the benchmark), alpha P&L at the registered beta, trade count
toward the "12 months or 30 trades, whichever is later" rule, and a fidelity
count (logged decisions that no longer match the rule recomputed; must be 0).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ETFS = ("SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "IEF", "TLT")
LOOKBACK_MONTHS = 6
TOP_K = 4
BETA = 0.540          # fixed at registration (plan: pod_dualmom_forward.json)
SHADOW_START = pd.Timestamp("2026-10-01")
JUDGE_ON = "2027-09-30"
MIN_TRADES = 30
LOG = Path("ledger") / "shadow" / "dualmom_6m_top4_u8.jsonl"


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby([index.year, index.month]).max().values)


def targets(month_px: pd.DataFrame, tbill_index: pd.Series) -> pd.DataFrame:
    """Weights decided at each month-end: top K by 6-month return, each 1/K if it beats T-bills."""
    r = month_px / month_px.shift(LOOKBACK_MONTHS) - 1
    hurdle = tbill_index / tbill_index.shift(LOOKBACK_MONTHS) - 1
    top = r.rank(axis=1, ascending=False, method="first") <= TOP_K
    return (top & r.gt(hurdle, axis=0)).astype(float) / TOP_K


def _completed_targets(prices: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    px = prices[list(ETFS)].dropna()
    ends = month_ends(px.index)
    if len(ends) and ends[-1] == px.index[-1]:
        ends = ends[:-1]      # the newest month is complete only once a later bar exists
    tbill_index = (1 + rf.reindex(px.index).fillna(0.0)).cumprod()
    return targets(px.loc[ends], tbill_index.loc[ends])


def _rows(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]


def update(prices: pd.DataFrame, rf: pd.Series, log: Path = LOG) -> list[dict]:
    """Append each completed month-end decision from SHADOW_START's month on, once each."""
    tg = _completed_targets(prices, rf)
    first = SHADOW_START - pd.offsets.MonthBegin(1)
    seen = {r["month_end"] for r in _rows(log)}
    new = []
    for me, row in tg.loc[first:].iterrows():
        key = str(me.date())
        if key in seen:
            continue
        new.append({"month_end": key, "weights": {k: float(v) for k, v in row.items()},
                    "tbills": float(1.0 - row.sum()), "logged_at": datetime.now(timezone.utc).isoformat()})
    if new:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            for rec in new:
                f.write(json.dumps(rec) + "\n")
    return new


def fidelity_mismatches(prices: pd.DataFrame, rf: pd.Series, log: Path = LOG) -> int:
    """Logged decisions that differ from the rule recomputed on the same month-end data."""
    tg = _completed_targets(prices, rf)
    bad = 0
    for r in _rows(log):
        me = pd.Timestamp(r["month_end"])
        if me in tg.index and any(abs(tg.loc[me, k] - v) > 1e-9 for k, v in r["weights"].items()):
            bad += 1
    return bad


def performance(prices: pd.DataFrame, rf: pd.Series, log: Path = LOG, cost_bps: float = 10.0) -> dict | None:
    """Pod vs benchmark since SHADOW_START, from the LOGGED weights."""
    rows = _rows(log)
    if not rows:
        return None
    w = pd.DataFrame({pd.Timestamp(r["month_end"]): r["weights"] for r in rows}).T.sort_index()
    px = prices[list(ETFS)].dropna()
    ret = px.pct_change().fillna(0.0)
    daily_w = w.reindex(px.index).ffill().shift(1).loc[SHADOW_START:].fillna(0.0)
    if daily_w.empty:
        return {"days": 0, "trades": 0, "latest": rows[-1]}
    ret = ret.loc[daily_w.index]
    t = rf.reindex(ret.index).fillna(0.0)
    turn = w.diff().abs().sum(axis=1).reindex(px.index).shift(1).loc[daily_w.index].fillna(0.0)
    pod = (daily_w * ret).sum(axis=1) + (1 - daily_w.sum(axis=1)) * t - turn * cost_bps / 1e4
    bench = ret.mean(axis=1)
    held = (w > 0).astype(int)
    trades = int(held.diff().abs().sum().sum())
    return {"days": len(pod), "pod_return": float((1 + pod).prod() - 1),
            "bench_return": float((1 + bench).prod() - 1),
            "alpha_pnl": float(((pod - t) - BETA * (bench - t)).sum()),
            "trades": trades, "latest": rows[-1]}


def run_shadow() -> str:
    """Fetch prices, log new month-end decisions, return a one-line status."""
    from meridian.data import load_ohlcv

    px = pd.DataFrame({s: load_ohlcv(s, "2025-01-01", use_cache=False)["adj_close"] for s in ETFS})
    irx = load_ohlcv("^IRX", "2025-01-01", use_cache=False)["close"]
    rf = (irx.reindex(px.index).ffill() / 100 / 252).shift(1).fillna(0.0)
    new = update(px, rf)
    perf = performance(px, rf)
    bad = fidelity_mismatches(px, rf)
    msg = f"Dual momentum shadow: {len(new)} new month-end decision(s) logged; fidelity mismatches {bad}"
    if perf and perf["days"]:
        held = [k for k, v in perf["latest"]["weights"].items() if v > 0]
        msg += (f"; since {SHADOW_START.date()}: pod {perf['pod_return']:+.2%}, hold {perf['bench_return']:+.2%}, "
                f"alpha P&L {perf['alpha_pnl']:+.2%}, trades {perf['trades']}/{MIN_TRADES}; "
                f"holding {', '.join(held) or 'all T-bills'}; judged {JUDGE_ON} or at {MIN_TRADES} trades")
    return msg
