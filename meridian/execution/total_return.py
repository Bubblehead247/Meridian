"""Total-return tracker: what the paper account doesn't pay.

Alpaca paper pays no dividends, coupons or T-bill yield. The C1 core is about
three-quarters bonds, dividend-paying ETFs and T-bills, so its paper equity
would trail its real-account result by a couple of points a year — and look
like a failure when it isn't. After each real session this appends, per market
day since the last row, the income the book's holdings would have earned:

    income = value held at the prior close x (dividend-adjusted return - price return)

using the sleeve position book (which mirrors the broker). The row carries the
account equity and ``tr_equity = equity + cumulative income``: the number to
judge the core by. Append-only; one row per market day.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LOG = Path("ledger") / "shadow" / "total_return.jsonl"


def _rows(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]


def book_shares(positions: dict[str, dict[str, float]]) -> dict[str, float]:
    """Account-level shares per symbol, summed across sleeves."""
    out: dict[str, float] = {}
    for sleeve in positions.values():
        for sym, qty in sleeve.items():
            out[sym] = out.get(sym, 0.0) + float(qty)
    return {s: q for s, q in out.items() if abs(q) > 1e-9}


def income_by_day(shares: dict[str, float], close: pd.DataFrame, adj: pd.DataFrame) -> pd.Series:
    """Daily income the holdings would have earned (index = market days)."""
    cols = [s for s in shares if s in close.columns and s in adj.columns]
    if not cols:
        return pd.Series(dtype=float)
    price_ret = close[cols].pct_change()
    adj_ret = adj[cols].pct_change()
    value = close[cols].shift(1) * pd.Series({s: shares[s] for s in cols})
    return (value * (adj_ret - price_ret)).sum(axis=1, min_count=1).dropna()


def update(account_equity: float | None, shares: dict[str, float], close: pd.DataFrame,
           adj: pd.DataFrame, log: Path = LOG) -> list[dict]:
    """Append a row for every market day after the last logged one; return them."""
    rows = _rows(log)
    last = pd.Timestamp(rows[-1]["date"]) if rows else None
    cum = rows[-1]["cumulative_income"] if rows else 0.0
    income = income_by_day(shares, close, adj)
    if last is not None:
        income = income.loc[income.index > last]
    elif len(income):
        income = income.iloc[-1:]          # a fresh log starts with the latest day only
    new = []
    for day, inc in income.items():
        cum += float(inc)
        new.append({"date": str(day.date()), "income": round(float(inc), 4),
                    "cumulative_income": round(cum, 4),
                    "account_equity": account_equity,
                    "tr_equity": None if account_equity is None else round(account_equity + cum, 2)})
    if new:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
    return new


def run_total_return(account_equity: float | None) -> str:
    """Fetch prices for the book's holdings, log income, return a one-line status."""
    from meridian.data import load_ohlcv
    from meridian.execution.positions import POSITIONS_FILE

    positions = json.loads(Path(POSITIONS_FILE).read_text(encoding="utf-8")) if Path(POSITIONS_FILE).exists() else {}
    shares = book_shares(positions)
    close, adj = {}, {}
    for s in shares:
        try:
            b = load_ohlcv(s, "2026-01-01", use_cache=False)
            close[s], adj[s] = b["close"], b["adj_close"]
        except Exception:
            pass
    new = update(account_equity, shares, pd.DataFrame(close), pd.DataFrame(adj))
    rows = _rows(LOG)
    cum = rows[-1]["cumulative_income"] if rows else 0.0
    return (f"Total return: {len(new)} day(s) logged; income paper didn't pay so far "
            f"${cum:,.2f}" + (f"; total-return equity ${rows[-1]['tr_equity']:,.2f}"
                              if rows and rows[-1]["tr_equity"] is not None else ""))
