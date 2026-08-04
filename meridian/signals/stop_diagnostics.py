"""Intrabar stop/gap diagnostics — visibility, not a change to executed P&L.

The signal engine (``signals/engine.py``) evaluates entries, exits, and stops
once per bar, from that bar's close-derived deviation score — by design, the
whole platform shares one causal, daily-close-driven rule (see that module's
docstring). That means a real intrabar move that would have triggered an exit
sooner is only caught at the next bar's close, one (or more) bars later than
a live intraday order would have caught it.

This module does not change execution, sizing, or the backtested P&L. It
answers, after the fact: for each recorded trade, did any bar's high/low
between entry and the recorded exit already reach the eventual exit price
before the exit bar itself? If so, the close-driven signal caught the move
late — how late, in bars. Purely diagnostic, driven only by the trade ledger
and OHLC bars; it never touches the estimator's online-update loop or the
recorded fill prices/returns.
"""

from __future__ import annotations

import pandas as pd


def flag_intrabar_stop_breaches(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Flag trades where price already reached the exit level before the exit bar.

    Args:
        trades: A backtester's trade ledger (``BacktestResult.trades``) — needs
            ``direction``, ``entry_time``, ``exit_time``, ``exit_price``.
        bars: OHLC frame aligned to the same index the trades were built from;
            needs ``high``/``low`` columns.

    Returns:
        ``trades`` with two added columns: ``intrabar_stop_breach`` (bool) —
        whether any bar strictly between entry and exit had its low (for a
        long) or high (for a short) reach the eventual exit price — and
        ``bars_late`` (int) — how many bars separated that first breach from
        the recorded exit (0 when no breach is flagged).
    """
    out = trades.copy()
    if trades.empty or bars is None or not {"high", "low"}.issubset(bars.columns):
        out["intrabar_stop_breach"] = pd.Series([False] * len(trades), dtype=bool, index=trades.index)
        out["bars_late"] = pd.Series([0] * len(trades), dtype=int, index=trades.index)
        return out

    idx = bars.index
    breach_flags: list[bool] = []
    bars_late: list[int] = []
    for _, t in trades.iterrows():
        entry_time, exit_time = t["entry_time"], t["exit_time"]
        direction, exit_price = t["direction"], t["exit_price"]
        span = bars.loc[(idx > entry_time) & (idx < exit_time)]
        if span.empty:
            breach_flags.append(False)
            bars_late.append(0)
            continue
        breach_mask = (span["low"] <= exit_price) if direction == 1 else (span["high"] >= exit_price)
        if breach_mask.any():
            first_breach_pos = int(breach_mask.to_numpy().argmax())
            breach_flags.append(True)
            bars_late.append(len(span) - first_breach_pos)
        else:
            breach_flags.append(False)
            bars_late.append(0)

    out["intrabar_stop_breach"] = pd.Series(breach_flags, index=trades.index)
    out["bars_late"] = pd.Series(bars_late, index=trades.index)
    return out
