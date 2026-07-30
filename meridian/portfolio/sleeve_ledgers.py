"""Persist one accounting ledger per sleeve, so sleeve performance is measurable.

``StrategyLedger`` and ``LedgerStore`` already existed but nothing in the live
path ever called them: ``initialize_ledgers`` had no non-test caller and no
per-strategy file was ever written. Sleeve profit, drawdown and win rate existed
only as in-memory objects that were thrown away at the end of each run, so the
only way to answer "how is the momentum sleeve doing?" was to reconstruct it by
hand from the fill log.

This module closes that gap. After each live session it reads the permanent fill
log, matches buys against sells first-in-first-out to get realized profit per
sleeve, marks whatever is still held at the latest price, and saves one JSON
ledger per sleeve.

**Why a subdirectory.** ``LedgerStore.list()`` globs ``*.json`` over its root, and
the default root ``ledger/`` also holds ``positions.json`` and
``pending_orders.json`` — files with a completely different shape. Pointed at
``ledger/`` the store would list those as sleeves and crash trying to load one as
a ``StrategyLedger``. Sleeve ledgers therefore live in ``ledger/sleeves/``.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

from meridian.portfolio.allocation import FAMILY_TO_SLEEVE, SLEEVE_ALLOCATIONS
from meridian.portfolio.ledger import LedgerStore, StrategyLedger

#: Sleeve ledgers live below the shared ledger dir, away from the position book.
SLEEVE_LEDGER_DIR = Path("ledger") / "sleeves"

#: The permanent per-fill log written by execution/reconcile.py.
TRADE_LOG_FILE = Path("ledger") / "trade_log.jsonl"


def sleeve_for(family: str) -> str:
    """Which sleeve funds a family. Families not remapped fund themselves."""
    return FAMILY_TO_SLEEVE.get(family, family)


def read_fills(path: Path | None = None) -> list[dict]:
    """Read the permanent fill log, oldest first. Bad lines are skipped."""
    target = Path(path) if path is not None else TRADE_LOG_FILE
    if not target.exists():
        return []
    out = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def pnl_by_sleeve(
    fills: list[dict],
    prices: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Realized and unrealized profit per sleeve, from the fill log.

    Buys are matched against sells first-in-first-out per (sleeve, symbol), which
    is the convention brokers use for cost basis. Anything left unmatched is an
    open lot and gets marked at ``prices``.

    Args:
        fills: Records from :func:`read_fills`. Each needs ``symbol``, ``qty``
            (signed), ``price`` and ``family``.
        prices: Latest price per symbol, for marking open lots. Symbols with no
            price contribute no unrealized profit rather than a wrong one.

    Returns:
        ``{sleeve: {"realized": float, "unrealized": float, "open_lots": int}}``
    """
    prices = prices or {}
    # (sleeve, symbol) -> deque of [qty, price] still open
    lots: dict[tuple[str, str], deque] = defaultdict(deque)
    realized: dict[str, float] = defaultdict(float)

    for fill in fills:
        symbol = str(fill.get("symbol") or "").upper()
        family = str(fill.get("family") or "")
        sleeve = sleeve_for(family)
        try:
            qty = float(fill.get("qty") or 0.0)
            price = float(fill.get("price") or 0.0)
        except (TypeError, ValueError):
            continue
        if not symbol or qty == 0 or price <= 0:
            continue

        key = (sleeve, symbol)
        if qty > 0:
            lots[key].append([qty, price])
            continue

        # A sell: close the oldest lots first.
        to_close = -qty
        while to_close > 1e-9 and lots[key]:
            lot = lots[key][0]
            matched = min(to_close, lot[0])
            realized[sleeve] += (price - lot[1]) * matched
            lot[0] -= matched
            to_close -= matched
            if lot[0] <= 1e-9:
                lots[key].popleft()
        # Anything left over is a sell with no recorded buy — the fill log starts
        # partway through this account's life, so ignore it rather than inventing
        # a cost basis of zero and booking a fictional profit.

    out: dict[str, dict] = {}
    for sleeve in set(list(realized) + [s for s, _ in lots]):
        unrealized = 0.0
        open_lots = 0
        for (led_sleeve, symbol), remaining in lots.items():
            if led_sleeve != sleeve:
                continue
            last = prices.get(symbol)
            for qty, cost in remaining:
                open_lots += 1
                if last:
                    unrealized += (float(last) - cost) * qty
        out[sleeve] = {
            "realized": round(realized.get(sleeve, 0.0), 6),
            "unrealized": round(unrealized, 6),
            "open_lots": open_lots,
        }
    return out


def update_sleeve_ledgers(
    account_equity: float,
    prices: dict[str, float] | None = None,
    *,
    store: LedgerStore | None = None,
    trade_log_path: Path | None = None,
) -> dict[str, StrategyLedger]:
    """Write one ledger per sleeve, reflecting the fill log as it stands now.

    Idempotent: rerunning recomputes from the same fill log and produces the same
    numbers, so a double session cannot double-count profit. Existing ledgers keep
    their stage and their drawdown peak.

    Returns the ``{sleeve: ledger}`` map that was saved.
    """
    store = store or LedgerStore(SLEEVE_LEDGER_DIR)
    fills = read_fills(trade_log_path)
    measured = pnl_by_sleeve(fills, prices)

    saved: dict[str, StrategyLedger] = {}
    for sleeve, weight in SLEEVE_ALLOCATIONS.items():
        ledger = store.load(sleeve)
        if ledger is None:
            ledger = StrategyLedger(name=sleeve, family=sleeve, stage="paper")
        # Allocation follows equity, so the sleeve's share is always current.
        ledger.capital_alloc = round(weight * float(account_equity), 2)
        if ledger.equity_peak == 0.0:
            ledger.equity_peak = ledger.capital_alloc

        stats = measured.get(sleeve)
        if stats is not None:
            # Set realized directly rather than via record_trade: this is a
            # recomputation of the whole history, not one new closed trade.
            ledger.realized_pnl = stats["realized"]
            ledger.mark(stats["unrealized"])
        else:
            ledger.mark(0.0)

        store.save(ledger)
        saved[sleeve] = ledger
    return saved
